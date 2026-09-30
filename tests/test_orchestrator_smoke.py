"""Tests for orchestrator/ — the local, model-agnostic agent harness.

Exercises the full agent loop (system prompt -> LLM call -> action parse ->
tool dispatch -> observation -> ...) against an in-process stub HTTP server
standing in for an OpenAI-compatible model endpoint, so these tests need no
real local model to run. Also covers the sqlite store, the tool
allowlist/sandboxing, and fine-tune export directly.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator import agent, db, finetune, llm, tools, ui  # noqa: E402
from orchestrator.config import Config, load_config  # noqa: E402


# --- a stub OpenAI-compatible server ---------------------------------------

def make_stub_server(script: list[str]):
    """Serves POST /v1/chat/completions, returning `script[i]` as the
    assistant content on the i-th call, then repeating the last entry."""
    state = {"calls": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            _ = self.rfile.read(n)  # request body unused by the stub
            with lock:
                i = min(state["calls"], len(script) - 1)
                state["calls"] += 1
                content = script[i]
            body = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": content}}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


def action(obj: dict) -> str:
    return f"```action\n{json.dumps(obj)}\n```"


def stop_server(server: HTTPServer) -> None:
    server.shutdown()
    server.server_close()


class OrchestratorTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo_root = Path(self.tmp.name) / "repo"
        skills_dir = self.repo_root / "plugin" / "skills" / "demo-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: a fake skill for tests\n---\n\n# Demo\n\nDo the thing.\n",
            encoding="utf-8",
        )
        (self.repo_root / "README.md").write_text("hello from the fake repo\n", encoding="utf-8")
        self.db_path = Path(self.tmp.name) / "orchestrator.db"

    def make_config(self, base_url: str, **overrides) -> Config:
        cfg = Config(
            base_url=base_url,
            api_key="EMPTY",
            model="stub-model",
            temperature=0.0,
            max_tokens=256,
            max_steps=6,
            max_consecutive_errors=3,
            request_timeout=10,
            shell_timeout=30,
            repo_root=self.repo_root,
            db_path=self.db_path,
            finetune_dir=Path(self.tmp.name) / "finetune",
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


class TestDb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.con = db.connect(Path(self.tmp.name) / "t.db")
        self.addCleanup(self.con.close)

    def test_session_and_message_roundtrip(self):
        sid = db.create_session(self.con, "do a thing", "m", "http://x", finetune=True)
        db.add_message(self.con, sid, "system", "sys prompt")
        db.add_message(self.con, sid, "user", "do a thing")
        db.add_message(self.con, sid, "tool", "output here", tool_name="run_shell")
        n = db.bump_step(self.con, sid)
        self.assertEqual(n, 1)

        session = db.get_session(self.con, sid)
        self.assertEqual(session["status"], "running")
        self.assertEqual(session["finetune"], 1)
        self.assertEqual(session["step_count"], 1)

        db.end_session(self.con, sid, "completed", summary="all done")
        session = db.get_session(self.con, sid)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["summary"], "all done")

        rows = db.get_messages(self.con, sid)
        self.assertEqual([r["role"] for r in rows], ["system", "user", "tool"])
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3])

        api_msgs = db.to_api_messages(rows)
        self.assertEqual(api_msgs[2]["role"], "user")
        self.assertIn("Observation (run_shell)", api_msgs[2]["content"])

        listed = db.list_sessions(self.con)
        self.assertEqual(listed[0]["id"], sid)

    def test_memory_roundtrip(self):
        self.assertIsNone(db.memory_get(self.con, "missing"))
        db.memory_set(self.con, "foo", {"a": 1})
        self.assertEqual(db.memory_get(self.con, "foo"), {"a": 1})
        db.memory_set(self.con, "bar", "plain string")
        self.assertEqual(db.memory_list(self.con), {"bar": "plain string", "foo": {"a": 1}})

    def test_delete_session_removes_rows_and_reports_but_does_not_touch_images(self):
        sid = db.create_session(self.con, "do a thing", "m", "http://x")
        db.add_message(self.con, sid, "system", "sys prompt")
        db.add_message(self.con, sid, "tool", "saw something", tool_name="screenshot",
                        image_path="/fake/path/shot.png")

        result = db.delete_session(self.con, sid)
        self.assertEqual(result["session_id"], sid)
        self.assertEqual(result["message_count"], 2)
        self.assertEqual(result["image_paths"], ["/fake/path/shot.png"])

        self.assertIsNone(db.get_session(self.con, sid))
        self.assertEqual(db.get_messages(self.con, sid), [])

    def test_delete_session_returns_none_for_unknown_session(self):
        self.assertIsNone(db.delete_session(self.con, "no-such-session"))

    def test_set_review_then_clear(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        session = db.get_session(self.con, sid)
        self.assertEqual(session["reviewed"], 0)
        self.assertIsNone(session["reward"])

        self.assertTrue(db.set_review(self.con, sid, 2, "great recovery from an error"))
        session = db.get_session(self.con, sid)
        self.assertEqual(session["reviewed"], 1)
        self.assertEqual(session["reward"], 2)
        self.assertEqual(session["review_note"], "great recovery from an error")
        self.assertIsNotNone(session["reviewed_at"])

        self.assertTrue(db.clear_review(self.con, sid))
        session = db.get_session(self.con, sid)
        self.assertEqual(session["reviewed"], 0)
        self.assertIsNone(session["reward"])
        self.assertIsNone(session["review_note"])

    def test_set_review_rejects_out_of_range_reward(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        with self.assertRaises(ValueError):
            db.set_review(self.con, sid, 5)

    def test_set_review_returns_false_for_unknown_session(self):
        self.assertFalse(db.set_review(self.con, "no-such-session", 1))
        self.assertFalse(db.clear_review(self.con, "no-such-session"))

    def test_set_message_review_then_clear(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        seq = db.add_message(self.con, sid, "assistant", "```action\n{}\n```")

        row = db.get_messages(self.con, sid)[0]
        self.assertIsNone(row["reward"])

        self.assertTrue(db.set_message_review(self.con, sid, seq, -1, "wrong tool choice"))
        row = db.get_messages(self.con, sid)[0]
        self.assertEqual(row["reward"], -1)
        self.assertEqual(row["review_note"], "wrong tool choice")

        self.assertTrue(db.clear_message_review(self.con, sid, seq))
        row = db.get_messages(self.con, sid)[0]
        self.assertIsNone(row["reward"])
        self.assertIsNone(row["review_note"])

    def test_set_message_review_rejects_out_of_range_reward(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        seq = db.add_message(self.con, sid, "assistant", "x")
        with self.assertRaises(ValueError):
            db.set_message_review(self.con, sid, seq, 99)

    def test_create_session_persists_effort(self):
        sid = db.create_session(self.con, "t", "m", "http://x", effort="high")
        self.assertEqual(db.get_session(self.con, sid)["effort"], "high")

        sid2 = db.create_session(self.con, "t", "m", "http://x")
        self.assertIsNone(db.get_session(self.con, sid2)["effort"])

    def test_create_session_rejects_invalid_effort(self):
        with self.assertRaises(ValueError):
            db.create_session(self.con, "t", "m", "http://x", effort="ludicrous")

    def test_set_message_review_returns_false_for_unknown_step(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        self.assertFalse(db.set_message_review(self.con, sid, 999, 1))
        self.assertFalse(db.clear_message_review(self.con, sid, 999))

    def test_set_message_review_rejects_non_assistant_rows(self):
        # Only an assistant turn is an actual decision point -- a
        # system/user/tool row is context the model was given, not
        # something it decided, so there's nothing to score.
        sid = db.create_session(self.con, "t", "m", "http://x")
        system_seq = db.add_message(self.con, sid, "system", "sys prompt")
        user_seq = db.add_message(self.con, sid, "user", "do it")
        tool_seq = db.add_message(self.con, sid, "tool", "output", tool_name="run_shell")
        for seq in (system_seq, user_seq, tool_seq):
            with self.assertRaises(ValueError):
                db.set_message_review(self.con, sid, seq, 1)
            # rejection must not have partially written anything
            self.assertIsNone(db.get_messages(self.con, sid)[seq - 1]["reward"])


class TestTools(OrchestratorTestBase):
    def make_ctx(self, cfg=None):
        cfg = cfg or self.make_config("http://unused/v1")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        sid = db.create_session(con, "t", "m", cfg.base_url)
        return tools.ToolContext(repo_root=self.repo_root, con=con, session_id=sid, config=cfg)

    def test_shell_allowlist(self):
        self.assertTrue(tools.is_shell_allowed("python -m jobagent report daily"))
        self.assertTrue(tools.is_shell_allowed("git status"))
        self.assertFalse(tools.is_shell_allowed("rm -rf /"))
        self.assertFalse(tools.is_shell_allowed("python -m pip install evil"))
        self.assertFalse(tools.is_shell_allowed(""))

    def test_run_shell_rejects_disallowed_command(self):
        ctx = self.make_ctx()
        out = tools.tool_run_shell(ctx, {"command": "rm -rf /"})
        self.assertTrue(out.startswith("ERROR: command rejected"))

    def test_run_shell_executes_allowed_command(self):
        ctx = self.make_ctx()
        out = tools.tool_run_shell(ctx, {"command": "git status"})
        self.assertIn("exit_code:", out)
        self.assertFalse(out.startswith("ERROR: command rejected"))

    def test_read_file_blocks_path_traversal(self):
        ctx = self.make_ctx()
        out = tools.tool_read_file(ctx, {"path": "../outside.txt"})
        self.assertIn("outside the project root", out)

    def test_read_file_reads_inside_root(self):
        ctx = self.make_ctx()
        out = tools.tool_read_file(ctx, {"path": "README.md"})
        self.assertIn("hello from the fake repo", out)

    def test_list_and_read_skill(self):
        ctx = self.make_ctx()
        listing = tools.tool_list_skills(ctx, {})
        self.assertIn("demo-skill", listing)
        self.assertIn("a fake skill for tests", listing)

        content = tools.tool_read_skill(ctx, {"name": "demo-skill"})
        self.assertIn("Do the thing.", content)

        missing = tools.tool_read_skill(ctx, {"name": "nope"})
        self.assertTrue(missing.startswith("ERROR"))

    def test_memory_tool(self):
        ctx = self.make_ctx()
        self.assertEqual(tools.tool_memory(ctx, {"action": "set", "key": "k", "value": "v"}), "OK: remembered 'k'.")
        self.assertEqual(tools.tool_memory(ctx, {"action": "get", "key": "k"}), "v")
        self.assertIn("k: v", tools.tool_memory(ctx, {"action": "list"}))


class TestAgentLoop(OrchestratorTestBase):
    def test_full_loop_completes(self):
        server, state = make_stub_server([
            action({"tool": "memory", "input": {"action": "set", "key": "foo", "value": "bar"}}),
            action({"tool": "list_skills", "input": {}}),
            action({"final": "Done. Found demo-skill."}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")

        events = []
        session_id = agent.run_session(cfg, "test task", on_event=lambda k, p: events.append((k, p)))

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")
        self.assertIn("demo-skill", session["summary"])
        self.assertEqual(db.memory_get(con, "foo"), "bar")

        rows = db.get_messages(con, session_id)
        roles = [r["role"] for r in rows]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertEqual([k for k, _ in events][-1], "session_completed")

    def test_protocol_violation_then_recovery(self):
        server, state = make_stub_server([
            "I am thinking out loud with no action block.",
            action({"final": "Recovered."}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")

        session_id = agent.run_session(cfg, "test task")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["summary"], "Recovered.")

        rows = db.get_messages(con, session_id)
        protocol_errors = [r for r in rows if r["tool_name"] == "_protocol"]
        self.assertEqual(len(protocol_errors), 1)

    def test_aborts_after_repeated_protocol_violations(self):
        server, state = make_stub_server(["no action block, ever"])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", max_consecutive_errors=2, max_steps=10)

        session_id = agent.run_session(cfg, "test task")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "aborted")
        self.assertIn("protocol", session["error"])

    def test_unknown_tool_is_reported_not_fatal(self):
        server, state = make_stub_server([
            action({"tool": "does_not_exist", "input": {}}),
            action({"final": "Gave up on the bad tool."}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")

        session_id = agent.run_session(cfg, "test task")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")
        rows = db.get_messages(con, session_id)
        tool_errors = [r for r in rows if r["role"] == "tool" and "unknown tool" in r["content"]]
        self.assertEqual(len(tool_errors), 1)


class TestFinetuneExport(OrchestratorTestBase):
    def _completed_session(self, flagged: bool) -> tuple[str, Path]:
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t", finetune_flag=flagged)
        return session_id, cfg.finetune_dir

    def test_export_session_writes_valid_record(self):
        session_id, finetune_dir = self._completed_session(flagged=True)
        out_path = finetune_dir / f"{session_id}.jsonl"
        self.assertTrue(out_path.exists())  # auto-exported by drive_session since finetune=True
        record = json.loads(out_path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["schema"], "orchestrator-sft/1")
        self.assertEqual(record["session_id"], session_id)
        self.assertEqual(record["messages"][0]["role"], "system")
        self.assertEqual(record["messages"][-1]["role"], "assistant")

    def test_export_all_respects_flag_unless_all(self):
        flagged_id, _ = self._completed_session(flagged=True)
        unflagged_id, _ = self._completed_session(flagged=False)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)

        only_flagged_out = Path(self.tmp.name) / "flagged.jsonl"
        report = finetune.export_all(con, only_flagged_out, include_all=False)
        self.assertEqual(report["count"], 1)
        self.assertEqual(report["train_count"], 1)
        self.assertEqual(report["val_count"], 0)
        self.assertEqual(report["skipped_invalid"], [])
        lines = only_flagged_out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["session_id"], flagged_id)

        all_out = Path(self.tmp.name) / "all.jsonl"
        report_all = finetune.export_all(con, all_out, include_all=True)
        self.assertEqual(report_all["count"], 2)

    def test_export_all_require_review_excludes_unreviewed(self):
        reviewed_id, _ = self._completed_session(flagged=True)
        unreviewed_id, _ = self._completed_session(flagged=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        db.set_review(con, reviewed_id, 1, "good")

        out = Path(self.tmp.name) / "reviewed_only.jsonl"
        report = finetune.export_all(con, out, include_all=True, require_review=True)
        self.assertEqual(report["count"], 1)
        self.assertIn(unreviewed_id, report["skipped_unreviewed"])
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["session_id"], reviewed_id)

    def test_export_all_min_reward_excludes_unreviewed_and_low_reward(self):
        # The bug this guards against: an unreviewed session's reward is
        # NULL, never 0, so a naive `reward >= min_reward` filter must not
        # let it through -- same class of bug finetune/prepare_data.py's
        # own comment calls out for the harvested corpus.
        good_id, _ = self._completed_session(flagged=True)
        bad_id, _ = self._completed_session(flagged=True)
        unreviewed_id, _ = self._completed_session(flagged=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        db.set_review(con, good_id, 2)
        db.set_review(con, bad_id, -1)

        out = Path(self.tmp.name) / "min_reward.jsonl"
        report = finetune.export_all(con, out, include_all=True, min_reward=1)
        self.assertEqual(report["count"], 1)
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["session_id"], good_id)
        self.assertIn(bad_id, report["skipped_unreviewed"])
        self.assertIn(unreviewed_id, report["skipped_unreviewed"])

    def test_export_all_skips_non_completed_status_by_default(self):
        server, state = make_stub_server(["no action block, ever"])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", max_consecutive_errors=1, max_steps=5)
        aborted_id = agent.run_session(cfg, "t", finetune_flag=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, aborted_id)["status"], "aborted")

        out = Path(self.tmp.name) / "completed_only.jsonl"
        report = finetune.export_all(con, out, include_all=True)
        self.assertEqual(report["count"], 0)

        out_any = Path(self.tmp.name) / "any_status.jsonl"
        report_any = finetune.export_all(con, out_any, include_all=True, statuses=None)
        self.assertEqual(report_any["count"], 1)

    def test_export_all_reports_trajectory_mix(self):
        session_id, _ = self._completed_session(flagged=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        out = Path(self.tmp.name) / "mix.jsonl"
        report = finetune.export_all(con, out, include_all=True)
        self.assertEqual(report["by_category"].get("no_tool"), 1)
        raw_rows = db.get_messages(con, session_id)
        self.assertEqual(finetune.validate_trajectory(raw_rows), [])
        self.assertEqual(finetune.classify_trajectory(raw_rows)["no_tool"], True)

    def test_export_all_val_frac_splits_by_task_without_leakage(self):
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        for i in range(6):
            server, state = make_stub_server([action({"final": "ok"})])
            self.addCleanup(stop_server, server)
            port = server.server_address[1]
            cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
            # Two sessions per task string, so a leaking split would put the
            # same task on both sides.
            agent.run_session(cfg, f"task {i // 2}", finetune_flag=True)

        out = Path(self.tmp.name) / "split.jsonl"
        report = finetune.export_all(con, out, include_all=False, val_frac=0.3)
        self.assertGreater(report["val_count"], 0)
        self.assertEqual(report["train_count"] + report["val_count"], report["count"])
        train_tasks = {json.loads(l)["task"] for l in out.read_text(encoding="utf-8").splitlines()}
        val_path = Path(report["val_path"])
        val_tasks = {json.loads(l)["task"] for l in val_path.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(train_tasks & val_tasks, set())

    def test_strip_metadata_writes_bare_messages_plus_sidecar(self):
        session_id, _ = self._completed_session(flagged=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        out = Path(self.tmp.name) / "stripped.jsonl"
        report = finetune.export_all(con, out, include_all=True, strip_metadata=True)

        line = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(line), {"messages"})

        self.assertIsNotNone(report["train_meta_path"])
        meta_line = json.loads(Path(report["train_meta_path"]).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(meta_line["session_id"], session_id)
        self.assertEqual(meta_line["line"], 0)
        self.assertNotIn("messages", meta_line)

    def test_validate_export_file_flags_non_final_ending(self):
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        out = Path(self.tmp.name) / "bad.jsonl"
        out.write_text(json.dumps({
            "schema": "orchestrator-sft/1", "session_id": "x",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": "no action block here"},
            ],
        }) + "\n", encoding="utf-8")
        report = finetune.validate_export_file(out)
        self.assertEqual(report["invalid"], 1)
        self.assertIn("final", report["problems"][0]["errors"][0])


class TestImport(OrchestratorTestBase):
    def test_round_trip_export_then_import_reconstructs_tool_rows(self):
        # Real fidelity check: export a session with an actual tool call,
        # delete it, re-import the exported line, and confirm the 'tool'
        # role and tool_name come back -- not just plain 'user' text.
        server, state = make_stub_server([
            action({"tool": "list_skills", "input": {}}),
            action({"final": "done"}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t", finetune_flag=True)

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        original_rows = db.get_messages(con, session_id)
        record = finetune._session_record(con, session_id)

        db.delete_session(con, session_id)
        self.assertIsNone(db.get_session(con, session_id))

        image_dir = Path(self.tmp.name) / "imported_images"
        report = finetune.import_records(con, [record], image_dir)
        self.assertEqual(report["imported"], [session_id])
        self.assertEqual(report["skipped"], [])

        restored_rows = db.get_messages(con, session_id)
        self.assertEqual([r["role"] for r in restored_rows], [r["role"] for r in original_rows])
        tool_row = next(r for r in restored_rows if r["role"] == "tool")
        self.assertEqual(tool_row["tool_name"], "list_skills")
        self.assertEqual(finetune.validate_trajectory(restored_rows), [])

    def test_round_trip_preserves_step_reviews(self):
        server, state = make_stub_server([action({"final": "done"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t", finetune_flag=True)

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        final_row = next(r for r in db.get_messages(con, session_id) if r["role"] == "assistant")
        db.set_message_review(con, session_id, final_row["seq"], 2, "clean final answer")

        record = finetune._session_record(con, session_id)
        self.assertEqual(record["step_reviews"], [
            {"seq": final_row["seq"], "role": "assistant", "reward": 2, "note": "clean final answer"},
        ])

        db.delete_session(con, session_id)
        image_dir = Path(self.tmp.name) / "imported_images"
        report = finetune.import_records(con, [record], image_dir)
        self.assertEqual(report["imported"], [session_id])

        restored = db.get_messages(con, session_id)
        restored_final = next(r for r in restored if r["role"] == "assistant")
        self.assertEqual(restored_final["reward"], 2)
        self.assertEqual(restored_final["review_note"], "clean final answer")
        # Rows with no step review still come back with a NULL reward, not 0.
        self.assertIsNone(next(r for r in restored if r["role"] == "system")["reward"])

    def test_round_trip_preserves_effort(self):
        server, state = make_stub_server([action({"final": "done"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", effort="high")
        session_id = agent.run_session(cfg, "t", finetune_flag=True)

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        record = finetune._session_record(con, session_id)
        self.assertEqual(record["effort"], "high")

        db.delete_session(con, session_id)
        image_dir = Path(self.tmp.name) / "imported_images"
        report = finetune.import_records(con, [record], image_dir)
        self.assertEqual(report["imported"], [session_id])
        self.assertEqual(db.get_session(con, session_id)["effort"], "high")

    def test_round_trip_preserves_provider(self):
        # A pure data round-trip check -- built directly via db.* rather
        # than a live run, since a real "anthropic" session needs an
        # Anthropic-shaped stub (see TestAnthropicClient), not the
        # OpenAI-shaped one make_stub_server gives every other test here.
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session_id = db.create_session(con, "t", "claude-x", "https://api.anthropic.com",
                                        finetune=True, provider="anthropic")
        db.add_message(con, session_id, "system", "sys")
        db.add_message(con, session_id, "user", "t")
        db.add_message(con, session_id, "assistant", action({"final": "done"}))
        db.end_session(con, session_id, "completed", summary="done")

        self.assertEqual(db.get_session(con, session_id)["provider"], "anthropic")
        record = finetune._session_record(con, session_id)
        self.assertEqual(record["provider"], "anthropic")

        db.delete_session(con, session_id)
        image_dir = Path(self.tmp.name) / "imported_images"
        report = finetune.import_records(con, [record], image_dir)
        self.assertEqual(report["imported"], [session_id])
        self.assertEqual(db.get_session(con, session_id)["provider"], "anthropic")

    def test_duplicate_session_id_skipped_unless_overwrite(self):
        session_id, _ = self._completed_session(flagged=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        record = finetune._session_record(con, session_id)
        image_dir = Path(self.tmp.name) / "imported_images"

        result = finetune.import_record(con, record, image_dir)
        self.assertFalse(result["imported"])
        self.assertIn("already exists", result["reason"])

        result = finetune.import_record(con, record, image_dir, overwrite=True)
        self.assertTrue(result["imported"])

    def test_missing_messages_field_is_rejected(self):
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        result = finetune.import_record(con, {"session_id": "x"}, Path(self.tmp.name))
        self.assertFalse(result["imported"])
        self.assertIn("messages", result["reason"])

    def test_structurally_invalid_record_is_rejected(self):
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        result = finetune.import_record(con, {
            "session_id": "bad1", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "no action block at all"},
            ],
        }, Path(self.tmp.name))
        self.assertFalse(result["imported"])
        self.assertIsNone(db.get_session(con, "bad1"))

    def test_import_lines_skips_bad_json_and_reports_it(self):
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        good = json.dumps({"session_id": "good1", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": action({"final": "ok"})},
        ]})
        lines = ["not json at all", "", good]
        report = finetune.import_lines(con, lines, Path(self.tmp.name))
        self.assertEqual(report["imported"], ["good1"])
        self.assertEqual(len(report["skipped"]), 1)
        self.assertIn("invalid JSON", report["skipped"][0]["reason"])

    def test_bare_messages_only_record_synthesizes_a_session_id(self):
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        result = finetune.import_record(con, {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": action({"final": "ok"})},
        ]}, Path(self.tmp.name))
        self.assertTrue(result["imported"])
        self.assertIsNotNone(result["session_id"])
        session = db.get_session(con, result["session_id"])
        self.assertEqual(session["task"], "hi")
        self.assertEqual(session["finetune"], 1)  # mark_finetune defaults True

    def _completed_session(self, flagged: bool) -> tuple[str, Path]:
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t", finetune_flag=flagged)
        return session_id, cfg.finetune_dir


class TestDbImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.con = db.connect(Path(self.tmp.name) / "t.db")
        self.addCleanup(self.con.close)

    def test_to_api_messages_embeds_image_as_data_uri(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        png_path = Path(self.tmp.name) / "shot.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n fake bytes")
        db.add_message(self.con, sid, "tool", "saw something", tool_name="screenshot", image_path=str(png_path))

        rows = db.get_messages(self.con, sid)
        self.assertEqual(rows[0]["image_path"], str(png_path))

        api_msgs = db.to_api_messages(rows)
        self.assertIsInstance(api_msgs[0]["content"], list)
        text_block = next(b for b in api_msgs[0]["content"] if b["type"] == "text")
        image_block = next(b for b in api_msgs[0]["content"] if b["type"] == "image_url")
        self.assertIn("saw something", text_block["text"])
        self.assertTrue(image_block["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_include_images_false_falls_back_to_text(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        png_path = Path(self.tmp.name) / "shot.png"
        png_path.write_bytes(b"fake")
        db.add_message(self.con, sid, "tool", "saw something", tool_name="screenshot", image_path=str(png_path))

        rows = db.get_messages(self.con, sid)
        api_msgs = db.to_api_messages(rows, include_images=False)
        self.assertIsInstance(api_msgs[0]["content"], str)
        self.assertIn("saw something", api_msgs[0]["content"])
        self.assertIn("screenshot omitted", api_msgs[0]["content"])

    def test_missing_image_file_degrades_to_text(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        db.add_message(self.con, sid, "tool", "saw something", tool_name="screenshot",
                        image_path=str(Path(self.tmp.name) / "does_not_exist.png"))

        rows = db.get_messages(self.con, sid)
        api_msgs = db.to_api_messages(rows)
        self.assertIsInstance(api_msgs[0]["content"], str)
        self.assertIn("missing", api_msgs[0]["content"])

    def test_ordinary_tool_message_unaffected(self):
        sid = db.create_session(self.con, "t", "m", "http://x")
        db.add_message(self.con, sid, "tool", "plain text output", tool_name="list_skills")
        rows = db.get_messages(self.con, sid)
        self.assertIsNone(rows[0]["image_path"])
        api_msgs = db.to_api_messages(rows)
        self.assertIsInstance(api_msgs[0]["content"], str)


class TestVisionConfig(unittest.TestCase):
    def test_vision_defaults_off(self):
        cfg = Config(
            base_url="x", api_key="k", model="m", temperature=0.0, max_tokens=1,
            max_steps=1, max_consecutive_errors=1, request_timeout=1, shell_timeout=1,
            repo_root=Path("."),
        )
        self.assertFalse(cfg.vision)

    def test_env_var_enables_vision(self):
        import os

        from orchestrator.config import load_config

        old = os.environ.get("ORCHESTRATOR_VISION")
        os.environ["ORCHESTRATOR_VISION"] = "true"
        try:
            self.assertTrue(load_config().vision)
        finally:
            if old is None:
                os.environ.pop("ORCHESTRATOR_VISION", None)
            else:
                os.environ["ORCHESTRATOR_VISION"] = old

    def test_with_overrides_sets_vision_without_mutating_original(self):
        from orchestrator.config import load_config, with_overrides

        cfg = load_config()
        cfg2 = with_overrides(cfg, vision=True)
        self.assertTrue(cfg2.vision)
        self.assertFalse(cfg.vision)


class TestToolAvailability(OrchestratorTestBase):
    def test_screenshot_only_advertised_when_vision_enabled(self):
        cfg_off = self.make_config("http://unused/v1", vision=False)
        cfg_on = self.make_config("http://unused/v1", vision=True)

        self.assertNotIn("screenshot", tools.available_tools(cfg_off))
        self.assertIn("screenshot", tools.available_tools(cfg_on))
        self.assertNotIn("screenshot", tools.tool_help_text(cfg_off))
        self.assertIn("screenshot", tools.tool_help_text(cfg_on))

        self.assertNotIn("screenshot", agent.build_system_prompt(cfg_off))
        self.assertIn("screenshot", agent.build_system_prompt(cfg_on))

    def test_browse_rule_flips_between_decline_and_permission(self):
        cfg_off = self.make_config("http://unused/v1", browse=False)
        cfg_on = self.make_config("http://unused/v1", browse=True)

        prompt_off = agent.build_system_prompt(cfg_off)
        prompt_on = agent.build_system_prompt(cfg_on)
        self.assertIn("you cannot perform it", prompt_off)
        self.assertNotIn("jobagent browse", prompt_off)
        self.assertIn("jobagent browse", prompt_on)
        self.assertIn("job pending", prompt_on)
        self.assertNotIn("you cannot perform it", prompt_on)

    def test_browse_is_pinned_to_the_session_not_the_resuming_cfg(self):
        # Mirrors vision/interactive's own pinning guarantee (see
        # agent.drive_session's comment): a session started with browse
        # permission must keep it even if whatever cfg later calls
        # drive_session (a different call site, a changed default) doesn't
        # have it set.
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        start_cfg = self.make_config(f"http://127.0.0.1:{port}/v1", browse=True)
        session_id = agent.start_session(start_cfg, "t")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["browse"], 1)
        system_row = db.get_messages(con, session_id)[0]
        self.assertEqual(system_row["role"], "system")
        self.assertIn("jobagent browse", system_row["content"])

        resume_cfg = self.make_config(f"http://127.0.0.1:{port}/v1", browse=False)
        agent.drive_session(resume_cfg, session_id)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")

    def test_effort_is_pinned_to_the_session_not_the_resuming_cfg(self):
        # The bug this guards against: with_overrides() drops None values
        # (right for turning CLI args into overrides, wrong for pinning
        # re-application), so a naive `with_overrides(cfg, effort=session
        # ["effort"])` would silently skip the override whenever a session
        # was never given an effort (the common case) -- letting whatever
        # the *resuming* call's cfg happens to carry leak in instead. See
        # agent.drive_session's comment; this asserts the actual fix
        # (dataclasses.replace, not with_overrides) by spying on the cfg
        # build_client is actually called with mid-drive_session.
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        start_cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.start_session(start_cfg, "t")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertIsNone(db.get_session(con, session_id)["effort"])

        resume_cfg = self.make_config(f"http://127.0.0.1:{port}/v1", effort="high")
        real_build_client = agent.build_client
        captured = {}

        def spy(cfg):
            captured["effort"] = cfg.effort
            return real_build_client(cfg)

        with patch("orchestrator.agent.build_client", side_effect=spy):
            agent.drive_session(resume_cfg, session_id)

        self.assertIsNone(captured["effort"])
        self.assertEqual(db.get_session(con, session_id)["status"], "completed")

    def test_effort_pinned_value_is_actually_used_on_resume(self):
        # The other direction: a session that *was* given an effort must
        # keep using it even when resumed via a cfg with a different (or
        # unset) one.
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        start_cfg = self.make_config(f"http://127.0.0.1:{port}/v1", effort="low")
        session_id = agent.start_session(start_cfg, "t")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["effort"], "low")

        resume_cfg = self.make_config(f"http://127.0.0.1:{port}/v1")  # effort unset here
        real_build_client = agent.build_client
        captured = {}

        def spy(cfg):
            captured["effort"] = cfg.effort
            return real_build_client(cfg)

        with patch("orchestrator.agent.build_client", side_effect=spy):
            agent.drive_session(resume_cfg, session_id)

        self.assertEqual(captured["effort"], "low")

    def test_provider_model_base_url_pinned_to_session_not_resuming_cfg(self):
        # The real, reported bug: `model`/`base_url` have been columns on
        # `sessions` since the very first schema, but agent.drive_session
        # never actually read them back before this fix -- so resuming a
        # paused session via a call site that didn't re-specify
        # --provider/--model/--base-url (e.g. `sessions answer <id> "..."`
        # with no flags, or the UI's answer box) silently fell back to
        # whatever *that* call's own default config said. Prove it at the
        # HTTP level, not just by inspecting cfg: two real stub servers,
        # session started against one, resumed via a cfg pointed at the
        # other -- the resume request must still land on the *original*
        # server, with the *original* model name in the payload.
        server_a, state_a = make_stub_server([
            action({"ask": "need input"}), action({"final": "done"}),
        ])
        self.addCleanup(stop_server, server_a)
        server_b, state_b = make_stub_server([action({"final": "wrong server answered"})])
        self.addCleanup(stop_server, server_b)
        port_a, port_b = server_a.server_address[1], server_b.server_address[1]

        start_cfg = self.make_config(f"http://127.0.0.1:{port_a}/v1", model="model-a", interactive=True)
        session_id = agent.run_session(start_cfg, "t")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["status"], "waiting_for_user")
        self.assertEqual(state_a["calls"], 1)

        agent.answer_question(start_cfg, session_id, "here's your answer")
        # Simulates the bug exactly: resume via a *different* cfg (a
        # different default endpoint/model), the way a bare `sessions
        # answer` call (no flags re-specified) builds one from config.json.
        resume_cfg = self.make_config(f"http://127.0.0.1:{port_b}/v1", model="model-b")
        agent.drive_session(resume_cfg, session_id)

        self.assertEqual(db.get_session(con, session_id)["status"], "completed")
        self.assertEqual(db.get_session(con, session_id)["summary"], "done")
        self.assertEqual(state_a["calls"], 2)  # the resume call landed on server A
        self.assertEqual(state_b["calls"], 0)  # never touched server B at all


class TestScreenshotTool(OrchestratorTestBase):
    def make_ctx(self, vision: bool):
        cfg = self.make_config("http://unused/v1", vision=vision)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        sid = db.create_session(con, "t", "m", cfg.base_url)
        return tools.ToolContext(repo_root=self.repo_root, con=con, session_id=sid, config=cfg)

    def test_disabled_by_default_never_shells_out(self):
        ctx = self.make_ctx(vision=False)
        with patch("orchestrator.tools.subprocess.run") as run:
            out = tools.tool_screenshot(ctx, {})
        run.assert_not_called()
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("ERROR"))
        self.assertIn("vision", out.lower())

    def test_success_returns_tool_result_with_image_path(self):
        ctx = self.make_ctx(vision=True)
        fake_png = Path(self.tmp.name) / "shot.png"
        fake_png.write_bytes(b"\x89PNG fake bytes")
        payload = {"ok": True, "path": str(fake_png), "url": "https://example.com/job/1",
                   "viewport": {"width": 1280, "height": 800}}
        fake_proc = MagicMock(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

        with patch("orchestrator.tools.subprocess.run", return_value=fake_proc) as run:
            result = tools.tool_screenshot(ctx, {"label": "job page!", "full": True})

        run.assert_called_once()
        called_command = run.call_args[0][0]
        self.assertIn("browse shot", called_command)
        self.assertIn("--full", called_command)
        self.assertIsInstance(result, tools.ToolResult)
        self.assertEqual(result.image_path, fake_png)
        self.assertIn("example.com/job/1", result.text)

    def test_driver_failure_reported_as_text_error(self):
        ctx = self.make_ctx(vision=True)
        payload = {"ok": False, "error": "no browser driver connected"}
        fake_proc = MagicMock(returncode=1, stdout=json.dumps(payload), stderr="")

        with patch("orchestrator.tools.subprocess.run", return_value=fake_proc):
            result = tools.tool_screenshot(ctx, {})

        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("ERROR"))
        self.assertIn("no browser driver connected", result)

    def test_missing_screenshot_file_reported_as_text_error(self):
        ctx = self.make_ctx(vision=True)
        missing = Path(self.tmp.name) / "does_not_exist.png"
        payload = {"ok": True, "path": str(missing)}
        fake_proc = MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")

        with patch("orchestrator.tools.subprocess.run", return_value=fake_proc):
            result = tools.tool_screenshot(ctx, {})

        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("ERROR"))


class TestAgentVisionLoop(OrchestratorTestBase):
    def test_screenshot_flows_through_to_a_multimodal_llm_turn(self):
        fake_png = Path(self.tmp.name) / "shot.png"
        fake_png.write_bytes(b"\x89PNG fake bytes")
        captured_requests: list[dict] = []

        def fake_screenshot(ctx, input):
            return tools.ToolResult(text="a screenshot of the page", image_path=fake_png)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                captured_requests.append(body)
                if len(captured_requests) == 1:
                    content = action({"tool": "screenshot", "input": {"label": "x"}})
                else:
                    content = action({"final": "saw the screenshot"})
                resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]

        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", vision=True)
        with patch("orchestrator.tools.tool_screenshot", fake_screenshot):
            session_id = agent.run_session(cfg, "look at the page")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")

        rows = db.get_messages(con, session_id)
        shot_row = next(r for r in rows if r["tool_name"] == "screenshot")
        self.assertEqual(shot_row["image_path"], str(fake_png))

        self.assertIn("screenshot", captured_requests[0]["messages"][0]["content"])

        tool_turn = captured_requests[1]["messages"][-1]
        self.assertIsInstance(tool_turn["content"], list)
        self.assertIn("image_url", [b["type"] for b in tool_turn["content"]])
        image_block = next(b for b in tool_turn["content"] if b["type"] == "image_url")
        self.assertTrue(image_block["image_url"]["url"].startswith("data:image"))

    def test_resuming_with_send_images_false_strips_an_earlier_screenshot(self):
        # The bug this guards against: a session takes a screenshot (vision
        # was on then), later pauses on `ask`, and gets resumed -- possibly
        # against a different model/endpoint that turns out not to accept
        # image input at all (a real 400: "messages contain images, but
        # <model> does not support image inputs"). send_images is not
        # pinned to the session the way vision/interactive/browse are
        # (see agent.drive_session's comment), specifically so a resume
        # call can recover a session stuck like this.
        fake_png = Path(self.tmp.name) / "shot.png"
        fake_png.write_bytes(b"\x89PNG fake bytes")
        captured_requests: list[dict] = []

        def fake_screenshot(ctx, input):
            return tools.ToolResult(text="a screenshot of the page", image_path=fake_png)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                captured_requests.append(body)
                if len(captured_requests) == 1:
                    content = action({"tool": "screenshot", "input": {"label": "x"}})
                elif len(captured_requests) == 2:
                    content = action({"ask": "need input to continue"})
                else:
                    content = action({"final": "done"})
                resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]

        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", vision=True, interactive=True)
        with patch("orchestrator.tools.tool_screenshot", fake_screenshot):
            session_id = agent.run_session(cfg, "look at the page")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["status"], "waiting_for_user")

        from orchestrator.config import with_overrides

        agent.answer_question(cfg, session_id, "here's your answer")
        resume_cfg = with_overrides(cfg, send_images=False)
        agent.drive_session(resume_cfg, session_id)

        self.assertEqual(db.get_session(con, session_id)["status"], "completed")
        resume_request = captured_requests[-1]
        for m in resume_request["messages"]:
            self.assertNotIsInstance(m["content"], list, "no message should still carry an image block")
        blob = json.dumps(resume_request)
        self.assertIn("screenshot omitted", blob)
        self.assertNotIn("data:image", blob)


class TestInteractiveConfig(unittest.TestCase):
    def test_interactive_defaults_off(self):
        cfg = Config(
            base_url="x", api_key="k", model="m", temperature=0.0, max_tokens=1,
            max_steps=1, max_consecutive_errors=1, request_timeout=1, shell_timeout=1,
            repo_root=Path("."),
        )
        self.assertFalse(cfg.interactive)

    def test_env_var_enables_interactive(self):
        import os

        from orchestrator.config import load_config

        old = os.environ.get("ORCHESTRATOR_INTERACTIVE")
        os.environ["ORCHESTRATOR_INTERACTIVE"] = "true"
        try:
            self.assertTrue(load_config().interactive)
        finally:
            if old is None:
                os.environ.pop("ORCHESTRATOR_INTERACTIVE", None)
            else:
                os.environ["ORCHESTRATOR_INTERACTIVE"] = old

    def test_with_overrides_sets_interactive(self):
        from orchestrator.config import load_config, with_overrides

        cfg = load_config()
        cfg2 = with_overrides(cfg, interactive=True)
        self.assertTrue(cfg2.interactive)


class TestBrowseConfig(unittest.TestCase):
    def test_browse_defaults_off(self):
        cfg = Config(
            base_url="x", api_key="k", model="m", temperature=0.0, max_tokens=1,
            max_steps=1, max_consecutive_errors=1, request_timeout=1, shell_timeout=1,
            repo_root=Path("."),
        )
        self.assertFalse(cfg.browse)

    def test_env_var_enables_browse(self):
        import os

        from orchestrator.config import load_config

        old = os.environ.get("ORCHESTRATOR_BROWSE")
        os.environ["ORCHESTRATOR_BROWSE"] = "true"
        try:
            self.assertTrue(load_config().browse)
        finally:
            if old is None:
                os.environ.pop("ORCHESTRATOR_BROWSE", None)
            else:
                os.environ["ORCHESTRATOR_BROWSE"] = old

    def test_with_overrides_sets_browse(self):
        from orchestrator.config import load_config, with_overrides

        cfg = load_config()
        cfg2 = with_overrides(cfg, browse=True)
        self.assertTrue(cfg2.browse)
        self.assertFalse(cfg.browse)
        self.assertFalse(cfg.interactive)


class TestSendImagesConfig(unittest.TestCase):
    def test_send_images_defaults_on(self):
        cfg = Config(
            base_url="x", api_key="k", model="m", temperature=0.0, max_tokens=1,
            max_steps=1, max_consecutive_errors=1, request_timeout=1, shell_timeout=1,
            repo_root=Path("."),
        )
        self.assertTrue(cfg.send_images)

    def test_env_var_disables_send_images(self):
        import os

        from orchestrator.config import load_config

        old = os.environ.get("ORCHESTRATOR_SEND_IMAGES")
        os.environ["ORCHESTRATOR_SEND_IMAGES"] = "false"
        try:
            self.assertFalse(load_config().send_images)
        finally:
            if old is None:
                os.environ.pop("ORCHESTRATOR_SEND_IMAGES", None)
            else:
                os.environ["ORCHESTRATOR_SEND_IMAGES"] = old

    def test_with_overrides_disables_send_images_without_mutating_original(self):
        from orchestrator.config import load_config, with_overrides

        cfg = load_config()
        cfg2 = with_overrides(cfg, send_images=False)
        self.assertFalse(cfg2.send_images)
        self.assertTrue(cfg.send_images)


class TestEffortConfig(unittest.TestCase):
    def test_effort_defaults_unset(self):
        cfg = Config(
            base_url="x", api_key="k", model="m", temperature=0.0, max_tokens=1,
            max_steps=1, max_consecutive_errors=1, request_timeout=1, shell_timeout=1,
            repo_root=Path("."),
        )
        self.assertIsNone(cfg.effort)

    def test_env_var_sets_effort(self):
        import os

        from orchestrator.config import load_config

        old = os.environ.get("ORCHESTRATOR_EFFORT")
        os.environ["ORCHESTRATOR_EFFORT"] = "xhigh"
        try:
            self.assertEqual(load_config().effort, "xhigh")
        finally:
            if old is None:
                os.environ.pop("ORCHESTRATOR_EFFORT", None)
            else:
                os.environ["ORCHESTRATOR_EFFORT"] = old

    def test_with_overrides_sets_effort_without_mutating_original(self):
        from orchestrator.config import load_config, with_overrides

        cfg = load_config()
        cfg2 = with_overrides(cfg, effort="max")
        self.assertEqual(cfg2.effort, "max")
        self.assertIsNone(cfg.effort)


class TestAskProtocol(OrchestratorTestBase):
    def test_ask_disallowed_when_not_interactive(self):
        server, state = make_stub_server([
            action({"ask": "which company should I target?"}),
            action({"final": "recorded the gap and finished"}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", interactive=False)

        session_id = agent.run_session(cfg, "test task")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")

        rows = db.get_messages(con, session_id)
        ask_rows = [r for r in rows if r["tool_name"] == "_ask_user"]
        self.assertEqual(len(ask_rows), 1)
        self.assertTrue(ask_rows[0]["content"].startswith("ERROR"))

    def test_ask_pauses_session_and_resumes_to_completion(self):
        server, state = make_stub_server([
            action({"ask": "which company should I target?"}),
            action({"final": "done, used Acme"}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", interactive=True)

        # `run_session` with no `on_ask` drives once and surfaces the pause,
        # like the UI does — it must not block or loop on its own.
        session_id = agent.run_session(cfg, "test task")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        messages = db.get_messages(con, session_id)
        self.assertEqual(session["status"], "waiting_for_user")
        self.assertEqual(agent.pending_question(session, messages), "which company should I target?")
        self.assertEqual(state["calls"], 1)  # never called the model a 2nd time while paused

        agent.answer_question(cfg, session_id, "Acme")
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "running")

        agent.drive_session(cfg, session_id)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["summary"], "done, used Acme")

        messages = db.get_messages(con, session_id)
        answer_rows = [r for r in messages if r["role"] == "user" and r["content"] == "Acme"]
        self.assertEqual(len(answer_rows), 1)

    def test_answer_question_rejects_non_paused_session(self):
        cfg = self.make_config("http://unused/v1", interactive=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session_id = db.create_session(con, "t", "m", cfg.base_url, interactive=True)
        with self.assertRaises(ValueError):
            agent.answer_question(cfg, session_id, "anything")

    def test_retry_session_rejects_non_retryable_status(self):
        cfg = self.make_config("http://unused/v1")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session_id = db.create_session(con, "t", "m", cfg.base_url)  # status: running
        with self.assertRaises(ValueError):
            agent.retry_session(cfg, session_id)

    def test_retry_session_resumes_from_exactly_where_it_failed(self):
        """A `failed` session's `client.chat()` call raises before anything
        gets appended to the transcript -- retrying after fixing the real
        cause (here: a placeholder api_key against a real remote host) must
        pick up with exactly the same messages, not lose or duplicate a
        step. Regression coverage for the retry feature end to end."""
        cfg = self.make_config("https://api.example.com/v1", api_key="EMPTY")
        session_id = agent.run_session(cfg, "do the thing")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "failed")
        self.assertIn("No API key configured", session["error"])
        messages_before = db.get_messages(con, session_id)
        self.assertEqual(len(messages_before), 2)  # system + user only -- never got to call the model

        fixed_cfg = self.make_config("https://api.example.com/v1", api_key="sk-real-key")
        agent.retry_session(fixed_cfg, session_id)
        self.assertEqual(db.get_session(con, session_id)["status"], "running")

        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": action({"final": "done"})}}]}
        ).encode("utf-8")
        fake_resp.__enter__ = lambda self: fake_resp
        fake_resp.__exit__ = lambda self, *a: None
        with patch("urllib.request.urlopen", return_value=fake_resp) as urlopen:
            agent.drive_session(fixed_cfg, session_id)
        urlopen.assert_called_once()

        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["summary"], "done")
        self.assertEqual(len(db.get_messages(con, session_id)), 3)  # +1: the successful assistant reply

    def test_retry_session_clears_stale_error_immediately(self):
        """Regression test: retry_session must clear `error` the instant it
        flips status back to `running`, not just once the retry eventually
        succeeds or fails again -- otherwise the UI footer and `sessions
        show` (which both display `session["error"]` whenever it's set,
        regardless of current status) keep showing the *previous* failure's
        message for the whole duration of a genuinely-running retry, making
        it look stuck erroring with no way to retry again."""
        cfg = self.make_config("https://api.example.com/v1", api_key="EMPTY")
        session_id = agent.run_session(cfg, "do the thing")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["status"], "failed")
        self.assertIsNotNone(db.get_session(con, session_id)["error"])

        fixed_cfg = self.make_config("https://api.example.com/v1", api_key="sk-real-key")
        agent.retry_session(fixed_cfg, session_id)

        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "running")
        self.assertIsNone(session["error"])

    def test_request_stop_rejects_non_running_session(self):
        cfg = self.make_config("http://unused/v1", interactive=True)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session_id = db.create_session(con, "t", "m", cfg.base_url, interactive=True)
        db.set_status(con, session_id, "waiting_for_user")
        with self.assertRaises(ValueError):
            agent.request_stop(cfg, session_id)

    def test_request_stop_force_ends_an_orphaned_session_with_stale_heartbeat(self):
        """A session can be orphaned -- its driving UI thread died with a
        server restart, or its driving CLI process was killed -- and left
        stuck showing `running` forever with nothing left to notice a
        cooperative stop flag. request_stop must detect this (a stale
        heartbeat) and end the session itself, rather than leaving a flag
        nothing will ever read."""
        cfg = self.make_config("http://unused/v1")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session_id = db.create_session(con, "t", "m", cfg.base_url)
        stale_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
        con.execute("UPDATE sessions SET heartbeat_at=? WHERE id=?", (stale_ts, session_id))
        con.commit()

        agent.request_stop(cfg, session_id)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "stopped")
        self.assertIn("no heartbeat", session["error"])

    def test_request_stop_leaves_a_freshly_heartbeating_session_running(self):
        """A session whose heartbeat is recent must NOT be force-ended --
        only flagged, same as before -- since something might genuinely
        still be mid-step and due to notice the flag momentarily."""
        cfg = self.make_config("http://unused/v1")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session_id = db.create_session(con, "t", "m", cfg.base_url)
        db.touch_heartbeat(con, session_id)

        agent.request_stop(cfg, session_id)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "running")
        self.assertEqual(session["stop_requested"], 1)

    def test_stopped_session_finishes_current_step_but_skips_the_next(self):
        """Stop is cooperative -- checked once per step, before the *next*
        model call, not mid-request -- so a stop requested while step 2's
        request is in flight still lets step 2 finish; only step 3 (the
        model call that would follow it) never happens."""
        request_count = {"n": 0}
        session_holder: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                idx = request_count["n"]
                request_count["n"] += 1
                if idx == 1:  # while step 2's "model call" is in flight
                    con = db.connect(cfg.db_path)
                    try:
                        db.request_stop(con, session_holder["id"])
                    finally:
                        con.close()
                content = action({"tool": "memory", "input": {"action": "list"}}) if idx < 2 \
                    else action({"final": "should never get here"})
                resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]

        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.start_session(cfg, "do the thing")
        session_holder["id"] = session_id
        agent.drive_session(cfg, session_id)

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "stopped")
        self.assertEqual(request_count["n"], 2)  # never made a 3rd (post-stop) call
        assistant_msgs = [m for m in db.get_messages(con, session_id) if m["role"] == "assistant"]
        self.assertEqual(len(assistant_msgs), 2)  # both in-flight steps completed, nothing partial

    def test_retry_session_resumes_a_stopped_session(self):
        server, state = make_stub_server([
            action({"tool": "memory", "input": {"action": "list"}}),
            action({"final": "done"}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.start_session(cfg, "do the thing")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        db.request_stop(con, session_id)  # stop before the very first step
        agent.drive_session(cfg, session_id)
        self.assertEqual(db.get_session(con, session_id)["status"], "stopped")
        self.assertEqual(state["calls"], 0)  # stopped before ever calling the model

        agent.retry_session(cfg, session_id)
        self.assertEqual(db.get_session(con, session_id)["status"], "running")
        agent.drive_session(cfg, session_id)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["summary"], "done")
        self.assertEqual(state["calls"], 2)

    def test_drive_interactively_with_on_ask_completes_in_one_call(self):
        server, state = make_stub_server([
            action({"ask": "which company should I target?"}),
            action({"final": "done, used Acme"}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", interactive=True)

        asked = []

        def on_ask(question: str) -> str:
            asked.append(question)
            return "Acme"

        session_id = agent.run_session(cfg, "test task", on_ask=on_ask)

        self.assertEqual(asked, ["which company should I target?"])
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["summary"], "done, used Acme")

    def test_step_budget_spans_pause_and_resume(self):
        server, state = make_stub_server([
            action({"ask": "q1"}),
            action({"ask": "q2"}),
            action({"final": "should never get here"}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", interactive=True, max_steps=2)

        session_id = agent.run_session(cfg, "test task")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "waiting_for_user")
        self.assertEqual(session["step_count"], 1)

        agent.answer_question(cfg, session_id, "a1")
        agent.drive_session(cfg, session_id)
        session = db.get_session(con, session_id)
        self.assertEqual(session["status"], "waiting_for_user")
        self.assertEqual(session["step_count"], 2)

        agent.answer_question(cfg, session_id, "a2")
        agent.drive_session(cfg, session_id)
        session = db.get_session(con, session_id)
        # max_steps (2) already spent across the two pauses — the loop must
        # not spend a 3rd step even though one more `drive_session` call ran.
        self.assertEqual(session["status"], "max_steps_reached")
        self.assertEqual(state["calls"], 2)


class TestUI(OrchestratorTestBase):
    def start_ui(self, cfg) -> str:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler(cfg))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # addCleanup runs LIFO -- register server_close() first so
        # shutdown() (stop the accept loop) actually runs *before*
        # server_close() (close the socket); the reverse order races
        # serve_forever's select() against the now-closed fd on Windows.
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def _post(self, url: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, url: str) -> dict:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _delete(self, url: str) -> dict:
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_run_endpoint_applies_provider_model_base_url_overrides(self):
        # Server-level cfg deliberately points nowhere real; the /api/run
        # override should be what actually gets used for this session --
        # same per-run-only override semantics the CLI's --provider/--model/
        # --base-url already have via with_overrides.
        target_server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, target_server)
        target_port = target_server.server_address[1]
        wrong_cfg = self.make_config("http://127.0.0.1:1/v1", model="wrong-model")
        base = self.start_ui(wrong_cfg)

        res = self._post(f"{base}/api/run", {
            "task": "t", "provider": "openai", "model": "overridden-model",
            "base_url": f"http://127.0.0.1:{target_port}/v1",
        })
        session_id = res["session_id"]

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = None
        for _ in range(50):
            session = db.get_session(con, session_id)
            if session and session["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["model"], "overridden-model")
        self.assertEqual(session["base_url"], f"http://127.0.0.1:{target_port}/v1")

    def test_finetune_export_endpoint_returns_report_shape(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t", finetune_flag=True)
        base = self.start_ui(cfg)

        res = self._post(f"{base}/api/finetune/export", {
            "out": str(Path(self.tmp.name) / "ui_export.jsonl"), "all": True,
        })
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["train_count"], 1)
        self.assertEqual(res["val_count"], 0)
        self.assertIn("by_category", res)
        self.assertIn("skipped_invalid", res)

    def test_review_endpoint_sets_and_clears(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t")
        base = self.start_ui(cfg)

        res = self._post(f"{base}/api/sessions/{session_id}/review", {"reward": 2, "note": "nice"})
        self.assertTrue(res["ok"])
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, session_id)
        self.assertEqual(session["reward"], 2)
        self.assertEqual(session["review_note"], "nice")

        res = self._post(f"{base}/api/sessions/{session_id}/review", {"clear": True})
        self.assertTrue(res["ok"])
        self.assertEqual(db.get_session(con, session_id)["reviewed"], 0)

    def test_review_endpoint_rejects_bad_reward(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t")
        base = self.start_ui(cfg)
        try:
            self._post(f"{base}/api/sessions/{session_id}/review", {"reward": 99})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def _completed_session(self, flagged: bool) -> tuple[str, Path]:
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t", finetune_flag=flagged)
        return session_id, cfg.finetune_dir

    def test_delete_endpoint_removes_session_and_export_but_keeps_screenshots_by_default(self):
        session_id, finetune_dir = self._completed_session(flagged=True)
        export_path = finetune_dir / f"{session_id}.jsonl"
        self.assertTrue(export_path.exists())

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        image_path = Path(self.tmp.name) / "shot.png"
        image_path.write_bytes(b"fake png")
        db.add_message(con, session_id, "tool", "saw something", tool_name="screenshot",
                        image_path=str(image_path))

        base = self.start_ui(self.make_config("http://unused/v1"))
        res = self._delete(f"{base}/api/sessions/{session_id}")

        self.assertTrue(res["ok"])
        self.assertTrue(res["removed_export"])
        self.assertEqual(res["screenshot_count"], 1)
        self.assertEqual(res["removed_screenshots"], 0)
        self.assertIsNone(db.get_session(con, session_id))
        self.assertFalse(export_path.exists())
        self.assertTrue(image_path.is_file())  # left in place -- not requested

    def test_delete_endpoint_removes_screenshots_when_asked(self):
        session_id, finetune_dir = self._completed_session(flagged=False)
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        image_path = Path(self.tmp.name) / "shot2.png"
        image_path.write_bytes(b"fake png")
        db.add_message(con, session_id, "tool", "saw something", tool_name="screenshot",
                        image_path=str(image_path))

        base = self.start_ui(self.make_config("http://unused/v1"))
        res = self._delete(f"{base}/api/sessions/{session_id}?delete_screenshots=1")

        self.assertEqual(res["removed_screenshots"], 1)
        self.assertFalse(image_path.exists())

    def test_delete_endpoint_404s_for_unknown_session(self):
        base = self.start_ui(self.make_config("http://unused/v1"))
        try:
            self._delete(f"{base}/api/sessions/deadbeef")
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_answer_endpoint_send_images_false_strips_screenshot_on_resume(self):
        fake_png = Path(self.tmp.name) / "shot.png"
        fake_png.write_bytes(b"\x89PNG fake bytes")
        captured_requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                captured_requests.append(body)
                if len(captured_requests) == 1:
                    content = action({"tool": "screenshot", "input": {"label": "x"}})
                elif len(captured_requests) == 2:
                    content = action({"ask": "need input"})
                else:
                    content = action({"final": "done"})
                resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]

        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", vision=True, interactive=True)
        with patch("orchestrator.tools.tool_screenshot",
                   lambda ctx, input: tools.ToolResult(text="a screenshot", image_path=fake_png)):
            session_id = agent.run_session(cfg, "look at the page")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["status"], "waiting_for_user")

        base = self.start_ui(cfg)
        res = self._post(f"{base}/api/sessions/{session_id}/answer",
                          {"answer": "here's your answer", "send_images": False})
        self.assertTrue(res["ok"])

        for _ in range(50):
            if db.get_session(con, session_id)["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(db.get_session(con, session_id)["status"], "completed")

        resume_request = captured_requests[-1]
        for m in resume_request["messages"]:
            self.assertNotIsInstance(m["content"], list)
        self.assertIn("screenshot omitted", json.dumps(resume_request))

    def test_answer_endpoint_api_key_override_used_on_resume_not_session_default(self):
        """api_key is deliberately never pinned to the session row -- a
        resume from the UI has to be able to supply it fresh, the same way
        `sessions answer --api-key` already can from the CLI. Regression
        test for the answer box previously having no way to do this at all,
        silently falling back to whatever key the `orchestrator ui` process
        itself started with."""
        captured_auth: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                captured_auth.append(self.headers.get("Authorization", ""))
                if len(captured_auth) == 1:
                    content = action({"ask": "need input"})
                else:
                    content = action({"final": "done"})
                resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]

        # Started with the project's own local-server placeholder -- the
        # resume must NOT keep using this once a real key is supplied.
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", api_key="EMPTY", interactive=True)
        session_id = agent.run_session(cfg, "do the thing")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["status"], "waiting_for_user")

        base = self.start_ui(cfg)
        res = self._post(f"{base}/api/sessions/{session_id}/answer",
                          {"answer": "here's your answer", "api_key": "sk-real-key"})
        self.assertTrue(res["ok"])

        for _ in range(50):
            if db.get_session(con, session_id)["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(db.get_session(con, session_id)["status"], "completed")

        self.assertEqual(captured_auth, ["Bearer EMPTY", "Bearer sk-real-key"])

    def test_retry_endpoint_404s_for_unknown_session(self):
        base = self.start_ui(self.make_config("http://unused/v1"))
        try:
            self._post(f"{base}/api/sessions/deadbeef/retry", {})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_retry_endpoint_409s_for_non_retryable_session(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t")  # completes -- not retryable
        base = self.start_ui(cfg)
        try:
            self._post(f"{base}/api/sessions/{session_id}/retry", {})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 409)

    def test_retry_endpoint_api_key_override_resumes_failed_session(self):
        """The retry button's whole point: a session that died on a
        placeholder api_key against a real remote host resumes once a real
        key is supplied on the retry call, without needing to restart the
        task from scratch."""
        captured_auth: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                captured_auth.append(self.headers.get("Authorization", ""))
                content = action({"final": "done"})
                resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]

        # 127.0.0.1 counts as local, so this session doesn't fail its own
        # preflight -- simulate the DeepSeek-style failure some other way:
        # the model call itself errors out (e.g. a timeout), which is
        # exactly the `failed` path retry_session is built for.
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", api_key="EMPTY")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            session_id = agent.run_session(cfg, "do the thing")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        self.assertEqual(db.get_session(con, session_id)["status"], "failed")

        base = self.start_ui(cfg)
        res = self._post(f"{base}/api/sessions/{session_id}/retry", {"api_key": "sk-real-key"})
        self.assertTrue(res["ok"])

        for _ in range(50):
            if db.get_session(con, session_id)["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(db.get_session(con, session_id)["status"], "completed")
        self.assertEqual(captured_auth, ["Bearer sk-real-key"])

    def test_stop_endpoint_404s_for_unknown_session(self):
        base = self.start_ui(self.make_config("http://unused/v1"))
        try:
            self._post(f"{base}/api/sessions/deadbeef/stop", {})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_stop_endpoint_409s_for_non_running_session(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t")  # completes -- not stoppable
        base = self.start_ui(cfg)
        try:
            self._post(f"{base}/api/sessions/{session_id}/stop", {})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 409)

    def test_stop_endpoint_flags_a_running_session_which_then_stops_itself(self):
        """The UI's Stop button is a pure flag-setter -- it's the session's
        own already-running background thread (spawned by an earlier /run
        or /answer call) that notices the flag and actually ends the
        session, at its next step boundary."""
        request_count = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                request_count["n"] += 1
                content = action({"tool": "memory", "input": {"action": "list"}})
                resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1", max_steps=50)

        base = self.start_ui(cfg)
        res = self._post(f"{base}/api/run", {"task": "loop forever"})
        session_id = res["session_id"]

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        # Let it take at least one real step before stopping it.
        for _ in range(50):
            if request_count["n"] >= 1:
                break
            time.sleep(0.05)

        stop_res = self._post(f"{base}/api/sessions/{session_id}/stop", {})
        self.assertTrue(stop_res["ok"])

        for _ in range(100):
            if db.get_session(con, session_id)["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(db.get_session(con, session_id)["status"], "stopped")

    def test_retry_endpoint_accepts_a_stopped_session(self):
        server, state = make_stub_server([
            action({"tool": "memory", "input": {"action": "list"}}),
            action({"final": "done"}),
        ])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.start_session(cfg, "t")

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        db.request_stop(con, session_id)
        agent.drive_session(cfg, session_id)
        self.assertEqual(db.get_session(con, session_id)["status"], "stopped")

        base = self.start_ui(cfg)
        res = self._post(f"{base}/api/sessions/{session_id}/retry", {})
        self.assertTrue(res["ok"])

        for _ in range(50):
            if db.get_session(con, session_id)["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(db.get_session(con, session_id)["status"], "completed")

    def test_import_endpoint_loads_content_into_sessions(self):
        cfg = self.make_config("http://unused/v1")
        base = self.start_ui(cfg)
        content = json.dumps({"session_id": "imported1", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": action({"final": "ok"})},
        ]})
        res = self._post(f"{base}/api/sessions/import", {"content": content})
        self.assertEqual(res["imported"], ["imported1"])
        self.assertEqual(res["skipped"], [])

        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        session = db.get_session(con, "imported1")
        self.assertIsNotNone(session)
        self.assertEqual(session["finetune"], 1)

    def test_import_endpoint_requires_content(self):
        cfg = self.make_config("http://unused/v1")
        base = self.start_ui(cfg)
        try:
            self._post(f"{base}/api/sessions/import", {})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_step_review_endpoint_sets_and_clears(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        final_row = next(r for r in db.get_messages(con, session_id) if r["role"] == "assistant")
        base = self.start_ui(cfg)

        res = self._post(f"{base}/api/sessions/{session_id}/messages/{final_row['seq']}/review",
                          {"reward": 1, "note": "fine"})
        self.assertTrue(res["ok"])
        row = db.get_messages(con, session_id)[final_row["seq"] - 1]
        self.assertEqual(row["reward"], 1)
        self.assertEqual(row["review_note"], "fine")

        res = self._post(f"{base}/api/sessions/{session_id}/messages/{final_row['seq']}/review", {"clear": True})
        self.assertTrue(res["ok"])
        row = db.get_messages(con, session_id)[final_row["seq"] - 1]
        self.assertIsNone(row["reward"])

    def test_step_review_endpoint_404s_for_unknown_step(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t")
        base = self.start_ui(cfg)
        try:
            self._post(f"{base}/api/sessions/{session_id}/messages/999/review", {"reward": 1})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_step_review_endpoint_rejects_non_assistant_row(self):
        server, state = make_stub_server([action({"final": "ok"})])
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        cfg = self.make_config(f"http://127.0.0.1:{port}/v1")
        session_id = agent.run_session(cfg, "t")
        con = db.connect(self.db_path)
        self.addCleanup(con.close)
        system_row = next(r for r in db.get_messages(con, session_id) if r["role"] == "system")
        base = self.start_ui(cfg)
        try:
            self._post(f"{base}/api/sessions/{session_id}/messages/{system_row['seq']}/review", {"reward": 1})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_annotate_page_is_served(self):
        cfg = self.make_config("http://unused/v1")
        base = self.start_ui(cfg)
        with urllib.request.urlopen(f"{base}/annotate", timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            body = resp.read().decode("utf-8")
        self.assertIn("renderStepCard", body)
        self.assertIn("sessionReviewBar", body)
        self.assertIn("exportBtn", body)
        self.assertIn("importBtn", body)

    def test_main_page_no_longer_has_review_export_import(self):
        # Review, export and import all moved to /annotate -- the main
        # page keeps only the review-status badge on session rows.
        cfg = self.make_config("http://unused/v1")
        base = self.start_ui(cfg)
        with urllib.request.urlopen(base + "/", timeout=10) as resp:
            body = resp.read().decode("utf-8")
        for marker in ("id=\"reviewPanel\"", "id=\"exportBtn\"", "id=\"importBtn\"",
                       "id=\"exportPanel\"", "id=\"importPanel\""):
            self.assertNotIn(marker, body)
        self.assertIn("reviewBadgeHtml", body)  # status badge stays
        self.assertIn('href="/annotate"', body)


class TestAnthropicMessageConversion(unittest.TestCase):
    def test_system_pulled_out_and_roles_kept(self):
        system_text, converted = llm.to_anthropic_messages([
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "```action\n{\"final\": \"ok\"}\n```"},
        ])
        self.assertEqual(system_text, "sys prompt")
        self.assertEqual([m["role"] for m in converted], ["user", "assistant"])
        self.assertEqual(converted[0]["content"], [{"type": "text", "text": "do it"}])

    def test_consecutive_user_turns_are_merged(self):
        # Mirrors what db.to_api_messages actually produces across an ask/answer
        # pause: the question (tool row, mapped to 'user') immediately followed
        # by the human's answer (also 'user'), with no assistant turn between.
        _, converted = llm.to_anthropic_messages([
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "```action\n{\"ask\": \"which company?\"}\n```"},
            {"role": "user", "content": "Observation (_ask_user): which company?"},
            {"role": "user", "content": "Acme"},
        ])
        roles = [m["role"] for m in converted]
        self.assertEqual(roles, ["user", "assistant", "user"])
        merged_texts = [b["text"] for b in converted[-1]["content"]]
        self.assertEqual(merged_texts, ["Observation (_ask_user): which company?", "Acme"])

    def test_image_block_converted_from_data_uri(self):
        _, converted = llm.to_anthropic_messages([
            {"role": "user", "content": [
                {"type": "text", "text": "saw something"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
            ]},
        ])
        blocks = converted[0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "saw something"})
        self.assertEqual(blocks[1], {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "QUJD",
        }})


class TestLooksLocal(unittest.TestCase):
    def test_localhost_and_loopback_and_private_are_local(self):
        for url in (
            "http://localhost:8000/v1",
            "http://127.0.0.1:1234/v1",
            "http://[::1]:8000/v1",
            "http://0.0.0.0:8000/v1",
            "http://192.168.1.5:8000/v1",
            "http://10.0.0.7:8000/v1",
        ):
            self.assertTrue(llm._looks_local(url), url)

    def test_real_remote_hosts_are_not_local(self):
        for url in (
            "https://api.deepseek.com/v1",
            "https://api.anthropic.com",
            "https://example.com",
        ):
            self.assertFalse(llm._looks_local(url), url)


class TestLLMClient(unittest.TestCase):
    """Direct coverage of the plain OpenAI-compatible client's error paths
    -- previously only exercised indirectly through the stub-server-based
    agent loop tests, which never hit a URLError."""

    def test_url_error_against_local_host_hints_at_local_server(self):
        client = llm.LLMClient(base_url="http://127.0.0.1:1234/v1", api_key="k", model="m")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            with self.assertRaises(llm.LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}])
        msg = str(cm.exception)
        self.assertIn("Is the local model server running", msg)
        self.assertNotIn("firewall", msg)

    def test_url_error_against_remote_host_hints_at_network_blocking(self):
        client = llm.LLMClient(base_url="https://api.deepseek.com", api_key="k", model="m")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            with self.assertRaises(llm.LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}])
        msg = str(cm.exception)
        self.assertIn("remote host", msg)
        self.assertIn("firewall", msg)
        self.assertNotIn("local model server", msg)

    def test_placeholder_key_against_remote_host_fails_fast_without_a_request(self):
        client = llm.LLMClient(base_url="https://api.deepseek.com", api_key="EMPTY", model="m")
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(llm.LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}])
        urlopen.assert_not_called()
        self.assertIn("No API key configured", str(cm.exception))

    def test_empty_key_against_remote_host_fails_fast(self):
        client = llm.LLMClient(base_url="https://api.deepseek.com", api_key="", model="m")
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(llm.LLMError):
                client.chat([{"role": "user", "content": "hi"}])
        urlopen.assert_not_called()

    def test_placeholder_key_against_local_host_is_allowed(self):
        """A local server (e.g. vLLM) that doesn't check the key at all is
        the whole reason "EMPTY" is the project default -- must not be
        blocked by the new preflight check."""
        client = llm.LLMClient(base_url="http://127.0.0.1:1234/v1", api_key="EMPTY", model="m")
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}
        ).encode("utf-8")
        fake_resp.__enter__ = lambda self: fake_resp
        fake_resp.__exit__ = lambda self, *a: None
        with patch("urllib.request.urlopen", return_value=fake_resp):
            reply = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "ok")


class TestAnthropicClient(unittest.TestCase):
    def make_stub(self, response_body: dict):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                captured["body"] = json.loads(self.rfile.read(n) or b"{}")
                captured["headers"] = dict(self.headers)
                captured["path"] = self.path
                body = json.dumps(response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, captured

    def test_chat_sends_system_field_and_parses_text_blocks(self):
        server, captured = self.make_stub({
            "content": [{"type": "text", "text": "```action\n{\"final\": \"done\"}\n```"}],
        })
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        client = llm.AnthropicClient(
            base_url=f"http://127.0.0.1:{port}", api_key="sk-test", model="claude-x",
        )
        reply = client.chat([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ])
        self.assertEqual(reply, '```action\n{"final": "done"}\n```')
        self.assertEqual(captured["path"], "/v1/messages")
        self.assertEqual(captured["headers"]["X-Api-Key"], "sk-test")
        self.assertEqual(captured["headers"]["Anthropic-Version"], llm.ANTHROPIC_API_VERSION)
        self.assertEqual(captured["body"]["system"], "sys")
        self.assertEqual(captured["body"]["messages"], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])

    def test_http_error_raises_llm_error(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                body = json.dumps({"error": {"message": "bad api key"}}).encode("utf-8")
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(stop_server, server)
        port = server.server_address[1]
        client = llm.AnthropicClient(base_url=f"http://127.0.0.1:{port}", api_key="bad", model="claude-x")
        with self.assertRaises(llm.LLMError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_url_error_against_local_host_hints_at_local_proxy(self):
        client = llm.AnthropicClient(base_url="http://127.0.0.1:1234", api_key="k", model="claude-x")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            with self.assertRaises(llm.LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}])
        msg = str(cm.exception)
        self.assertIn("local proxy", msg)
        self.assertNotIn("firewall", msg)

    def test_url_error_against_remote_host_hints_at_network_blocking(self):
        client = llm.AnthropicClient(base_url="https://api.anthropic.com", api_key="k", model="claude-x")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            with self.assertRaises(llm.LLMError) as cm:
                client.chat([{"role": "user", "content": "hi"}])
        msg = str(cm.exception)
        self.assertIn("firewall", msg)
        self.assertNotIn("local proxy", msg)


class TestClaudeCLIClient(unittest.TestCase):
    def test_chat_parses_result_and_builds_expected_command(self):
        fake_proc = MagicMock(returncode=0, stdout=json.dumps({
            "is_error": False, "result": '```action\n{"final": "done"}\n```',
        }), stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc) as run:
            client = llm.ClaudeCLIClient(model="sonnet", timeout=30)
            reply = client.chat([
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "do it"},
            ])
        self.assertEqual(reply, '```action\n{"final": "done"}\n```')
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[0], "claude")
        self.assertIn("-p", cmd)
        self.assertIn("--restricted", cmd)
        self.assertIn("sonnet", cmd)
        # The prompt must go over stdin, not as a `-p <prompt>` argv element
        # -- an unbounded, ever-growing transcript in argv blows past
        # Windows' command-line length limit partway through a real
        # session (see the WinError 206 comment on ClaudeCLIClient.chat).
        self.assertNotIn("sys prompt", cmd)
        self.assertEqual(cmd[cmd.index("-p") + 1], "--output-format")
        prompt = run.call_args.kwargs["input"]
        self.assertIn("sys prompt", prompt)
        self.assertIn("[USER]\ndo it", prompt)

    def test_effort_omitted_from_argv_when_unset(self):
        fake_proc = MagicMock(returncode=0, stdout=json.dumps({
            "is_error": False, "result": "ok",
        }), stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc) as run:
            llm.ClaudeCLIClient(model="sonnet").chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("--effort", run.call_args[0][0])

    def test_effort_included_in_argv_when_set(self):
        fake_proc = MagicMock(returncode=0, stdout=json.dumps({
            "is_error": False, "result": "ok",
        }), stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc) as run:
            llm.ClaudeCLIClient(model="sonnet", effort="xhigh").chat([{"role": "user", "content": "hi"}])
        cmd = run.call_args[0][0]
        self.assertIn("--effort", cmd)
        self.assertEqual(cmd[cmd.index("--effort") + 1], "xhigh")

    def test_image_block_degrades_to_placeholder_text(self):
        prompt = llm.render_transcript_for_cli([
            {"role": "user", "content": [
                {"type": "text", "text": "saw something"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
            ]},
        ])
        self.assertIn("saw something", prompt)
        self.assertIn("screenshot omitted", prompt)

    def test_is_error_response_raises_llm_error(self):
        fake_proc = MagicMock(returncode=0, stdout=json.dumps({
            "is_error": True, "result": "usage limit reached",
        }), stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc):
            client = llm.ClaudeCLIClient()
            with self.assertRaises(llm.LLMError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_missing_binary_raises_llm_error_after_retries(self):
        with patch("orchestrator.llm.subprocess.run", side_effect=FileNotFoundError()) as run:
            with patch("orchestrator.llm.shutil.which", return_value=None):
                client = llm.ClaudeCLIClient(retry_delay=0)
                with self.assertRaises(llm.LLMError) as ctx:
                    client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(run.call_count, client.max_retries + 1)
        self.assertIn("finds nothing on PATH", str(ctx.exception))

    def test_transient_file_not_found_recovers_on_retry(self):
        # Mirrors the real failure mode: the binary genuinely is on PATH
        # (shutil.which finds it) but one spawn attempt transiently fails
        # -- the next attempt should succeed without raising.
        fake_proc = MagicMock(returncode=0, stdout=json.dumps({
            "is_error": False, "result": "ok after retry",
        }), stderr="")
        with patch("orchestrator.llm.subprocess.run", side_effect=[FileNotFoundError(), fake_proc]) as run:
            with patch("orchestrator.llm.shutil.which", return_value="/usr/bin/claude"):
                client = llm.ClaudeCLIClient(retry_delay=0)
                reply = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "ok after retry")
        self.assertEqual(run.call_count, 2)

    def test_persistent_failure_reports_which_found_it_anyway(self):
        with patch("orchestrator.llm.subprocess.run", side_effect=FileNotFoundError()):
            with patch("orchestrator.llm.shutil.which", return_value="C:\\Users\\x\\.local\\bin\\claude.exe"):
                client = llm.ClaudeCLIClient(retry_delay=0)
                with self.assertRaises(llm.LLMError) as ctx:
                    client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("resolves to", str(ctx.exception))
        self.assertIn("not a missing install", str(ctx.exception))

    def test_command_line_too_long_gets_its_own_diagnostic(self):
        winerror_206 = FileNotFoundError()
        winerror_206.winerror = 206  # ERROR_FILENAME_EXCED_RANGE on Windows
        with patch("orchestrator.llm.subprocess.run", side_effect=winerror_206):
            with patch("orchestrator.llm.shutil.which", return_value="C:\\claude.exe"):
                client = llm.ClaudeCLIClient(retry_delay=0)
                with self.assertRaises(llm.LLMError) as ctx:
                    client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("command line was too long", str(ctx.exception))
        self.assertIn("WinError 206", str(ctx.exception))

    def test_long_transcript_never_lands_in_argv(self):
        # The real bug this guards against: agent.py resends the entire,
        # ever-growing session transcript every turn, and it used to be
        # passed as a `-p <prompt>` argv element -- fine for a short
        # session, fatal (WinError 206) once a long one pushed the command
        # line past Windows' ~32K-character limit. It must go over stdin.
        huge_observation = "x" * 100_000
        fake_proc = MagicMock(returncode=0, stdout=json.dumps({
            "is_error": False, "result": "ok",
        }), stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc) as run:
            client = llm.ClaudeCLIClient()
            client.chat([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "```action\n{}\n```"},
                {"role": "user", "content": huge_observation},
            ])
        cmd = run.call_args[0][0]
        total_argv_chars = sum(len(a) for a in cmd)
        self.assertLess(total_argv_chars, 1000)
        self.assertIn(huge_observation, run.call_args.kwargs["input"])

    def test_nonzero_exit_raises_llm_error_with_stderr(self):
        fake_proc = MagicMock(returncode=1, stdout="", stderr="not logged in")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc):
            client = llm.ClaudeCLIClient()
            with self.assertRaises(llm.LLMError) as ctx:
                client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("not logged in", str(ctx.exception))

    def test_nonzero_exit_with_json_stdout_extracts_human_message(self):
        # The bug this guards against: `claude -p --output-format json`
        # still writes a JSON result object to stdout even when it fails
        # (a rate limit, an upstream API error, ...) -- exiting nonzero
        # used to skip straight past that and dump the raw JSON blob
        # instead of the actual human-readable `result` field.
        raw = json.dumps({
            "is_error": True, "api_error_status": 429,
            "result": "You've hit your session limit - resets 12:20am (Asia/Kolkata)",
        })
        fake_proc = MagicMock(returncode=1, stdout=raw, stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc):
            client = llm.ClaudeCLIClient()
            with self.assertRaises(llm.LLMError) as ctx:
                client.chat([{"role": "user", "content": "hi"}])
        message = str(ctx.exception)
        self.assertIn("HTTP 429", message)
        self.assertIn("session limit", message)
        self.assertNotIn("duration_api_ms", message)  # not the raw JSON dump

    def test_nonzero_exit_with_unparseable_stdout_falls_back_to_raw_dump(self):
        fake_proc = MagicMock(returncode=1, stdout="not json at all", stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc):
            client = llm.ClaudeCLIClient()
            with self.assertRaises(llm.LLMError) as ctx:
                client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("not json at all", str(ctx.exception))

    def test_is_error_with_zero_exit_also_includes_http_status(self):
        fake_proc = MagicMock(returncode=0, stdout=json.dumps({
            "is_error": True, "api_error_status": 529, "result": "Overloaded",
        }), stderr="")
        with patch("orchestrator.llm.subprocess.run", return_value=fake_proc):
            client = llm.ClaudeCLIClient()
            with self.assertRaises(llm.LLMError) as ctx:
                client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 529", str(ctx.exception))
        self.assertIn("Overloaded", str(ctx.exception))


class TestBuildClient(unittest.TestCase):
    def setUp(self):
        # Isolate from this developer machine's real, gitignored
        # orchestrator/config.json -- its base_url is set for local
        # OpenAI-compatible use and must not leak into these
        # from-scratch-defaults assertions (see the config.py fix this
        # test caught: base_url resolution has to happen *after* provider
        # is known, using real "was anything ever set" semantics, not an
        # equality check against a hardcoded literal).
        patcher = patch("orchestrator.config.CONFIG_JSON_PATH", Path("does-not-exist.json"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_provider_builds_openai_client(self):
        cfg = load_config({"repo_root": "."})
        client = llm.build_client(cfg)
        self.assertIsInstance(client, llm.LLMClient)
        self.assertEqual(client.base_url, "http://localhost:8000/v1")

    def test_claude_cli_provider_builds_claude_cli_client_with_no_key_needed(self):
        cfg = load_config({"repo_root": ".", "provider": "claude-cli", "model": "sonnet"})
        client = llm.build_client(cfg)
        self.assertIsInstance(client, llm.ClaudeCLIClient)
        self.assertEqual(client.model, "sonnet")
        self.assertIsNone(client.effort)

    def test_claude_cli_provider_passes_effort_through(self):
        cfg = load_config({"repo_root": ".", "provider": "claude-cli", "model": "sonnet", "effort": "high"})
        client = llm.build_client(cfg)
        self.assertEqual(client.effort, "high")

    def test_anthropic_provider_builds_anthropic_client_with_default_base_url(self):
        cfg = load_config({"repo_root": ".", "provider": "anthropic", "model": "claude-x"})
        client = llm.build_client(cfg)
        self.assertIsInstance(client, llm.AnthropicClient)
        self.assertEqual(client.base_url, llm.ANTHROPIC_DEFAULT_BASE_URL)

    def test_anthropic_provider_falls_back_to_env_api_key(self):
        import os
        old = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "sk-from-env"
        try:
            cfg = load_config({"repo_root": ".", "provider": "anthropic", "model": "claude-x"})
            client = llm.build_client(cfg)
            self.assertEqual(client.api_key, "sk-from-env")
        finally:
            if old is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_explicit_anthropic_base_url_is_respected(self):
        cfg = load_config({
            "repo_root": ".", "provider": "anthropic", "model": "claude-x",
            "base_url": "https://my-proxy.example/anthropic",
        })
        client = llm.build_client(cfg)
        self.assertEqual(client.base_url, "https://my-proxy.example/anthropic")

    def test_config_json_base_url_wins_even_across_a_provider_override(self):
        # base_url follows the same "explicit anywhere (file/env/CLI) wins"
        # precedence as every other config field -- it does NOT get
        # reset just because --provider was overridden without also
        # passing --base-url. A config.json written for a local
        # OpenAI-compatible server, combined with a one-off `--provider
        # anthropic` override elsewhere, is a real footgun (the Anthropic
        # call ends up pointed at the wrong host) -- but it fails loudly
        # (a connection error), and the fix is simple and explicit: pass
        # --base-url alongside --provider. See orchestrator/README.md.
        with tempfile.TemporaryDirectory() as d:
            config_path = Path(d) / "config.json"
            config_path.write_text(json.dumps({"base_url": "http://localhost:1234/v1"}), encoding="utf-8")
            with patch("orchestrator.config.CONFIG_JSON_PATH", config_path):
                cfg = load_config({"repo_root": ".", "provider": "anthropic", "model": "claude-x"})
        self.assertEqual(cfg.base_url, "http://localhost:1234/v1")


if __name__ == "__main__":
    unittest.main()
