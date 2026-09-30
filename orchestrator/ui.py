"""A minimal local web UI: start a run, watch the transcript fill in.

Same shape as `annotation_suite`'s server — stdlib `http.server`, one HTML
file, JSON endpoints, no build step and no third-party dependency. The
browser tab polls for updates rather than opening a websocket; sessions are
short and single-user, so that's simple and good enough.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import agent, db, finetune
from .config import Config, with_overrides

UI_HTML_PATH = Path(__file__).resolve().parent / "ui_page.html"
ANNOTATE_HTML_PATH = Path(__file__).resolve().parent / "annotate_page.html"


def make_handler(cfg: Config):
    class Handler(BaseHTTPRequestHandler):
        server_version = "orchestrator-ui/1"

        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, text: str, status: int = 200) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _binary(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            raw = self.rfile.read(n)
            try:
                return json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:
            parts = urllib.parse.urlsplit(self.path)
            qs = urllib.parse.parse_qs(parts.query)
            path = parts.path

            if path in ("/", "/index.html"):
                return self._html(UI_HTML_PATH.read_text(encoding="utf-8"))

            if path in ("/annotate", "/annotate.html"):
                return self._html(ANNOTATE_HTML_PATH.read_text(encoding="utf-8"))

            if path == "/api/config":
                return self._json(cfg.masked())

            if path == "/api/sessions":
                limit = int(qs.get("limit", ["50"])[0])
                con = db.connect(cfg.db_path)
                try:
                    rows = db.list_sessions(con, limit=limit)
                finally:
                    con.close()
                return self._json({"sessions": rows})

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)", path)
            if m:
                session_id = m.group(1)
                con = db.connect(cfg.db_path)
                try:
                    session = db.get_session(con, session_id)
                    if session is None:
                        return self._json({"error": "no such session"}, 404)
                    messages = db.get_messages(con, session_id)
                finally:
                    con.close()
                return self._json({"session": session, "messages": messages})

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)/messages/(\d+)/image", path)
            if m:
                session_id, seq = m.group(1), int(m.group(2))
                con = db.connect(cfg.db_path)
                try:
                    image_path = db.get_message_image_path(con, session_id, seq)
                finally:
                    con.close()
                if not image_path:
                    return self._json({"error": "no image on this message"}, 404)
                p = Path(image_path)
                if not p.is_file():
                    return self._json({"error": "image file is missing on disk"}, 404)
                mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                return self._binary(p.read_bytes(), mime)

            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path

            if path == "/api/run":
                data = self._body()
                task = str(data.get("task", "")).strip()
                if not task:
                    return self._json({"error": "task is required"}, 400)
                finetune_flag = bool(data.get("finetune", False))
                run_overrides = {}
                if "vision" in data:
                    run_overrides["vision"] = bool(data["vision"])
                if "interactive" in data:
                    run_overrides["interactive"] = bool(data["interactive"])
                if "browse" in data:
                    run_overrides["browse"] = bool(data["browse"])
                # Per-run only -- doesn't touch the server's own default cfg,
                # so leaving these blank in the form just uses whatever
                # `orchestrator ui` was started with (same as the CLI's
                # --provider/--model/--base-url/--api-key overrides).
                for key in ("provider", "model", "base_url", "api_key", "effort"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        run_overrides[key] = val.strip()
                run_cfg = with_overrides(cfg, **run_overrides) if run_overrides else cfg
                session_id = agent.start_session(run_cfg, task, finetune_flag)
                thread = threading.Thread(
                    target=agent.drive_session, args=(run_cfg, session_id), daemon=True
                )
                thread.start()
                return self._json({"session_id": session_id})

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)/answer", path)
            if m:
                session_id = m.group(1)
                data = self._body()
                answer = str(data.get("answer", "")).strip()
                if not answer:
                    return self._json({"error": "answer is required"}, 400)
                con = db.connect(cfg.db_path)
                try:
                    session = db.get_session(con, session_id)
                finally:
                    con.close()
                if session is None:
                    return self._json({"error": "no such session"}, 404)
                if session["status"] != "waiting_for_user":
                    return self._json(
                        {"error": f"session is not waiting for an answer (status: {session['status']})"}, 409
                    )
                # send_images (unlike provider/model/vision/etc.) is safe to
                # accept here even though this resumes an *existing*
                # session, not a fresh one -- it only ever removes what
                # gets sent, never grants capability, so it's how a session
                # stuck replaying a screenshot into a model that doesn't
                # accept images gets un-stuck (see agent.drive_session).
                #
                # api_key is a different case but belongs here too: it's
                # deliberately never pinned to the session row (it's a
                # secret), so a resume always needs it supplied fresh from
                # *somewhere* -- the CLI's `sessions answer --api-key` does
                # this already, but until now the UI's answer box had no
                # equivalent, silently falling back to whatever `api_key`
                # the `orchestrator ui` process itself started with (often
                # the local-server placeholder "EMPTY") and surfacing as a
                # confusing 401 from the real provider instead of the
                # session's own key.
                resume_overrides = {}
                if "send_images" in data:
                    resume_overrides["send_images"] = bool(data["send_images"])
                if isinstance(data.get("api_key"), str) and data["api_key"].strip():
                    resume_overrides["api_key"] = data["api_key"].strip()
                resume_cfg = with_overrides(cfg, **resume_overrides) if resume_overrides else cfg
                agent.answer_question(cfg, session_id, answer)
                thread = threading.Thread(
                    target=agent.drive_session, args=(resume_cfg, session_id), daemon=True
                )
                thread.start()
                return self._json({"ok": True})

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)/stop", path)
            if m:
                session_id = m.group(1)
                con = db.connect(cfg.db_path)
                try:
                    session = db.get_session(con, session_id)
                finally:
                    con.close()
                if session is None:
                    return self._json({"error": "no such session"}, 404)
                if session["status"] != "running":
                    return self._json(
                        {"error": f"session is not running (status: {session['status']})"}, 409
                    )
                # No background thread to spawn here, unlike /answer and
                # /retry -- agent.request_stop just sets the flag (or, for
                # an orphaned session with a stale heartbeat, ends it
                # directly itself). See its docstring.
                agent.request_stop(cfg, session_id)
                return self._json({"ok": True})

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)/retry", path)
            if m:
                session_id = m.group(1)
                data = self._body()
                con = db.connect(cfg.db_path)
                try:
                    session = db.get_session(con, session_id)
                finally:
                    con.close()
                if session is None:
                    return self._json({"error": "no such session"}, 404)
                if session["status"] not in ("failed", "aborted", "stopped"):
                    return self._json(
                        {"error": f"session is not in a retryable state (status: {session['status']})"}, 409
                    )
                # Same override set as /answer and for the same reasons --
                # send_images only ever removes capability (safe to vary per
                # call) and api_key is never pinned to the session at all
                # (it's a secret), so a retry after a bad-key or network
                # failure needs a place to supply a fixed one.
                resume_overrides = {}
                if "send_images" in data:
                    resume_overrides["send_images"] = bool(data["send_images"])
                if isinstance(data.get("api_key"), str) and data["api_key"].strip():
                    resume_overrides["api_key"] = data["api_key"].strip()
                resume_cfg = with_overrides(cfg, **resume_overrides) if resume_overrides else cfg
                agent.retry_session(cfg, session_id)
                thread = threading.Thread(
                    target=agent.drive_session, args=(resume_cfg, session_id), daemon=True
                )
                thread.start()
                return self._json({"ok": True})

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)/review", path)
            if m:
                session_id = m.group(1)
                data = self._body()
                con = db.connect(cfg.db_path)
                try:
                    if data.get("clear"):
                        ok = db.clear_review(con, session_id)
                    else:
                        reward = data.get("reward")
                        if reward is not None:
                            reward = int(reward)
                        note = data.get("note") or None
                        try:
                            ok = db.set_review(con, session_id, reward, note)
                        except ValueError as e:
                            return self._json({"error": str(e)}, 400)
                finally:
                    con.close()
                if not ok:
                    return self._json({"error": "no such session"}, 404)
                return self._json({"ok": True})

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)/messages/(\d+)/review", path)
            if m:
                session_id, seq = m.group(1), int(m.group(2))
                data = self._body()
                con = db.connect(cfg.db_path)
                try:
                    if data.get("clear"):
                        ok = db.clear_message_review(con, session_id, seq)
                    else:
                        reward = data.get("reward")
                        if reward is not None:
                            reward = int(reward)
                        note = data.get("note") or None
                        try:
                            ok = db.set_message_review(con, session_id, seq, reward, note)
                        except ValueError as e:
                            return self._json({"error": str(e)}, 400)
                finally:
                    con.close()
                if not ok:
                    return self._json({"error": "no such step"}, 404)
                return self._json({"ok": True})

            if path == "/api/sessions/import":
                data = self._body()
                content = data.get("content")
                if not isinstance(content, str) or not content.strip():
                    return self._json({"error": "'content' (JSONL text) is required"}, 400)
                image_dir = cfg.finetune_dir / "imported_images"
                con = db.connect(cfg.db_path)
                try:
                    report = finetune.import_lines(
                        con, content.splitlines(), image_dir,
                        overwrite=bool(data.get("overwrite", False)),
                        mark_finetune=bool(data.get("mark_finetune", True)),
                    )
                finally:
                    con.close()
                return self._json(report)

            if path == "/api/finetune/export":
                data = self._body()
                out = str(data.get("out") or (cfg.finetune_dir / "export.jsonl"))
                include_all = bool(data.get("all", False))
                statuses = None if data.get("any_status") else tuple(data.get("statuses") or ["completed"])
                con = db.connect(cfg.db_path)
                try:
                    report = finetune.export_all(
                        con, Path(out), include_all=include_all, statuses=statuses,
                        val_frac=float(data.get("val_frac") or 0.0),
                        strip_metadata=bool(data.get("strip_metadata", False)),
                        require_review=bool(data.get("require_review", False)),
                        min_reward=int(data["min_reward"]) if data.get("min_reward") not in (None, "") else None,
                    )
                finally:
                    con.close()
                return self._json({"out": out, **report})

            self._json({"error": "not found"}, 404)

        def do_DELETE(self) -> None:
            parts = urllib.parse.urlsplit(self.path)
            qs = urllib.parse.parse_qs(parts.query)
            path = parts.path

            m = re.fullmatch(r"/api/sessions/([a-fA-F0-9]+)", path)
            if m:
                session_id = m.group(1)
                delete_screenshots = qs.get("delete_screenshots", ["0"])[0] == "1"
                con = db.connect(cfg.db_path)
                try:
                    result = db.delete_session(con, session_id)
                finally:
                    con.close()
                if result is None:
                    return self._json({"error": "no such session"}, 404)

                export_path = cfg.finetune_dir / f"{session_id}.jsonl"
                removed_export = export_path.is_file()
                if removed_export:
                    export_path.unlink()

                removed_images = 0
                if delete_screenshots:
                    for p in result["image_paths"]:
                        fp = Path(p)
                        if fp.is_file():
                            fp.unlink()
                            removed_images += 1

                return self._json({
                    "ok": True, "session_id": session_id, "message_count": result["message_count"],
                    "removed_export": removed_export, "screenshot_count": len(result["image_paths"]),
                    "removed_screenshots": removed_images,
                })

            self._json({"error": "not found"}, 404)

    return Handler


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(cfg))
    url = f"http://{host}:{port}/"
    print(f"orchestrator UI serving on {url}  (Ctrl+C to stop)")
    print(f"  model: {cfg.model} @ {cfg.base_url}")
    print(f"  db: {cfg.db_path or 'orchestrator/data/orchestrator.db'}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
