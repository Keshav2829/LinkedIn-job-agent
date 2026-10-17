"""End-to-end smoke test: one full day in the life of the agent."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Agent:
    def __init__(self, db):
        self.db = db

    def __call__(self, *args):
        env = dict(os.environ, LINKEDIN_AGENT_DB=self.db, PYTHONPATH=str(ROOT))
        p = subprocess.run([sys.executable, "-m", "jobagent", *args],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        assert p.returncode == 0, f"{args}\n{p.stdout}\n{p.stderr}"
        return json.loads(p.stdout)


class TestFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ja = Agent(str(Path(cls.tmp.name) / "t.db"))
        cls.ja("init")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_init_seeds_questions(self):
        r = self.ja("init")
        self.assertTrue(r["ok"])
        self.assertGreater(self.ja("answers", "missing")["count"], 20)

    def test_02_profile_mirrors_into_answer_bank(self):
        self.ja("profile", "set", "--json", json.dumps({
            "name": "Keshav Agrawal", "current_title": "Senior Software Engineer",
            "current_company": "Acme Corp", "experience_years": 5,
            "location": "Bengaluru, India", "email": "k@example.com",
            "skills": ["Python", "Django", "AWS", "Kubernetes", "PostgreSQL"],
            "education": [{"school": "NIT Trichy", "degree": "B.Tech"}],
        }))
        self.assertEqual(self.ja("answers", "get", "--key", "current_title")["value"],
                         "Senior Software Engineer")

    def test_03_answer_resolution_handles_paraphrase(self):
        self.ja("answers", "set", "--key", "notice_period", "--value", "60 days")
        self.ja("answers", "set", "--key", "authorized_to_work", "--value", "yes")
        r = self.ja("answers", "resolve", "--question", "How soon can you join us?")
        self.assertTrue(r["found"])
        self.assertEqual(r["key"], "notice_period")

        r = self.ja("answers", "resolve", "--question",
                    "Are you legally authorized to work in India?", "--options", "Yes|No")
        self.assertEqual(r["answer"], "Yes")

        r = self.ja("answers", "resolve", "--question", "What is your favourite colour?")
        self.assertFalse(r["found"])

    def test_04_prefs_and_scoring(self):
        self.ja("prefs", "set", "--json", json.dumps({
            "target_titles": ["Senior Software Engineer", "Backend Engineer"],
            "locations": ["Bengaluru", "Remote"], "remote_ok": True,
            "company_kind": "product", "min_headcount": 200,
            "exclude_keywords": ["unpaid"], "company_blacklist": ["Bad Corp"],
        }))
        self.ja("job", "upsert", "--json", json.dumps({
            "job_id": "1001", "title": "Senior Software Engineer",
            "company_name": "Stripe", "location": "Bengaluru, India",
            "url": "https://linkedin.com/jobs/view/1001", "easy_apply": 1,
            "description": "Python, Django and AWS at scale. Kubernetes a plus.",
        }))
        good = self.ja("job", "score", "--job-id", "1001")
        self.assertGreaterEqual(good["score"], 70)
        self.assertTrue(good["auto_apply"])

        self.ja("job", "upsert", "--json", json.dumps({
            "job_id": "1002", "title": "Unpaid Marketing Intern",
            "company_name": "Bad Corp", "location": "Nowhere", "easy_apply": 0}))
        bad = self.ja("job", "score", "--job-id", "1002")
        self.assertEqual(bad["score"], 0)
        self.assertEqual(bad["status"], "skipped")

    def test_05_dedupe_and_quota(self):
        self.assertTrue(self.ja("job", "seen", "--job-id", "1001")["known"])
        self.assertFalse(self.ja("job", "seen", "--job-id", "9999")["known"])

        q = self.ja("job", "queue", "--limit", "5")
        self.assertEqual(q["count"], 1)

        self.ja("job", "applied", "--job-id", "1001", "--answers", json.dumps({"notice_period": "60 days"}))
        again = self.ja("job", "applied", "--job-id", "1001")
        self.assertTrue(again["duplicate"])
        self.assertEqual(self.ja("job", "queue")["count"], 0)
        self.assertEqual(self.ja("run", "quota")["applications"]["used"], 1)

    def test_05a_job_queue_requires_approval(self):
        self.ja("job", "upsert", "--json", json.dumps({
            "job_id": "1003", "title": "Senior Software Engineer",
            "company_name": "Stripe", "location": "Bengaluru, India",
            "url": "https://linkedin.com/jobs/view/1003", "easy_apply": 1,
            "description": "Python, Django and AWS at scale. Kubernetes a plus.",
        }))
        self.ja("job", "upsert", "--json", json.dumps({
            "job_id": "1004", "title": "Senior Software Engineer",
            "company_name": "Stripe", "location": "Bengaluru, India",
            "url": "https://linkedin.com/jobs/view/1004", "easy_apply": 1,
            "description": "Python, Django and AWS at scale. Kubernetes a plus.",
        }))
        self.assertTrue(self.ja("job", "score", "--job-id", "1003")["auto_apply"])
        self.assertTrue(self.ja("job", "score", "--job-id", "1004")["auto_apply"])

        self.assertFalse(self.ja("job", "queue")["approval_required"])
        self.ja("config", "set", "--json", json.dumps({"apply_requires_approval": True}))
        try:
            # Gate on: neither job reaches `queue`, even though both
            # auto-scored, until explicitly approved.
            gated = self.ja("job", "queue")
            self.assertEqual(gated["count"], 0)
            self.assertTrue(gated["approval_required"])
            pending = self.ja("job", "pending")
            self.assertEqual(pending["count"], 2)
            self.assertEqual({j["job_id"] for j in pending["pending"]}, {"1003", "1004"})

            self.ja("job", "approve", "--job-id", "1003")
            self.ja("job", "reject", "--job-id", "1004", "--reason", "not_interested")

            approved_queue = self.ja("job", "queue")
            self.assertEqual(approved_queue["count"], 1)
            self.assertEqual(approved_queue["jobs"][0]["job_id"], "1003")
            # Decided jobs (either way) drop out of pending.
            self.assertEqual(self.ja("job", "pending")["count"], 0)

            missing = self.ja("job", "approve", "--job-id", "no-such-job")
            self.assertFalse(missing["ok"])
        finally:
            self.ja("config", "set", "--json", json.dumps({"apply_requires_approval": False}))

        # Gate off again: `queue` reverts to filtering on 'queued', so the
        # now-'approved'/'rejected' jobs from the gated run above aren't
        # what it looks at any more -- the decision persists (not read back
        # into 'queued') even though nothing here actually applies to it,
        # to avoid perturbing the application counts later tests assert on.
        self.assertEqual(self.ja("job", "queue")["count"], 0)
        self.assertEqual(self.ja("job", "get", "--job-id", "1003")["status"], "approved")
        self.assertEqual(self.ja("job", "get", "--job-id", "1004")["status"], "rejected")

    def test_06_outreach_queue_requires_approval(self):
        self.ja("contact", "upsert", "--json", json.dumps({
            "profile_url": "https://linkedin.com/in/priya-hr", "name": "Priya Sharma",
            "company_name": "Stripe", "role_type": "hr", "degree": "2nd", "mutual_count": 4}))
        d = self.ja("outreach", "draft", "--contact-url", "https://linkedin.com/in/priya-hr",
                    "--kind", "invite", "--template", "invite_hr", "--job-id", "1001")
        self.assertEqual(d["status"], "queued")
        self.assertTrue(d["within_limit"], f"invite note too long: {d['chars']}")
        self.assertIn("Priya", d["body"])
        self.assertIn("Stripe", d["body"])

        dupe = self.ja("outreach", "draft", "--contact-url", "https://linkedin.com/in/priya-hr",
                       "--kind", "invite")
        self.assertTrue(dupe.get("duplicate"))

        self.assertEqual(self.ja("outreach", "next")["count"], 0)  # nothing approved yet
        self.ja("outreach", "approve", "--id", str(d["id"]))
        nxt = self.ja("outreach", "next")
        self.assertEqual(nxt["count"], 1)
        self.ja("outreach", "sent", "--id", str(d["id"]))
        self.assertEqual(self.ja("run", "quota")["invites"]["used"], 1)
        self.assertEqual(self.ja("contact", "seen",
                                 "--url", "https://linkedin.com/in/priya-hr")["relation"], "invited")

    def test_07_accepted_invite_triggers_followup(self):
        self.assertEqual(self.ja("contact", "pending-invites")["count"], 1)
        self.ja("contact", "mark", "--url", "https://linkedin.com/in/priya-hr",
                "--relation", "connected")
        fu = self.ja("report", "followups")
        self.assertEqual(len(fu["accepted_awaiting_message"]), 1)

        # With no attachment and no resume_url there is no way to deliver the
        # resume, so a resume-bearing message must be refused, not sent hollow.
        blocked = self.ja("outreach", "draft", "--contact-url", "https://linkedin.com/in/priya-hr",
                          "--kind", "hr_pitch", "--job-id", "1001",
                          "--extra", json.dumps({"one_specific_reason": "x"}))
        self.assertEqual(blocked.get("error"), "no_resume_channel")

        # Given a link, the message carries the link (the Browser MCP path).
        self.ja("profile", "set", "--json", json.dumps({"resume_url": "https://ex.com/cv.pdf"}))
        d = self.ja("outreach", "draft", "--contact-url", "https://linkedin.com/in/priya-hr",
                    "--kind", "hr_pitch", "--job-id", "1001",
                    "--extra", json.dumps({"one_specific_reason": "I shipped a payments ledger at scale."}))
        self.assertIn("https://ex.com/cv.pdf", d["body"])
        self.assertNotIn("attached", d["body"].lower())
        self.ja("outreach", "approve", "--all")
        self.ja("outreach", "sent", "--id", str(d["id"]))
        self.assertEqual(len(self.ja("report", "followups")["accepted_awaiting_message"]), 0)

    def test_08_replies_are_deduped_and_notified(self):
        r = self.ja("reply", "add", "--contact-url", "https://linkedin.com/in/priya-hr",
                    "--snippet", "Sure, send me your resume.", "--sentiment", "positive")
        self.assertTrue(r["is_new"])
        again = self.ja("reply", "add", "--contact-url", "https://linkedin.com/in/priya-hr",
                        "--snippet", "Sure, send me your resume.")
        self.assertFalse(again["is_new"])
        self.assertEqual(self.ja("reply", "unnotified")["count"], 1)
        self.ja("reply", "mark-notified", "--all")
        self.assertEqual(self.ja("reply", "unnotified")["count"], 0)

    def test_085_contact_url_normalisation(self):
        """A trailing slash or tracking param must not create a second contact."""
        a = self.ja("contact", "seen", "--url", "https://linkedin.com/in/priya-hr/")
        b = self.ja("contact", "seen", "--url", "https://linkedin.com/in/priya-hr?trk=abc")
        self.assertTrue(a["known"] and b["known"])
        self.assertEqual(a["id"], b["id"])

    def test_09_reports(self):
        p = self.ja("report", "pipeline")
        self.assertEqual(p["applications"], 1)
        self.assertEqual(p["contacts"]["connected"], 1)
        d = self.ja("report", "daily")
        self.assertEqual(len(d["applied_today"]), 1)
        self.assertEqual(len(d["sent_today"]), 2)

    def test_10_trace_step_vocabulary_is_closed(self):
        """An invented step name splits examples off a canonical step. The
        first corpus accumulated five of them before anything checked."""
        ok = self.ja("trace", "log", "--json", json.dumps({
            "skill": "linkedin-apply", "step": "score_job", "task_id": "job_1",
            "plan_step": "score the open job", "instruction": "score it",
            "action": {"kind": "tool_call",
                       "tool": {"name": "job.score", "args": {"job_id": "job_1"},
                                "result": {"score": 80}}},
            "outcome": "success"}))
        self.assertTrue(ok["ok"], ok)

        bad = self.ja("trace", "log", "--json", json.dumps({
            "skill": "linkedin-apply", "step": "clicked_the_blue_button",
            "task_id": "job_1"}))
        self.assertFalse(bad["ok"])
        self.assertIn("closed vocabulary", bad["error"])

        bad_skill = self.ja("trace", "log", "--json", json.dumps({
            "skill": "linkedin-magic", "step": "open_job", "task_id": "job_1"}))
        self.assertFalse(bad_skill["ok"])

    def test_11_trace_aliases_and_task_id_normalisation(self):
        """The five names that already leaked are rewritten, not lost. Bare
        numeric and job-URL task_ids collapse onto the canonical job_<id> so
        `trace task` can group an episode."""
        r = self.ja("trace", "log", "--json", json.dumps({
            "skill": "linkedin-apply", "step": "select_resume",
            "task_id": "4455863129", "plan_step": "confirm the stored resume matches profile",
            "instruction": "pick resume",
            "action": {"kind": "read"},
            "observation": {"url": "https://www.linkedin.com/jobs/view/4455863129/",
                             "a11y": "pages/test-select-resume.a11y.yaml",
                             "page_text": "pages/test-select-resume.txt"},
            "driver": "claude-in-chrome",
            "outcome": "success"}))
        self.assertTrue(r["ok"], r)
        self.assertTrue(any("normalized" in w for w in r["warnings"]))

        t = self.ja("trace", "task", "--task-id", "job_4455863129")
        self.assertEqual([s["step"] for s in t["steps"]], ["upload_resume"])

        r2 = self.ja("trace", "log", "--json", json.dumps({
            "skill": "linkedin-apply", "step": "open_job",
            "task_id": "https://www.linkedin.com/jobs/view/4445891402/",
            "plan_step": "open the next queued job", "instruction": "open it",
            "action": {"kind": "navigate", "value": "https://www.linkedin.com/jobs/view/4445891402/"},
            "observation": {"url": "https://www.linkedin.com/jobs/view/4445891402/",
                             "a11y": "pages/test-open-job.a11y.yaml",
                             "page_text": "pages/test-open-job.txt"},
            "driver": "claude_in_chrome"}))
        self.assertTrue(r2["ok"], r2)
        t2 = self.ja("trace", "task", "--task-id", "job_4445891402")
        self.assertEqual(len(t2["steps"]), 1)

    def test_115_backfill_targets_one_record_when_a_step_repeats(self):
        """fill_screening_field / apply_filter / retry chains repeat a step
        inside one episode. Without step_index the backfill silently edited
        the LAST such record."""
        for i, val in enumerate(["Easy Apply", "Past week", "Remote"]):
            r0 = self.ja("trace", "log", "--json", json.dumps({
                "skill": "linkedin-apply", "step": "apply_filter",
                "task_id": "search_rep", "plan_step": "narrow to Easy Apply, last week, remote",
                "instruction": f"toggle {val}",
                "action": {"kind": "read", "value": val}, "driver": "claude_in_chrome",
                "observation": {"url": "https://www.linkedin.com/jobs/search/",
                                 "a11y": "pages/test-search.a11y.yaml",
                                 "page_text": "pages/test-search.txt"},
                "outcome": "success", "action_value": val}))
            self.assertTrue(r0["ok"], r0)
        steps = self.ja("trace", "task", "--task-id", "search_rep")["steps"]
        self.assertEqual([s["action_value"] for s in steps],
                         ["Easy Apply", "Past week", "Remote"])

        # target the FIRST one explicitly
        r = self.ja("trace", "log", "--json", json.dumps({
            "backfill": True, "skill": "linkedin-apply", "step": "apply_filter",
            "task_id": "search_rep", "step_index": 0,
            "action": {"coords": {"x": 334, "y": 80}},
            "observation": {"viewport": {"width": 1536, "height": 639}}}))
        self.assertTrue(r["ok"], r)
        self.assertTrue(r.get("backfilled"))

        steps = self.ja("trace", "task", "--task-id", "search_rep")["steps"]
        self.assertEqual(len(steps), 3, "backfill must not append a duplicate")
        self.assertEqual(steps[0]["action"]["coords"], {"x": 334, "y": 80})
        self.assertEqual(steps[0]["action_value"], "Easy Apply")
        for s in steps[1:]:
            self.assertNotIn("coords", s.get("action", {}), "only the targeted record may change")

    def test_12_trace_logging_can_be_switched_off(self):
        self.assertFalse(self.ja("trace", "disable")["trace_logging"])
        skipped = self.ja("trace", "log", "--json", json.dumps({
            "skill": "linkedin-apply", "step": "open_job", "task_id": "job_off"}))
        self.assertTrue(skipped["ok"])
        self.assertIn("disabled", skipped["skipped"])
        self.assertEqual(self.ja("trace", "task", "--task-id", "job_off")["steps"], [])
        self.assertTrue(self.ja("trace", "enable")["trace_logging"])

    def test_13_stats_names_the_phases_never_logged(self):
        """The search phase was missing from 154 real records and record
        count alone never showed it."""
        s = self.ja("trace", "stats")
        self.assertEqual(s["off_vocabulary_steps"], [])
        self.assertIn("search_jobs", s["never_logged_steps"])
        self.assertNotIn("score_job", s["never_logged_steps"])


class TestBrowse(unittest.TestCase):
    """The `browse` group drives a real Chrome, so the parts that need one are
    skipped unless a debugging endpoint is actually listening. What is always
    checked is the contract that matters when it is *not*: one JSON object on
    stdout, never a traceback."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = str(Path(cls.tmp.name) / "b.db")
        Agent(cls.db)("init")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_raw(self, *args):
        """Like Agent, but tolerates a non-zero exit — that is the case here."""
        env = dict(os.environ, LINKEDIN_AGENT_DB=self.db, PYTHONPATH=str(ROOT))
        p = subprocess.run([sys.executable, "-m", "jobagent", "browse", *args],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        self.assertEqual(p.stderr.strip(), "", f"browse wrote to stderr: {p.stderr}")
        return json.loads(p.stdout)

    def test_01_unreachable_chrome_is_a_clean_json_error(self):
        r = self.run_raw("status", "--cdp", "http://127.0.0.1:9")
        self.assertFalse(r["ok"])
        self.assertIn("remote-debugging-port", r["error"])

    def test_02_missing_arguments_do_not_crash(self):
        for args in (("open",), ("find",), ("fill", "--ref", "f1"), ("dump",)):
            with self.subTest(args=args):
                r = self.run_raw(*args)
                self.assertFalse(r["ok"])
                self.assertIn("needs", r["error"])

    def test_03_every_verb_is_reachable(self):
        from jobagent.cli import build_parser
        verbs = build_parser()._subparsers._group_actions[0].choices["browse"] \
            ._actions[1].choices
        for v in ("status", "open", "outline", "find", "form", "click", "fill",
                  "select", "check", "upload", "press", "scroll", "text", "shot", "dump"):
            self.assertIn(v, verbs)

    def test_04_live_page_returns_a_small_answer(self):
        """The whole point: a find on a busy page costs hundreds of bytes,
        not the hundred thousand a whole-tree snapshot costs."""
        import urllib.request
        cdp = os.environ.get("JOBAGENT_CDP", "http://127.0.0.1:9222")
        try:
            urllib.request.urlopen(cdp + "/json/version", timeout=2).read()
        except Exception:
            self.skipTest(f"no Chrome on {cdp}")
        r = self.run_raw("status")
        self.assertTrue(r["ok"])
        self.assertLess(len(json.dumps(r)), 4000)

    def test_05_outline_finds_the_list_and_stays_small(self):
        """`outline` is the bounded answer to "what is on this page" — the one
        thing find/form cannot give you. It has to pick the results list out
        of the nav, sidebar and footer around it, and stay small doing it.

        Opt-in via JOBAGENT_TEST_CDP: this navigates a tab, and nobody wants
        `pytest` yanking their real browser to a fixture page.
        """
        cdp = os.environ.get("JOBAGENT_TEST_CDP")
        if not cdp:
            self.skipTest("set JOBAGENT_TEST_CDP to run the live outline test")

        page = Path(self.tmp.name) / "results.html"
        page.write_text("""<!doctype html><title>jobs</title>
          <nav><ul><li><a href=#>Home</a></li><li><a href=#>Jobs</a></li>
                   <li><a href=#>Me</a></li></ul></nav>
          <main><h1>AI Engineer jobs</h1><ul>
            <li><a href="https://example.com/jobs/view/1/">Senior AI Engineer</a>
                <div>Test Corp - Bengaluru - Easy Apply - 3 days ago</div></li>
            <li><a href="https://example.com/jobs/view/2/">LLM Agents Engineer</a>
                <div>Nimbus Labs - Remote - Easy Apply - 1 week ago</div></li>
            <li><a href="https://example.com/jobs/view/3/">ML Engineer II</a>
                <div>Orbit Systems - Hyderabad - 2 days ago</div></li>
          </ul></main>
          <footer><ul><li><a href=#>About</a></li><li><a href=#>Help</a></li>
                      <li><a href=#>Privacy</a></li></ul></footer>""", encoding="utf-8")

        opened = self.run_raw("open", "--url", page.as_uri(), "--cdp", cdp, "--new-tab")
        self.assertTrue(opened["ok"], opened.get("error"))
        r = self.run_raw("outline", "--cdp", cdp, "--tab", "results.html")

        self.assertTrue(r["ok"])
        self.assertEqual(len(r["rows"]), 3, "picked the wrong list")
        self.assertIn("Senior AI Engineer", r["rows"][0]["text"])
        # The href is what dedupe runs on; losing it costs a page load per job.
        self.assertTrue(r["rows"][0]["href"].endswith("/jobs/view/1/"))
        self.assertIn("AI Engineer jobs", r["headings"])
        self.assertFalse(r["dialog_open"])
        self.assertLess(len(json.dumps(r)), 6000, "outline must stay bounded")

    def test_06_scope_follows_a_dialog_inside_an_open_shadow_root(self):
        """LinkedIn renders some of its own overlays -- the connect-request
        "Add a note / Send without a note" popup among them -- inside an
        open shadow root (its own `#interop-outlet`, not a browser
        extension). Plain `document.querySelectorAll` never descends into
        a shadow root, so `scope()` used to miss the dialog entirely and
        find/text fell back to `document.body`, matching the page *behind*
        the overlay -- the bug this guards. A decoy with the same button
        text sits in the light DOM; `find`/`text` must resolve to the
        shadow-hosted dialog's copy instead.

        Opt-in via JOBAGENT_TEST_CDP, same as test_05.
        """
        cdp = os.environ.get("JOBAGENT_TEST_CDP")
        if not cdp:
            self.skipTest("set JOBAGENT_TEST_CDP to run the live overlay test")

        page = Path(self.tmp.name) / "overlay.html"
        page.write_text("""<!doctype html><title>profile</title>
          <main>
            <button>Send</button>
            <div>Background noise that is not the real target.</div>
          </main>
          <div id="overlay-outlet"></div>
          <script>
            const root = document.getElementById('overlay-outlet').attachShadow({mode: 'open'});
            root.innerHTML = `
              <div role="dialog" style="position:fixed;top:40px;left:40px;
                   width:400px;height:220px;background:#fff;">
                <h2>Add a note</h2>
                <textarea>Hi there, I would love to connect.</textarea>
                <button>Send without a note</button>
              </div>`;
          </script>""", encoding="utf-8")

        opened = self.run_raw("open", "--url", page.as_uri(), "--cdp", cdp, "--new-tab")
        self.assertTrue(opened["ok"], opened.get("error"))

        found = self.run_raw("find", "--query", "send", "--cdp", cdp, "--tab", "overlay.html")
        self.assertTrue(found["ok"])
        self.assertTrue(found["matches"], "shadow-hosted dialog button should have matched")
        self.assertEqual(found["matches"][0]["name"], "Send without a note",
                          "matched the light-DOM decoy instead of the shadow-hosted dialog's button")

        text = self.run_raw("text", "--cdp", cdp, "--tab", "overlay.html")
        self.assertTrue(text["ok"])
        self.assertIn("Add a note", text["text"])
        self.assertNotIn("Background noise", text["text"])

    def test_07_outline_row_target_skips_a_small_nested_button(self):
        """LinkedIn's own messaging list has no `<a href>` per row at all --
        the row's click target is a plain `<div tabindex="0">` sized to the
        whole row, and the only real `<button>` inside is a small "open
        options" kebab menu. The old fallback (`a[href] -> button -> el`)
        grabbed that kebab for every row, so clicking a conversation opened
        a dropdown instead of the thread -- the bug this guards. The
        stamped ref must resolve to the row-sized div, not the button.

        Opt-in via JOBAGENT_TEST_CDP, same as test_05.
        """
        cdp = os.environ.get("JOBAGENT_TEST_CDP")
        if not cdp:
            self.skipTest("set JOBAGENT_TEST_CDP to run the live row-target test")

        page = Path(self.tmp.name) / "rows.html"
        page.write_text("""<!doctype html><title>inbox</title>
          <style>
            .row { position: relative; width: 300px; height: 90px; }
            .row-link { position: absolute; inset: 0; cursor: pointer; }
            .kebab { position: absolute; top: 4px; right: 4px; width: 24px; height: 24px; }
          </style>
          <ul>
            <li class="row" id="row1">
              <div class="row-link" tabindex="0" onclick="document.title='opened-1'">Alice Jun 1 Hi there</div>
              <button class="kebab" onclick="document.title='menu-1'">...</button>
            </li>
            <li class="row" id="row2">
              <div class="row-link" tabindex="0" onclick="document.title='opened-2'">Bob Jun 2 Hello</div>
              <button class="kebab" onclick="document.title='menu-2'">...</button>
            </li>
            <li class="row" id="row3">
              <div class="row-link" tabindex="0" onclick="document.title='opened-3'">Cara Jun 3 Hey</div>
              <button class="kebab" onclick="document.title='menu-3'">...</button>
            </li>
          </ul>""", encoding="utf-8")

        opened = self.run_raw("open", "--url", page.as_uri(), "--cdp", cdp, "--new-tab")
        self.assertTrue(opened["ok"], opened.get("error"))

        r = self.run_raw("outline", "--cdp", cdp, "--tab", "rows.html")
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["rows"]), 3)

        clicked = self.run_raw("click", "--ref", r["rows"][1]["ref"], "--cdp", cdp, "--tab", "rows.html")
        self.assertTrue(clicked["ok"], clicked.get("error"))

        status = self.run_raw("status", "--cdp", cdp, "--tab", "rows.html")
        self.assertEqual(status["active"]["title"], "opened-2",
                          "clicked the row's kebab menu instead of the row itself")

    def test_08_scroll_moves_an_inner_container_not_just_window(self):
        """LinkedIn almost never scrolls `window` -- the real scrollbar sits
        on an inner container under a fixed header (`<main id="workspace">`
        on the invitation manager, messaging list, etc.), so
        `document.scrollingElement` never moves. The old `scroll` wheeled
        whatever the mouse's default (0, 0) position happened to be over --
        the fixed header, not the content -- so nothing on the page moved.
        This fixture reproduces that shape with an onscroll handler as a
        side-effect probe; the bug this guards is a title that never changes.

        Opt-in via JOBAGENT_TEST_CDP, same as test_05.
        """
        cdp = os.environ.get("JOBAGENT_TEST_CDP")
        if not cdp:
            self.skipTest("set JOBAGENT_TEST_CDP to run the live scroll test")

        page = Path(self.tmp.name) / "innerscroll.html"
        items = "".join(f'<div class="item">item {i}</div>' for i in range(40))
        page.write_text(f"""<!doctype html><title>list</title>
          <style>
            html, body {{ margin: 0; height: 100%; }}
            header {{ position: fixed; top: 0; left: 0; right: 0; height: 50px;
                      background: #eee; z-index: 2; }}
            main {{ position: absolute; top: 50px; left: 0; right: 0; bottom: 0;
                    overflow-y: auto; }}
            .item {{ height: 60px; border-bottom: 1px solid #ccc; }}
          </style>
          <header>fixed header, not the scroll container</header>
          <main id="workspace"
                onscroll="document.title = 'scrolled-' + Math.round(this.scrollTop)">
            {items}
          </main>""", encoding="utf-8")

        opened = self.run_raw("open", "--url", page.as_uri(), "--cdp", cdp, "--new-tab")
        self.assertTrue(opened["ok"], opened.get("error"))

        scrolled = self.run_raw("scroll", "--dy", "600", "--cdp", cdp, "--tab", "innerscroll.html")
        self.assertTrue(scrolled["ok"], scrolled.get("error"))

        status = self.run_raw("status", "--cdp", cdp, "--tab", "innerscroll.html")
        title = status["active"]["title"]
        self.assertTrue(title.startswith("scrolled-"),
                         f"scroll never reached the inner container, title is {title!r}")
        self.assertGreater(int(title.split("-", 1)[1]), 0,
                            "onscroll fired but scrollTop did not advance")

    def test_09_fill_types_newlines_without_submitting_a_richtext_box(self):
        """LinkedIn's message compose box is a contenteditable richtext
        editor that binds a bare Enter to *submit* (there's a "Press Enter
        to send" toggle right next to it) -- Shift+Enter is "new line".
        `fill` used to hand the whole multi-line value to Playwright's
        `.type()`, which turns each "\n" into a literal Enter keypress, so
        it could submit a half-typed body instead of inserting a line
        break. This fixture reproduces that exact keydown contract (bare
        Enter -> preventDefault + record a premature "submit") and the bug
        this guards is that side effect firing at all.

        Opt-in via JOBAGENT_TEST_CDP, same as test_05.
        """
        cdp = os.environ.get("JOBAGENT_TEST_CDP")
        if not cdp:
            self.skipTest("set JOBAGENT_TEST_CDP to run the live fill test")

        page = Path(self.tmp.name) / "compose.html"
        page.write_text("""<!doctype html><title>compose</title>
          <div contenteditable="true" role="textbox" aria-label="Write a message"
               id="box"></div>
          <script>
            document.getElementById('box').addEventListener('keydown', (e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.title = 'submitted-early';
              }
            });
          </script>""", encoding="utf-8")

        opened = self.run_raw("open", "--url", page.as_uri(), "--cdp", cdp, "--new-tab")
        self.assertTrue(opened["ok"], opened.get("error"))

        form = self.run_raw("form", "--cdp", cdp, "--tab", "compose.html")
        self.assertTrue(form["ok"], form.get("error"))
        field = next(f for f in form["fields"] if "write a message" in f["label"].lower())

        body = "line one\nline two\nline three"
        filled = self.run_raw("fill", "--ref", field["ref"], "--value", body,
                               "--cdp", cdp, "--tab", "compose.html")
        self.assertTrue(filled["ok"], filled.get("error"))
        self.assertTrue(filled["matches"],
                         f"typed body did not round-trip: {filled['value']!r}")

        status = self.run_raw("status", "--cdp", cdp, "--tab", "compose.html")
        self.assertNotEqual(status["active"]["title"], "submitted-early",
                             "a bare Enter fired mid-type -- the box would have submitted early")

    def test_10_fill_value_accepts_a_literal_backslash_n_from_the_shell(self):
        """No shell -- bash, PowerShell, cmd -- turns a literal `\\n` typed
        inside a quoted `--value` into a real newline; all three hand it to
        the process as the two characters backslash + n. That is exactly
        how a human or an LLM-driven agent naturally writes a multi-line
        message body on one command line, so the CLI decodes it before it
        ever reaches `fill`. This test builds the argv the way a shell
        actually would -- a Python string containing a literal backslash
        and 'n', not a real newline -- to catch a regression where that
        decode is lost and the box ends up with the literal text "\\n" in
        it instead of a line break.

        Opt-in via JOBAGENT_TEST_CDP, same as test_05.
        """
        cdp = os.environ.get("JOBAGENT_TEST_CDP")
        if not cdp:
            self.skipTest("set JOBAGENT_TEST_CDP to run the live fill test")

        page = Path(self.tmp.name) / "compose2.html"
        page.write_text("""<!doctype html><title>compose2</title>
          <div contenteditable="true" role="textbox" aria-label="Write a message"
               id="box"></div>""", encoding="utf-8")

        opened = self.run_raw("open", "--url", page.as_uri(), "--cdp", cdp, "--new-tab")
        self.assertTrue(opened["ok"], opened.get("error"))

        form = self.run_raw("form", "--cdp", cdp, "--tab", "compose2.html")
        field = next(f for f in form["fields"] if "write a message" in f["label"].lower())

        # literal backslash-n, two characters, not an embedded newline -- the
        # shape a shell argument actually has.
        shell_value = "line one\\nline two\\nline three"
        filled = self.run_raw("fill", "--ref", field["ref"], "--value", shell_value,
                               "--cdp", cdp, "--tab", "compose2.html")
        self.assertTrue(filled["ok"], filled.get("error"))
        self.assertNotIn("\\n", filled["value"],
                          "literal backslash-n was typed as text instead of decoded to a newline")
        self.assertEqual(filled["value"].count("\n"), 2,
                          f"expected two real line breaks, got {filled['value']!r}")


class TestHarvest(unittest.TestCase):
    """`trace log` records what the agent remembered; harvest records what
    happened. These build a miniature transcript and check the parts that
    are easy to get quietly wrong: the thinking placeholder, redaction, and
    the click coordinates that made the 4 Sep corpus untrainable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = str(self.root / "data" / "state.db")
        (self.root / "data" / "traces").mkdir(parents=True)
        Agent(self.db)("init")

        def msg(role, content, ts, **kw):
            return json.dumps({"type": role, "timestamp": ts, "cwd": str(self.root),
                               "sessionId": "s1", "version": "2.1.251",
                               "message": {"role": role, "model": "claude-sonnet-5",
                                           "content": content}, **kw})

        # thinking present-but-stripped, then a real click with coordinates
        lines = [
            msg("user", "Apply to one job", "2026-09-05T10:00:00.000Z"),
            msg("assistant", [
                {"type": "thinking", "thinking": "", "signature": "opaque"},
                {"type": "text", "text": "Easy Apply is visible. Clicking it."},
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "python -m jobagent browse click --ref e1"}},
            ], "2026-09-05T10:00:05.000Z"),
            msg("user", [{"type": "tool_result", "tool_use_id": "t1",
                          "content": json.dumps({"ok": True, "clicked": "e1",
                                                 "name": "Easy Apply",
                                                 "coords": [941, 84],
                                                 "bbox": [884, 68, 113, 32],
                                                 "scroll": [0, 0],
                                                 "viewport": [1034, 646]})}],
                "2026-09-05T10:00:07.000Z"),
            # no thinking block at all, and PII in the typed value
            msg("assistant", [
                {"type": "tool_use", "id": "t2", "name": "Bash",
                 "input": {"command": "python -m jobagent browse fill --ref f1 "
                                      "--value 9876543210"}},
            ], "2026-09-05T10:00:12.000Z"),
            msg("user", [{"type": "tool_result", "tool_use_id": "t2",
                          "content": json.dumps({"ok": True, "ref": "f1",
                                                 "value": "9876543210",
                                                 "error": ""})}],
                "2026-09-05T10:00:14.000Z"),
        ]
        (self.root / "session.jsonl").write_text("\n".join(lines), encoding="utf-8")

        (self.root / "data" / "traces" / "2026-09-05.jsonl").write_text("\n".join([
            json.dumps({"skill": "linkedin-apply", "step": "_task_start",
                        "task_id": "job_1", "run_id": 1, "goal": "apply to one job",
                        "ts": "2026-09-05T10:00:00+00:00"}),
            json.dumps({"skill": "linkedin-apply", "step": "_task_end",
                        "task_id": "job_1", "outcome": "applied",
                        "ts": "2026-09-05T10:00:20+00:00"}),
        ]), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def harvest(self, *extra):
        env = dict(os.environ, LINKEDIN_AGENT_DB=self.db, PYTHONPATH=str(ROOT))
        p = subprocess.run([sys.executable, "-m", "jobagent", "harvest",
                            "--transcript", str(self.root / "session.jsonl"),
                            "--out", str(self.root / "data" / "sft"), *extra],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        assert p.returncode == 0, p.stdout + p.stderr
        manifest = json.loads(p.stdout)
        rows = [json.loads(l) for l in
                open(manifest["out"], encoding="utf-8") if l.strip()]
        return manifest, rows

    def test_01_every_tool_call_becomes_a_record(self):
        m, rows = self.harvest()
        self.assertEqual(m["records"], 2)
        self.assertEqual([r["action"]["kind"] for r in rows],
                         ["browse.click", "browse.fill"])

    def test_02_thinking_states_are_explicit(self):
        _, rows = self.harvest()
        self.assertEqual(rows[0]["thinking"]["state"], "stripped")
        self.assertEqual(rows[0]["thinking"]["placeholder"], "<THINKING_STRIPPED>")
        self.assertEqual(rows[1]["thinking"]["state"], "absent")
        # A later generation pass fills text and flips this; until then a
        # synthesised rationale can never be mistaken for a real one.
        self.assertFalse(rows[0]["thinking"]["generated"])
        self.assertIsNone(rows[0]["thinking"]["text"])

    def test_03_rationale_is_the_text_that_preceded_the_call(self):
        _, rows = self.harvest()
        self.assertIn("Clicking it", rows[0]["rationale"])
        self.assertIsNone(rows[1]["rationale"])

    def test_04_click_coordinates_travel_with_their_frame(self):
        """A coordinate without scroll+viewport names a place on a screen,
        not on a page — the exact defect the 4 Sep audit found."""
        _, rows = self.harvest()
        eff = rows[0]["action"]["effect"]
        self.assertEqual(eff["coords"], [941, 84])
        self.assertEqual(eff["bbox"], [884, 68, 113, 32])
        self.assertEqual(eff["scroll"], [0, 0])
        self.assertEqual(eff["viewport"], [1034, 646])

    def test_05_pii_is_redacted_by_default(self):
        _, rows = self.harvest()
        blob = json.dumps(rows[1])
        self.assertNotIn("9876543210", blob)
        self.assertIn("<PHONE>", blob)
        _, raw = self.harvest("--no-redact")
        self.assertIn("9876543210", json.dumps(raw[1]))

    def test_06_trace_labels_are_joined_by_time(self):
        _, rows = self.harvest()
        for r in rows:
            self.assertEqual(r["task"]["task_id"], "job_1")
            self.assertEqual(r["task"]["skill"], "linkedin-apply")
            self.assertEqual(r["task"]["goal"], "apply to one job")

    def test_07_observation_is_what_the_model_actually_saw(self):
        _, rows = self.harvest()
        # step 2's observation is step 1's result, not a re-read of the page
        self.assertIn("Easy Apply", rows[1]["observation"]["text"])
        self.assertEqual(rows[1]["observation"]["kind"], "result_of:browse.click")

    def test_08_messages_format_is_sft_ready(self):
        m, rows = self.harvest("--format", "messages")
        self.assertTrue(m["out"].endswith(".messages.jsonl"))
        roles = [x["role"] for x in rows[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertIn("<THINKING_STRIPPED>", rows[0]["messages"][2]["content"])
        self.assertIn("browse click --ref e1", rows[0]["messages"][2]["content"])

    def test_09_kind_filter(self):
        m, rows = self.harvest("--kind", "browse.fill")
        self.assertEqual([r["action"]["kind"] for r in rows], ["browse.fill"])


class TestBrowserResilience(unittest.TestCase):
    """The two failures a real 5 Sep run hit, pinned so they stay fixed.

    Both are driven through fakes rather than a browser: the point is the
    retry logic, and a test that needs Chrome running is a test that gets
    skipped.
    """

    def test_01_a_read_survives_the_page_navigating_underneath_it(self):
        from jobagent import browser as B

        class FakePage:
            def __init__(self):
                self.calls = 0
                self.waited = False

            def evaluate(self, script, arg=None):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError(
                        "Page.evaluate: Execution context was destroyed, "
                        "most likely because of a navigation")
                return {"ok": True, "call": self.calls}

            def wait_for_load_state(self, *a, **kw):
                self.waited = True

            def wait_for_timeout(self, ms):
                pass

        page = FakePage()
        self.assertEqual(B._eval(page, "() => 1")["ok"], True)
        self.assertTrue(page.waited, "should wait for the new document first")
        # 1 failed read, 1 re-prime of the helpers, 1 successful retry
        self.assertGreaterEqual(page.calls, 2)

    def test_02_a_real_error_is_raised_at_once_not_retried(self):
        from jobagent import browser as B

        class Broken:
            def __init__(self):
                self.calls = 0

            def evaluate(self, script, arg=None):
                self.calls += 1
                raise RuntimeError("ReferenceError: nope is not defined")

            def wait_for_load_state(self, *a, **kw):
                raise AssertionError("must not wait on a non-navigation error")

            def wait_for_timeout(self, ms):
                pass

        page = Broken()
        with self.assertRaises(RuntimeError):
            B._eval(page, "() => nope")
        self.assertEqual(page.calls, 1, "a broken script should fail fast")

    def test_03_navigation_errors_are_matched_case_insensitively(self):
        from jobagent import browser as B
        for msg in ("Execution context was destroyed",
                    "Target closed", "frame was detached",
                    "Cannot find context with specified id"):
            self.assertTrue(
                any(t in msg.lower() for t in B._NAV_ERRORS),
                f"{msg!r} should be treated as a navigation error")

    def test_04_timeout_default_is_generous_and_overridable(self):
        from jobagent import browser as B
        before = B.DEFAULT_TIMEOUT_MS
        try:
            # 8s lost a real click on 5 Sep; the default must be above it.
            self.assertGreaterEqual(before, 15000)
            B.set_timeout(30000)
            self.assertEqual(B.DEFAULT_TIMEOUT_MS, 30000)
            B.set_timeout(None)
            self.assertEqual(B.DEFAULT_TIMEOUT_MS, 30000, "None must not reset it")
            B.set_timeout(0)
            self.assertEqual(B.DEFAULT_TIMEOUT_MS, 30000, "0 must not disable it")
        finally:
            B.DEFAULT_TIMEOUT_MS = before

    def test_05_a_persistently_intercepted_click_falls_back_to_force(self):
        """A shadow-DOM overlay (a browser extension in the CDP profile, not
        LinkedIn's own UI) that covers the target on every attempt is not the
        transient "not yet settled" case the retry loop was built for —
        waiting and re-locating never clears it. The last attempt must
        bypass Playwright's actionability check instead of just giving up."""
        from contextlib import contextmanager

        from jobagent import browser as B

        class FakeLocator:
            def __init__(self):
                self.click_calls = []

            def inner_text(self):
                return "Apply"

            def get_attribute(self, name):
                return None

            def scroll_into_view_if_needed(self):
                pass

            def bounding_box(self):
                return {"x": 10, "y": 20, "width": 100, "height": 30}

            def click(self, force=False):
                self.click_calls.append(force)
                if not force:
                    raise RuntimeError(
                        "Locator.click: Timeout 15000ms exceeded.\n"
                        '<div id="interop-shadowdom">...</div> intercepts pointer events'
                    )

        class FakePage:
            def wait_for_timeout(self, ms):
                pass

            def wait_for_load_state(self, *a, **kw):
                pass

        loc = FakeLocator()
        page = FakePage()

        @contextmanager
        def fake_session(cdp_url=None, tab=None):
            yield page, None

        state = {"url": "https://linkedin.com/x", "scroll": [0, 0], "viewport": [1000, 800]}

        orig = (B.session, B._prime, B._locate, B._state)
        B.session, B._prime = fake_session, (lambda p: None)
        B._locate = lambda p, ref, db_path=None: loc
        B._state = lambda p: state
        try:
            res = B.click(ref="e1")
        finally:
            B.session, B._prime, B._locate, B._state = orig

        self.assertTrue(res["ok"])
        self.assertEqual(res["attempts"], 2)
        self.assertEqual(loc.click_calls, [False, True],
                          "second attempt must pass force=True, not retry plain")


if __name__ == "__main__":
    unittest.main(verbosity=2)
