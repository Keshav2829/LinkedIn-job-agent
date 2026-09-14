"""For every assertion in the corpus, ask: what in the record could have produced it?

Not "is the field present" and not "is the pair complete" — the narrower
question the data has to answer before any of it is trustworthy:

    this record claims X. Where did X come from?

Six answers, in descending order of trust:

  SELF     derivable from the record or its episode (URL params, the action
           taken, counts of sibling records, a tool's own error string)
  CITED    a `retrieval` entry in this record supports it
  IN_SHOT  a page fact that is visibly present in the attached screenshot -
           a human (or OCR) can check it, though nothing links the claim to
           the element it came from, and no text form is stored
  EP_SHOT  a page fact whose only evidence is a screenshot attached to a
           DIFFERENT record of the same episode
  NO_SHOT  a page fact, and neither this record nor its episode has a
           screenshot - unverifiable from the corpus
  RECALLED requires knowledge that is on no page and in no store - it came
           from the annotating model's priors
"""
import json, sys, collections

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

# Where each asserted key's evidence would have to live.
SELF = {
    "decision","next_action","proceeding","self_cap","job_id","parsed","raw_request","attended",
    "filters_in_url","url_after_load","sortBy_dropped_by_linkedin","sortBy_requested_but_dropped",
    "filter","value","url_param_confirmed","failure","tool_error","cause","recovery","coords_are",
    "recovered_from","note","rule","dismissed_by","grounding_note","exact_match_for_query",
    "searches","candidates_screened","jobs_scored","applied","rejected","shortlisted","handed_off",
    "candidates_seen","queued_for_review","skipped","skipped_searches","not_pursued","total_steps",
    "purpose","verification","interruptions","phase_reached","reason","approval","approval_reason",
    "user_approved","auto_apply","pages_completed","apply_shape","filters_applied","applications",
    "scorer_decision","auto_apply_threshold","review_threshold","expected","options_offered",
    "confirm_button","count_source","alternatives_offered","accepted_default","page","chars",
}
CITED = {"score","match_score","quota_left","seniority_vs_profile","seniority_check","selected","dated"}
PAGE = {
    "result_count","result_count_before","result_count_after","result_count_before_filters",
    "result_count_after_filters","results_count_text","exact_matches","substitution_banner",
    "returned_titles_sample","filter_pills_offered","location_auto_applied","pill_now_selected",
    "search_box_text_became","dropdown_opened","surface","apply_kind","confirmation_text",
    "application_status_panel","validation_errors","email_prefilled","phone_prefilled",
    "first_name_prefilled","last_name_prefilled","country_code","entries_prefilled","entries",
    "most_recent","modal_kind","modal_opened","ats","ats_hint","sections_reviewed",
    "follow_company","follow_company_default","follow_company_before","follow_company_after",
    "follow_company_unchecked","screening_questions","disambiguation","contacts","panel_present",
    "names_extracted","job_poster","reachable_1st_degree","upsell","declined","feed_rendered",
    "signed_in_as_present","abort_signals_seen","actual","title","company","posted","applicants",
    "workplace","promoted","must_haves","nice_to_have","seniority","seniority_header",
    "seniority_requirements_block","seniority_conflict","linkedin_size_band","reasons",
}
RECALLED = {"reject_reason","company_size_claim","headcount","new_company"}

cnt = collections.Counter()
by_step = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)

# A claim can legitimately be sourced by a SIBLING record in the same episode
# - `_task_end.match_score` is backed by the `score_job` retrieval four
# records earlier. Counting those as unsourced would overstate the problem,
# so resolve retrieval at episode scope, not record scope.
ep_ret = collections.defaultdict(set)
ep_shot = collections.defaultdict(bool)
for r in rows:
    for e in (r.get("retrieval") or []):
        ep_ret[r.get("task_id")].add(e.get("source"))
    if r.get("screenshot"):
        ep_shot[r.get("task_id")] = True

for r in rows:
    res = r.get("result")
    if not isinstance(res, dict):
        continue
    tid = r.get("task_id")
    has_shot = bool(r.get("screenshot"))
    has_ret  = bool(r.get("retrieval")) or bool(ep_ret[tid])
    for k, v in res.items():
        if k in SELF:
            c = "SELF"
        elif k in CITED:
            c = "CITED" if has_ret else "NO_SHOT"
        elif k in RECALLED:
            c = "CITED" if has_ret else "RECALLED"
        elif k in PAGE:
            c = "IN_SHOT" if has_shot else ("EP_SHOT" if ep_shot[tid] else "NO_SHOT")
        else:
            c = "UNCLASSIFIED"
        cnt[c] += 1
        by_step[r["step"]][c] += 1
        if c in ("RECALLED", "NO_SHOT", "EP_SHOT") and len(examples[c]) < 3:
            examples[c].append((r["step"], k, str(v)[:64]))

tot = sum(cnt.values())
print(f"{path}\n{len(rows)} records, {tot} individual assertions\n")
order = ["SELF", "CITED", "IN_SHOT", "EP_SHOT", "NO_SHOT", "RECALLED", "UNCLASSIFIED"]
for c in order:
    if not cnt[c]: continue
    print(f"  {cnt[c]:>4}  ({100*cnt[c]/tot:4.1f}%)  {c}")
print()
print(f"{'step':<26} {'SELF':>5} {'CITED':>6} {'INSHOT':>7} {'EPSHOT':>7} {'NOSHOT':>7} {'RECALL':>7}")
print("-" * 70)
for step in sorted(by_step, key=lambda s: -sum(by_step[s].values())):
    b = by_step[step]
    print(f"{step:<26} {b['SELF']:>5} {b['CITED']:>6} {b['IN_SHOT']:>7} {b['EP_SHOT']:>7} {b['NO_SHOT']:>7} {b['RECALLED']:>7}")
print("-" * 70)
verifiable = cnt["SELF"] + cnt["CITED"]
print(f"\nAssertions whose source is IN the record:        {verifiable:>4}  ({100*verifiable/tot:.1f}%)")
print(f"Assertions needing the attached screenshot:     {cnt['IN_SHOT']:>4}  ({100*cnt['IN_SHOT']/tot:.1f}%)"
      f"  - checkable by eye, but nothing links a claim to the element it came from")
print(f"Assertions whose only evidence is a screenshot")
print(f"  attached to a DIFFERENT record in the episode: {cnt['EP_SHOT']:>4}  ({100*cnt['EP_SHOT']/tot:.1f}%)")
print(f"Assertions with no evidence anywhere:           {cnt['NO_SHOT']+cnt['RECALLED']:>4}  "
      f"({100*(cnt['NO_SHOT']+cnt['RECALLED'])/tot:.1f}%)")
for c in ("EP_SHOT", "NO_SHOT", "RECALLED"):
    if examples[c]:
        print(f"\n  {c} examples:")
        for s, k, v in examples[c]:
            print(f"    {s}.{k} = {v}")
