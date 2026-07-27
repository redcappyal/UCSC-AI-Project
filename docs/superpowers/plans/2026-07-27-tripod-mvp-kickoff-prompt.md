# Kickoff prompt — tripod match-analysis MVP (autonomous session)

Copy-paste everything below the rule into a fresh Claude Code session opened at the
repo root. The session needs no other context. (Launch it only after the working tree
is back to a clean `main` — Step 0 makes the session check rather than assume.)

---

ultracode

(The word above is a harness keyword that opts this whole session into multi-agent
workflow orchestration. If your harness does not recognize it, ignore it and use
ordinary subagents where the rules below call for parallel work.)

You are executing a pre-approved plan to completion, autonomously, over as many
iterations as it takes. Do not re-litigate the design and do not stop to ask for
approval between phases — the only stopping conditions are in Step 0 and the
Definition of Done.

Read these two documents first; they are the contract for this session:

1. `docs/superpowers/specs/2026-07-27-tripod-match-analysis-design.md` — what we are
   building and why (the analysis ladder, capability gating, homography-gated auto
   court solve).
2. `docs/superpowers/plans/2026-07-27-tripod-match-analysis-mvp.md` — the 19 tasks you
   will execute, with acceptance gates per task.

Mission: check off **every checkbox in the plan**. The three human gates (spec §7:
person-detector weights, optional ball retrain, silver-label spot-check) are
deliberately NOT plan tasks — do not attempt them. When every checkbox is checked,
write `docs/HANDOFF-tripod-mvp.md` (Task 19 specifies its exact sections) and finish
with a summary.

**Step 0 — clean-start guard (before anything else):**
Run `git status`. If the tree is dirty or not on `main`, STOP and report what you
found — never stash, reset, or commit work you did not author (a concurrent
stereo-archive effort may be in flight). If clean:
`git checkout -b claude/tripod-mvp main`, then confirm
`.venv/bin/python -m pytest tests/ -q` passes with **zero collection errors and zero
failures** before starting Task 1. If collection errors mention stereo modules, the
archive landed half-way — stop and report rather than fixing someone else's branch.

Operating rules:

1. One commit per completed task on `claude/tripod-mvp`, conventional message. Never
   push.
2. Drive the plan with the superpowers:subagent-driven-development skill (fresh
   subagent per task; you review between tasks). The plan file's checkboxes are the
   progress ledger — check them off as you go and record divergences under
   `## Deviations`. Where any skill's ceremony conflicts with these rules (e.g. "stop
   and ask a human"), these rules win: apply rule 9 and keep going. Assume your
   context will be compacted — the plan file, not your memory, is the source of truth
   for where you are.
3. Work in the main checkout, not a worktree. If you do create a worktree anyway,
   immediately symlink the venv (`ln -sfn "<main-repo>/.venv" "<worktree>/.venv"`,
   see the /verify skill) — without it the pytest gate and the PostToolUse hook both
   break.
4. Follow each task's TDD cycle as written. The PostToolUse hook auto-runs the paired
   `tests/test_<module>.py` on every `*.py` edit and returns failures as a blocked
   edit — treat that as a failing test, never as an obstacle to route around.
5. `.venv/bin/python -m pytest tests/ -q` must be green before every commit. The venv
   is the only Python that works here (CLAUDE.md).
6. Eval discipline: for judge/calibration-touching changes run the /eval skill
   against the newest `eval_set/BASELINE-*.md`. Know its scope: it replays stored
   cases through the judge only — for detector-path equivalence (Task 3) the plan's
   hit-replay diff on stored CSVs is the real gate, not /eval. Never claim an
   improvement without the matching evidence.
7. UI tasks: DESIGN.md is law. Verify every UI change with the /verify skill in both
   themes at 390×844 before checking the task off.
8. Adversarial review at the risky points only: after Phase 4 (auto-solve), after
   Phase 5 (player tier), and once before the handoff, run the /code-review skill
   over the branch diff (fall back to superpowers:requesting-code-review if
   unavailable) and fix confirmed findings. Elsewhere, the per-task TDD + suite +
   eval gates suffice.
9. If the plan contradicts what you find in the code, the code wins: implement the
   grounded version, note it under `## Deviations`, and continue. If a whole task
   rests on a wrong assumption, redesign the smallest correct replacement that still
   meets the task's acceptance gate, and note that too. (Known live example: a
   stereo-archive branch may land before you start — adapt imports/paths to what you
   find.)
10. Do not touch: `ios/` sources, stereo/peer code (wherever it lives), or line-call
    output compatibility (existing result keys keep their shape; additions only).
    Never pass a possibly-non-ASCII path to `cv2.imread`/`cv2.imwrite` (CLAUDE.md
    explains why).

Definition of done: every plan checkbox checked; spec §8 demonstrated — including the
end-to-end pair `SquashAnalytics.mp4` (1080p60) + its 1080p30 proxy from
`tools/make_proxy.py`, one report with tiers 1-3 and one with tier 3 honestly gated
off; line-call eval unchanged; rally + auto-solve baselines committed; HANDOFF
written; suite green; final summary listing baselines and deviations.
