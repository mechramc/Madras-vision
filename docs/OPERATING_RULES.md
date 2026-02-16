# Madras Vision Operating Rules (Adopted Lessons)

## Purpose
Project-specific execution rules adopted from Blueflame `CLAUDE.md` and Agni `codex.md`/`CLAUDE.md`.

## Source-of-Truth Order
1. `README.md` (project architecture and phase intent)
2. `docs/PRD.md` (product requirements and success thresholds)
3. `docs/CANONICAL_TASK_LIST.md` (atomic execution order)
4. `docs/STATUS.md` (current phase state and blockers)
5. `CHECKPOINT.md` (session continuity handoff)

If any conflict appears, follow the highest-priority document above.

## Mandatory Workflow (Every Session)
1. Read `docs/STATUS.md` and `CHECKPOINT.md` before starting work.
2. Pick tasks only from `docs/CANONICAL_TASK_LIST.md`.
3. Implement the smallest viable change set for the active task.
4. Validate outputs (script-level checks/tests where applicable).
5. Update `CHECKPOINT.md` with completed work, decisions, blockers, next steps.
6. Update `docs/STATUS.md` when task/phase status changes.

## Scope Discipline
1. No coding before planning gates are approved (`MV-001`, `MV-002`).
2. No broad refactors unless explicitly requested.
3. Preserve existing user changes; never revert unrelated edits.
4. Multi-file work is executed sequentially and verified after each step.

## Definition of Done Gate (Per Task/Phase)
A task/phase is complete only when all are true:
1. Implemented: required code/docs exist.
2. Wired: IO contracts are connected end-to-end.
3. Reachable: user can run the documented command path.
4. Verified: output artifact exists and is inspectable.
5. Documented: status/checkpoint updated.

## Validation and Safety
1. Validate external/config/data inputs before expensive training steps.
2. Enforce schema checks for SFT and DPO datasets.
3. Track provenance/licensing metadata for all data sources.
4. Never commit secrets or credentials.

## Blocker Protocol
1. Report blockers immediately when discovered.
2. Record blocker, impact, and unblock condition in `docs/STATUS.md`.
3. Do not continue dependent tasks while blocker remains unresolved.

## Minimal-Change Rule
1. Prefer editing existing files over creating new abstractions.
2. Keep changes tightly scoped to one task ID.
3. Avoid unrelated formatting/reorganization churn.

## Madras Vision-Specific Adaptation
1. Evidence artifacts are mandatory for each phase:
1. Phase 1: tokenizer analysis report.
2. Phase 2: SFT checkpoints + training metrics.
3. Phase 3: DPO checkpoints + preference metrics.
4. Phase 4: merged/exported model + inference smoke output.
5. Phase 5: benchmark JSON + markdown report + audit sheet.
2. Never mark the project complete from “training succeeded” alone.
3. Completion requires metric comparison against PRD thresholds.
