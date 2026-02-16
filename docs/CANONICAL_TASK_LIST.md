# Canonical Task List (Atomic)

## Purpose
Single source of truth for execution order and delivery scope. Each item is atomic, testable, and independently completable.

## Status Legend
`todo` | `in_progress` | `blocked` | `done`

## Task Table

| ID | Task | Depends On | Output | Acceptance Criteria | Status |
|---|---|---|---|---|---|
| MV-001 | Confirm PRD baseline is approved | None | Approved PRD | Explicit user approval recorded | todo |
| MV-002 | Confirm canonical task list is approved | MV-001 | Approved task list | Explicit user approval recorded | todo |
| MV-003 | Freeze repository naming conventions | MV-002 | Naming standard note | Filenames and IDs follow a single convention | todo |
| MV-004 | Define script CLI contract document | MV-002 | CLI contract spec | Every script has args, inputs, outputs, exit codes | todo |
| MV-005 | Define dataset schema for SFT records | MV-004 | Schema spec | Required fields and types documented | todo |
| MV-006 | Define dataset schema for DPO records | MV-004 | Schema spec | Required fields and types documented | todo |
| MV-007 | Define metadata taxonomy (era/source/citation) | MV-005 | Taxonomy spec | Controlled values documented | todo |
| MV-008 | Define train/val/test split policy | MV-005 | Split policy doc | Leakage prevention rules documented | todo |
| MV-009 | Define evaluation rubric for historical grounding | MV-002 | Rubric doc | Scoring criteria and thresholds documented | todo |
| MV-010 | Define evaluation rubric for literary grounding | MV-009 | Rubric doc | Citation relevance/accuracy rubric documented | todo |
| MV-011 | Define hallucination audit procedure | MV-009 | Audit SOP | Sampling and pass/fail rules documented | todo |
| MV-012 | Define reproducibility checklist | MV-002 | Checklist doc | Seeds, versions, configs, commands listed | todo |
| MV-013 | Create sample SFT record fixtures | MV-005 | Fixture files | Fixtures validate against schema | todo |
| MV-014 | Create sample DPO record fixtures | MV-006 | Fixture files | Fixtures validate against schema | todo |
| MV-015 | Add data manifest template | MV-007 | Manifest template | Includes source, license, checksum fields | todo |
| MV-016 | Add source licensing register template | MV-002 | Licensing register | Each source has usage status field | todo |
| MV-017 | Implement schema validator for SFT | MV-005 | Validator module | Invalid records produce actionable errors | todo |
| MV-018 | Implement schema validator for DPO | MV-006 | Validator module | Invalid records produce actionable errors | todo |
| MV-019 | Implement image path existence validator | MV-005 | Validator module | Missing image paths are detected | todo |
| MV-020 | Implement metadata validator | MV-007 | Validator module | Invalid era/source/citation flags detected | todo |
| MV-021 | Implement dataset integrity report generator | MV-017 | Integrity JSON report | Record counts and error counts output | todo |
| MV-022 | Implement data split utility | MV-008 | Split artifacts | No duplicate IDs across splits | todo |
| MV-023 | Implement deterministic seed handling utility | MV-012 | Utility module | Seed controls random operations reproducibly | todo |
| MV-024 | Implement tokenizer corpus loader | MV-005 | Loader module | Tamil corpus files parsed successfully | todo |
| MV-025 | Implement tokenizer metrics calculator | MV-024 | Metrics module | tokens/word and aggregate stats computed | todo |
| MV-026 | Implement tokenizer evaluation CLI | MV-025 | `tokenizer_eval.py` | CLI writes report JSON to expected path | todo |
| MV-027 | Add tokenizer go/no-go threshold check | MV-026 | Decision output | Emits pass/fail against 2.5 and 3.0 thresholds | todo |
| MV-028 | Implement config loader for YAML configs | MV-004 | Config module | Invalid config errors are explicit | todo |
| MV-029 | Implement model loading wrapper | MV-028 | Loader module | Base model loads from configured path | todo |
| MV-030 | Implement LoRA adapter configuration wrapper | MV-028 | Adapter module | Target modules and ranks are applied | todo |
| MV-031 | Implement SFT dataset adapter | MV-005 | Adapter module | Converts records to trainer-ready format | todo |
| MV-032 | Implement SFT training loop entrypoint | MV-029 | `train_sft.py` | One epoch dry-run executes without crash | todo |
| MV-033 | Add SFT checkpoint save/resume | MV-032 | Checkpoint behavior | Training resumes from checkpoint | todo |
| MV-034 | Add SFT metric logging | MV-032 | Logs/telemetry | Loss and step metrics persisted | todo |
| MV-035 | Add SFT OOM recovery guidance output | MV-032 | Error handler | OOM message provides concrete mitigation hints | todo |
| MV-036 | Implement DPO dataset adapter | MV-006 | Adapter module | Chosen/rejected pairs formatted correctly | todo |
| MV-037 | Implement DPO training loop entrypoint | MV-036 | `train_dpo.py` | One epoch dry-run executes without crash | todo |
| MV-038 | Add DPO checkpoint save/resume | MV-037 | Checkpoint behavior | DPO resumes from checkpoint | todo |
| MV-039 | Add DPO metric logging | MV-037 | Logs/telemetry | Preference loss metrics persisted | todo |
| MV-040 | Implement adapter merge script | MV-033 | `merge_adapters.py` | Merged artifact is loadable | todo |
| MV-041 | Implement export command wrapper | MV-040 | `export_gguf.py` | Export process runs with quantization option | todo |
| MV-042 | Implement single-image inference smoke test | MV-040 | `test_inference.py` | Prompt + image inference returns response | todo |
| MV-043 | Implement benchmark dataset loader | MV-009 | Eval module | Benchmark items load with schema checks | todo |
| MV-044 | Implement baseline vs tuned comparator | MV-043 | Eval module | Produces side-by-side metric output | todo |
| MV-045 | Implement evaluation CLI | MV-044 | `evaluate.py` | Writes evaluation JSON to reports path | todo |
| MV-046 | Implement report markdown generator | MV-045 | `generate_report.py` | Generates readable summary markdown | todo |
| MV-047 | Add latency measurement utility | MV-042 | Utility module | ms/token measured and reported | todo |
| MV-048 | Add hallucination audit worksheet export | MV-011 | Audit artifact | Reviewer-ready worksheet generated | todo |
| MV-049 | Add training queue script hardening | MV-032 | `train_queue.sh` | Script exits on failure and logs clearly | todo |
| MV-050 | Add Docker image build verification steps | MV-012 | Verification checklist | Build succeeds with pinned deps | todo |
| MV-051 | Add environment preflight checks | MV-012 | Preflight script/spec | CUDA/GPU/tooling checks are explicit | todo |
| MV-052 | Add runbook for end-to-end execution | MV-045 | Runbook doc | One command sequence from data to report | todo |
| MV-053 | Add artifact retention policy | MV-002 | Policy note | Checkpoints/outputs retention documented | todo |
| MV-054 | Add model card template for final artifact | MV-046 | Model card template | Includes intended use and limitations | todo |
| MV-055 | Add risk register with owners | MV-001 | Risk register | Top risks assigned to owners | todo |
| MV-056 | Execute tokenizer phase on real corpus | MV-027 | Tokenizer report | Decision gate outcome recorded | todo |
| MV-057 | Execute SFT training run v1 | MV-035 | SFT checkpoints | Run completes with logged metrics | todo |
| MV-058 | Execute DPO training run v1 | MV-039 | DPO checkpoints | Run completes with logged metrics | todo |
| MV-059 | Merge and export deployment artifact | MV-041 | Merged/exported model | Artifact created and load test passes | todo |
| MV-060 | Execute full benchmark and generate report | MV-046 | Final eval report | Threshold comparison included | todo |
| MV-061 | Conduct hallucination manual audit | MV-048 | Audit results | Pass rate computed with rubric | todo |
| MV-062 | Validate success metrics against PRD thresholds | MV-060 | Threshold summary | Pass/fail for each KPI documented | todo |
| MV-063 | Finalize v1 release package | MV-062 | Release bundle | Configs, scripts, report, model card bundled | todo |
| MV-064 | Sign-off checkpoint with user | MV-063 | Approval record | User confirms v1 completion | todo |

## Execution Rules
1. No coding starts before MV-001 and MV-002 are marked `done`.
2. If a task changes scope, update this document before implementation.
3. Any blocked task must include reason and unblock condition.
4. New tasks must get new IDs; do not renumber existing IDs.
