# Product Requirements Document (PRD)

## Project
Madras Vision: Multimodal Historical and Literary Reasoning for Chennai Landmarks

## Version
v1.0 (Planning Baseline)

## Document Purpose
Define the product scope, requirements, constraints, quality bars, and acceptance criteria before implementation.

## Background and Problem
General-purpose multimodal models underperform on region-specific historical reasoning and literary grounding. They often:
1. Misidentify visually similar landmarks across time periods.
2. Produce shallow or fabricated cultural/literary context.
3. Fail to provide reliable local inference behavior for immersive applications.

Madras Vision addresses this by fine-tuning a vision-language model to reason jointly over images, historical context, and Tamil literary references.

## Product Vision
Given an image of a Chennai landmark, the model should return historically accurate, culturally grounded, and literarily meaningful explanations in a consistent scholarly tone, suitable for research and VR integration.

## Objectives
1. Build a reproducible end-to-end training and evaluation pipeline for domain-specific multimodal reasoning.
2. Improve domain benchmark accuracy versus base model by at least 30%.
3. Keep Tamil tokenization efficiency below 2.5 tokens per word on target corpus.
4. Achieve inference latency under 100 ms/token on target hardware for deployment scenarios.
5. Reduce historical/literary hallucinations to below 10% on audited samples.

## Non-Goals (v1)
1. Real-time multilingual speech synthesis and ASR integration.
2. Full autonomous archival ingestion from unrestricted web scraping.
3. Multi-GPU distributed training.
4. Public cloud deployment hardening.
5. Broad pan-India landmark generalization outside Chennai-focused scope.

## Primary Users
1. ML Engineer: Curates datasets, runs training/alignment/evaluation.
2. Researcher/Historian: Validates historical and literary correctness.
3. VR Integrator: Consumes exported model and invokes local inference APIs.

## Core Use Cases
1. Input an archival image and receive landmark identification with era-aware historical explanation.
2. Retrieve relevant Tamil literary references with grounded context.
3. Compare tuned model against base model on a fixed benchmark suite.
4. Export trained artifact for local serving in VR pipeline.

## Functional Requirements
1. Dataset pipeline supports image + conversational samples + metadata.
2. Dataset pipeline supports DPO preference pairs (prompt/chosen/rejected).
3. Tokenizer evaluation reports tokens-per-word, OOV proxies, and corpus statistics.
4. SFT trainer supports QLoRA/LoRA configs from YAML.
5. DPO trainer supports preference optimization from SFT checkpoint.
6. Training scripts support checkpoint save/resume.
7. Experiment logging supports run name, step metrics, and system telemetry.
8. Merge script fuses adapter into base model.
9. Export script outputs deployment artifact format (GGUF path supported in roadmap).
10. Inference test script runs single image prompt sanity check.
11. Evaluation script runs benchmark and baseline comparison.
12. Report generator produces human-readable markdown summary from evaluation JSON.

## Non-Functional Requirements
1. Reproducibility: Config-driven runs with version-pinned dependencies.
2. Reliability: Deterministic seeds and graceful failure messages.
3. Observability: Structured logs and metric tracking for each stage.
4. Maintainability: Modular scripts with clear IO contracts.
5. Data traceability: Every record includes provenance metadata.
6. Security/compliance: Respect source licenses and access controls.

## Data Requirements
1. Visual archive records with source attribution and licensing metadata.
2. Literary corpus with canonical citation identifiers.
3. SFT samples with instruction-response structure and metadata fields.
4. DPO samples with high-quality chosen/rejected separation.
5. Train/validation/test split rules to avoid leakage.
6. Validation checks for schema, corrupt images, missing files, malformed IDs.

## Model Requirements
1. Base model target: Llama-3.2-11B-Vision-Instruct (configurable).
2. Quantization target: 4-bit training path where supported.
3. LoRA target modules configurable.
4. Mixed precision support (bf16 where available).
5. Gradient checkpointing support.

## Evaluation Requirements
1. Madras benchmark with at least 100 image-question pairs.
2. Metrics:
1. Accuracy and groundedness versus base model.
2. Hallucination rate from manual audit rubric.
3. Latency under representative load.
4. Tokenizer efficiency metrics.
3. Evaluation output in JSON and markdown report.

## Success Metrics and Thresholds
1. Benchmark improvement: >= 30% over baseline.
2. Inference latency: < 100 ms/token (target hardware).
3. Hallucination pass rate: > 90%.
4. Tamil tokenization efficiency: < 2.5 tokens/word.

## Constraints and Assumptions
1. Hardware: single RTX 5090 class GPU with sufficient VRAM.
2. Training jobs serialized when VRAM budget requires.
3. Access to gated base model and source datasets is available.
4. Legal review for archival content usage is handled externally.

## Risks and Mitigations
1. Risk: Dataset noise causes weak alignment.
Mitigation: Strong schema validation and human spot-audits.
2. Risk: Literary hallucinations persist post-DPO.
Mitigation: Improve preference pair quality and rejection hardness.
3. Risk: OOM failures during training.
Mitigation: Reduce batch size, tune accumulation, enable checkpointing.
4. Risk: Evaluation blind spots.
Mitigation: Expand benchmark coverage across era/style/source.

## Milestones (Planning)
1. M1: Requirements and canonical tasks approved.
2. M2: Data contracts and validators implemented.
3. M3: Tokenizer analysis complete with go/no-go decision.
4. M4: SFT run complete with tracked metrics.
5. M5: DPO run complete and compared to SFT/base.
6. M6: Export + inference validation complete.
7. M7: Benchmark + final report complete.

## Definition of Done (v1)
1. All functional requirements implemented and documented.
2. End-to-end run succeeds from prepared data to evaluation report.
3. Metrics collected and compared against thresholds.
4. Artifact export and local inference smoke test pass.
5. Reproducible instructions validated on clean environment.

## Open Questions
1. Final approved licensing list for each archive source.
2. Required bilingual output coverage (Tamil-only vs Tamil+English mix).
3. Exact benchmark rubric weighting for historical vs literary scoring.
4. Mandatory safety filters for deployment context.
