# Madras Vision: Multimodal Historical & Literary Reasoning

**Author:** Ramchand Kumaresan
**Stack:** Llama-3.2-Vision-11B · Unsloth · RTX 5090 · Tamil Sangam Corpus

---

## Executive Summary

**Madras Vision** is a local-first AI project aimed at bridging what we call the *Temporal Semantic Gap* in Tamil cultural heritage. The goal: fine-tune a Multimodal Vision-Language Model (VLM) to perform **Historical Grounding** — identifying Chennai's architectural landmarks across three centuries and contextualizing them through classical Tamil literature, from the Sangam era to the modern period.

The project demonstrates end-to-end capability across high-fidelity multimodal dataset curation, low-rank adaptation (LoRA/QLoRA) on frontier vision models, and Direct Preference Optimization (DPO) for cultural nuance and tone alignment.

This README is designed as a **repeatable playbook** — the same infrastructure, tooling, and workflow patterns apply to any future fine-tuning project on this workstation.

---

## Problem Statement

General-purpose VLMs (base Llama-3.2, Gemini, etc.) suffer from **Cultural Hallucination** — they perform well on globally dominant landmarks but fail catastrophically on regional historical reasoning. Specifically, they cannot:

1. **Temporal Disambiguation** — Distinguish a 19th-century colonial building in Egmore from a modern replica or renovation.
2. **Literary Grounding** — Relate a visual landmark to a 2,000-year-old Tamil poem (e.g., *Pattinappalai*'s descriptions of ancient port life in Puhar/Kaveripoompattinam).
3. **Local Real-Time Reasoning** — Perform this inference locally with zero-latency for immersive applications (VR/AR).

Madras Vision closes this gap by creating a domain-specialized VLM that reasons across visual, historical, and literary dimensions simultaneously.

---

## Hardware Platform

| Component | Specification | Role |
| :--- | :--- | :--- |
| **GPU** | NVIDIA RTX 5090 (32 GB GDDR7, 1.79 TB/s bandwidth) | QLoRA fine-tuning, DPO alignment, vLLM inference |
| **CPU** | AMD Ryzen 9 9950X3D | Dataset preprocessing, synthetic data generation, tokenizer evaluation |
| **RAM** | 64 GB DDR5 (recommended minimum) | Large dataset handling, parallel preprocessing pipelines |
| **Storage** | 2 TB NVMe (OS + tools) + 4 TB NVMe (models + datasets) | Separate drives prevent I/O contention during training |

### Why This Hardware Works

The RTX 5090's 32 GB VRAM comfortably handles the Llama-3.2-11B-Vision model under 4-bit QLoRA quantization via Unsloth. At 4-bit precision, the base model weights occupy ~6–7 GB, leaving ~24 GB for the vision encoder, optimizer states, gradient buffers, activation cache, and batch processing. This provides ample headroom for batch size tuning and high-resolution image inputs without OOM pressure.

The 1.79 TB/s memory bandwidth delivers strong token generation throughput, well under the <100ms/token target for real-time VR inference. CUDA-native tooling (Unsloth, vLLM, TRL, bitsandbytes) runs at peak efficiency on this GPU with no framework translation needed.

### VRAM Budget (Estimated)

| Component | Estimated VRAM | Notes |
| :--- | :--- | :--- |
| Base model weights (4-bit NF4) | ~6 GB | Llama-3.2-11B-Vision at 4-bit quantization |
| Vision encoder | ~2 GB | Image preprocessing and patch embedding |
| LoRA adapter weights | ~1 GB | Rank 64, all attention + MLP layers |
| Optimizer states (AdamW 8-bit) | ~2 GB | 8-bit optimizer via bitsandbytes |
| Gradient buffers + activations | ~6–8 GB | With gradient checkpointing enabled |
| Batch data (images + tokens) | ~4–6 GB | Batch size 4, 2048 token sequences |
| **Total during SFT** | **~21–25 GB** | **Leaves 7–11 GB headroom on 32 GB** |

---

## Reproducible Environment Setup

This section ensures any training project on this workstation starts from a clean, isolated, and version-pinned foundation.

### Prerequisites

```bash
# Verify NVIDIA driver and CUDA
nvidia-smi                    # Should show RTX 5090, driver 570+
nvcc --version                # CUDA 12.6+

# Docker with NVIDIA runtime (for project isolation)
docker --version              # 27.x+
nvidia-container-toolkit      # Enables --gpus flag

# Verify GPU is accessible from Docker
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Option A: Docker-Based Isolation (Recommended for Parallel Projects)

Each training project runs in its own container with pinned dependencies, shared GPU access, and isolated filesystems. This is the recommended approach when running multiple fine-tuning projects in parallel or sequentially on the same workstation.

```bash
# Build the base training image (do this once)
docker build -t madras-train:base -f docker/Dockerfile.train .

# Launch a project-specific container
docker run -d \
  --name madras-vision \
  --gpus '"device=0"' \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/configs:/workspace/configs \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  madras-train:base \
  sleep infinity

# Attach to the container
docker exec -it madras-vision bash
```

**`docker/Dockerfile.train`:**

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin Python and pip
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    python -m pip install --upgrade pip==24.3.1

# Core ML stack (pinned versions — update at project start)
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    transformers==4.47.1 \
    accelerate==1.2.1 \
    datasets==3.2.0 \
    trl==0.13.0 \
    peft==0.14.0 \
    bitsandbytes==0.45.0 \
    vllm==0.6.6 \
    sentencepiece==0.2.0 \
    protobuf==5.29.3 \
    Pillow==11.1.0 \
    wandb==0.19.1

# Unsloth (install last, needs torch present)
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Workspace
WORKDIR /workspace
```

> **Note:** Pin versions to whatever is current at the time you start your project. The versions above are examples — always verify compatibility with your base model and CUDA version.

### Option B: Conda Environment (Single-Project or Quick Start)

```bash
# Create isolated environment
conda create -n madras-vision python=3.12 -y
conda activate madras-vision

# Install from lockfile
pip install -r requirements.txt

# Or install manually
pip install torch torchvision transformers accelerate datasets \
    trl peft bitsandbytes vllm sentencepiece wandb
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Shared Resources Across Projects

When running multiple fine-tuning projects, avoid redundant downloads and disk usage:

```
$HOME/
├── .cache/huggingface/        # Shared model cache (mount into every container)
├── .cache/wandb/              # Shared experiment tracking
└── training-projects/
    ├── madras-vision/         # This project
    ├── project-b/             # Another fine-tuning project
    └── project-c/             # Yet another
```

Mount the HuggingFace cache as a shared volume so base model weights are downloaded once and reused:

```bash
# In every docker run command, include:
-v $HOME/.cache/huggingface:/root/.cache/huggingface
```

---

## Technical Architecture

### Software Stack

| Layer | Technology | Notes |
| :--- | :--- | :--- |
| **Base Model** | `Llama-3.2-11B-Vision-Instruct` (Meta) | Primary candidate; `PaliGemma-22B` (Google) as fallback |
| **Fine-Tuning** | Unsloth | 2× faster training, ~70% VRAM reduction via 4-bit quantization |
| **Alignment** | TRL (Transformer Reinforcement Learning) | DPO for cultural tone and reasoning quality |
| **Inference** | vLLM / GGUF export | Production serving; GGUF for Unity/VR integration |
| **Experiment Tracking** | Weights & Biases (wandb) | Loss curves, VRAM usage, hyperparameter comparison |
| **Containerization** | Docker + NVIDIA Container Toolkit | Per-project isolation, reproducible environments |

### Model Download and Verification

```bash
# Download base model (requires HuggingFace access token for gated models)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir models/base/llama-3.2-11b-vision

# Verify download integrity
python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained('models/base/llama-3.2-11b-vision')
print(f'Parameters: {model.num_parameters():,}')
print('Model loaded successfully.')
"
```

---

## The Data Engine

The competitive moat of this project is the **Synthetic Image-Text-History Dataset** — a curated pipeline that pairs visual archives with literary and historical reasoning traces.

| Dataset Component | Source / Method |
| :--- | :--- |
| **Visual Archives** | Historical photographs (1880–1950) sourced from the British Library digital collections and Madras Musings archives |
| **Literary Mapping** | Gemini 1.5 Pro used to systematically map landmarks to verses in *Silappatikaram*, *Tevaram*, *Bharathiyar* poems, and other Sangam-era texts |
| **Reasoning Traces** | Synthetically generated "Thought Chains" describing architectural evolution (e.g., *"The absence of the LIC building confirms this photograph is pre-1959 Madras"*) |

### Data Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Visual Archives │───▶│  Landmark Tagger  │───▶│  Literary Mapper     │
│  (British Lib,   │    │  (CLIP + Manual   │    │  (Gemini 1.5 Pro    │
│   Madras Musings)│    │   Annotation)     │    │   Verse Retrieval)  │
└─────────────────┘    └──────────────────┘    └─────────┬───────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────────┐
                                               │  Reasoning Trace    │
                                               │  Generator          │
                                               │  (Synthetic CoT)    │
                                               └─────────┬───────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────────┐
                                               │  SFT + DPO Dataset  │
                                               │  (Image, Text,      │
                                               │   History, Verse)   │
                                               └─────────────────────┘
```

### Dataset Format

All datasets follow the HuggingFace `datasets` format for consistency across projects:

```json
{
  "image": "data/visual_archives/central_station_1920.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "<image>\nIdentify this building and provide its historical and literary context."
    },
    {
      "role": "assistant",
      "content": "This is Chennai Central Station, photographed circa 1920 during the British colonial period. The Neo-Gothic architecture designed by George Harding was completed in 1873..."
    }
  ],
  "metadata": {
    "era": "colonial",
    "date_range": "1910-1925",
    "literary_refs": ["pattinappalai:42", "bharathiyar:swadeshi_poems"],
    "source": "british_library"
  }
}
```

### DPO Preference Pairs Format

```json
{
  "image": "data/visual_archives/mylapore_temple_1890.jpg",
  "prompt": "Describe this temple and its significance.",
  "chosen": "The Kapaleeshwarar Temple in Mylapore, captured here in the late 19th century, shows the original Dravidian gopuram before the 1906 renovation. The Tevaram hymns of Thirunavukkarasar (7th century CE) describe this site as 'Mayilai', where Parvati worshipped Shiva in the form of a peacock...",
  "rejected": "This is a Hindu temple in Chennai. It appears to be an old photograph of a South Indian temple with a tall tower entrance. The temple is dedicated to Lord Shiva and is a popular tourist destination."
}
```

---

## Implementation Roadmap

### Phase 1 — Domain-Specific Tokenization *(Weeks 1–2)*

Standard tokenizers frequently struggle with complex Tamil ligatures and Grantha-derived characters, leading to excessive token counts ("token explosion") that degrade both training efficiency and inference quality.

```bash
# Run tokenizer evaluation
python scripts/tokenizer_eval.py \
  --model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --corpus data/sangam_texts/ \
  --output reports/tokenizer_analysis.json

# Expected output: tokens-per-word ratio, OOV rate, ligature handling stats
# Target: <2.5 tokens per Tamil word (comparable to English ~1.3)
```

**Decision gate:** If the token-per-word ratio exceeds 3.0 on Sangam texts, augment the vocabulary using `sentencepiece` before proceeding to SFT. Otherwise, proceed with the base tokenizer.

### Phase 2 — Supervised Fine-Tuning (SFT) *(Weeks 3–5)*

**Objective:** Visual Landmark Identification with Literary Context

```bash
# Launch SFT training
python scripts/train_sft.py \
  --config configs/qlora_sft.yaml \
  --wandb-project madras-vision \
  --wandb-run sft-v1

# Monitor in real-time
watch -n 5 nvidia-smi   # VRAM usage
wandb sync              # Sync to dashboard
```

**`configs/qlora_sft.yaml`:**

```yaml
# === Base Model ===
model_name: "meta-llama/Llama-3.2-11B-Vision-Instruct"
load_in_4bit: true
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_quant_type: "nf4"

# === LoRA Configuration ===
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"

# === Training ===
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
max_seq_length: 2048
bf16: true
gradient_checkpointing: true
optim: "adamw_8bit"

# === Data ===
dataset_path: "data/sft_dataset"
image_column: "image"

# === Output ===
output_dir: "outputs/sft-v1"
save_strategy: "steps"
save_steps: 100
logging_steps: 10
```

### Phase 3 — Alignment with DPO *(Weeks 6–8)*

Use Direct Preference Optimization to refine the model's reasoning depth and cultural tone.

```bash
# Launch DPO alignment (starts from SFT checkpoint)
python scripts/train_dpo.py \
  --config configs/dpo_alignment.yaml \
  --sft-checkpoint outputs/sft-v1/checkpoint-best \
  --wandb-project madras-vision \
  --wandb-run dpo-v1
```

**`configs/dpo_alignment.yaml`:**

```yaml
# === Base (from SFT checkpoint) ===
model_name: "outputs/sft-v1/checkpoint-best"
load_in_4bit: true

# === DPO Parameters ===
beta: 0.1
loss_type: "sigmoid"

# === Training ===
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
bf16: true

# === Data ===
dataset_path: "data/dpo_pairs"

# === Output ===
output_dir: "outputs/dpo-v1"
```

| | Preferred Response (A) | Rejected Response (B) |
| :--- | :--- | :--- |
| **Depth** | Historically grounded, multi-era analysis | Surface-level, modern-biased description |
| **Literary** | Accurate Sangam verse citation with context | Generic or fabricated literary references |
| **Tone** | Poetic, scholarly, culturally resonant | Flat, encyclopedic, culturally detached |

### Phase 4 — Export and Deployment

```bash
# Merge LoRA adapters into base model
python scripts/merge_adapters.py \
  --base-model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --adapter-path outputs/dpo-v1/checkpoint-best \
  --output-dir outputs/madras-vision-merged

# Export to GGUF for Unity/VR integration
python scripts/export_gguf.py \
  --model outputs/madras-vision-merged \
  --output outputs/madras-vision.gguf \
  --quantization q4_k_m

# Verify inference
python scripts/test_inference.py \
  --model outputs/madras-vision-merged \
  --image data/test/central_station_test.jpg \
  --prompt "Identify this building and provide its historical and literary context."
```

---

## Success Metrics

| Metric | Target | Method |
| :--- | :--- | :--- |
| **VLM-Bench Accuracy** | ≥30% improvement over base Llama-3.2 | Custom "Madras History" visual benchmark (100+ image-question pairs) |
| **Inference Latency** | <100 ms/token | RTX 5090 local inference via vLLM |
| **Hallucination Rate** | Qualitative pass rate >90% | Manual audit of literary citations against actual Sangam source texts |
| **Tokenizer Efficiency** | <2.5 tokens/word on Tamil text | Automated benchmark against Sangam corpus |

### Running the Benchmark

```bash
# Run full evaluation suite
python scripts/evaluate.py \
  --model outputs/madras-vision-merged \
  --benchmark evaluation/madras_bench/ \
  --output reports/eval_results.json \
  --compare-baseline meta-llama/Llama-3.2-11B-Vision-Instruct

# Generate comparison report
python scripts/generate_report.py \
  --results reports/eval_results.json \
  --output reports/madras_vision_eval_report.md
```

---

## Multi-Project Workflow

This workstation is designed to support multiple training projects. Here's how to manage them without conflicts on a single RTX 5090.

### GPU Time-Sharing Strategy

The RTX 5090's 32 GB VRAM supports one of the following concurrent configurations:

| Scenario | VRAM Allocation | Safe to Run Together? |
| :--- | :--- | :--- |
| 1 QLoRA training job | ~21–25 GB | Yes (alone) |
| 1 QLoRA training + 1 quantized inference | ~25 GB + ~7 GB | Yes (tight but works) |
| 2 QLoRA training jobs | ~42–50 GB | **No — will OOM** |
| 2–3 quantized inference jobs | ~7 GB each | Yes |

**Rule: never run two training jobs on the same GPU simultaneously.** Use sequential scheduling or a simple job queue.

### Sequential Job Scheduling

For hands-off overnight training across multiple projects:

```bash
#!/bin/bash
# scripts/train_queue.sh — Run training jobs sequentially

echo "[$(date)] Starting Madras Vision SFT..."
docker exec madras-vision python scripts/train_sft.py --config configs/qlora_sft.yaml
echo "[$(date)] Madras Vision SFT complete."

echo "[$(date)] Starting Project B SFT..."
docker exec project-b python scripts/train_sft.py --config configs/qlora_sft.yaml
echo "[$(date)] Project B SFT complete."

echo "[$(date)] All training jobs finished."
```

```bash
# Run the queue in background
nohup bash scripts/train_queue.sh > logs/train_queue.log 2>&1 &
```

### Container-Per-Project Isolation

```bash
# Project A: Madras Vision
docker run -d --name madras-vision --gpus '"device=0"' --shm-size=16g \
  -v ~/training-projects/madras-vision:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  madras-train:base sleep infinity

# Project B: Another fine-tuning project
docker run -d --name project-b --gpus '"device=0"' --shm-size=16g \
  -v ~/training-projects/project-b:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  madras-train:base sleep infinity
```

### Monitoring

```bash
# Real-time VRAM usage across all containers
watch -n 2 'nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv'

# GPU utilization and temperature
watch -n 5 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv'
```

### Experiment Tracking Across Projects

All projects report to the same Weights & Biases instance, organized by project name:

```bash
wandb login
# madras-vision → wandb project: "madras-vision"
# project-b     → wandb project: "project-b"
```

### Template for New Projects

To start a new fine-tuning project using this same infrastructure:

```bash
# 1. Clone the template structure
cp -r templates/training-project/ ~/training-projects/new-project/
cd ~/training-projects/new-project

# 2. Edit configs for your model and dataset
vim configs/qlora_sft.yaml

# 3. Launch in isolated container
docker run -d \
  --name new-project \
  --gpus '"device=0"' \
  --shm-size=16g \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  madras-train:base \
  sleep infinity

# 4. Start training
docker exec -it new-project bash
python scripts/train_sft.py --config configs/qlora_sft.yaml
```

---

## Future Integration — Project Madras VR

The trained model weights will be exported to **GGUF format** for deployment inside a Unity-based VR environment. The end-state experience: when a user "looks" at a building in the virtual reconstruction of historical Madras, the local model provides **live, literary commentary in Tamil and English** — merging architectural history with Sangam poetry in real time, entirely on-device.

### VR Inference Pipeline

```
Unity VR App
    │
    ├── Camera frame capture → JPEG/PNG
    │
    ├── HTTP POST to local inference server
    │   (llama.cpp server or vLLM, localhost:8080)
    │
    ├── Model: madras-vision.gguf (q4_k_m)
    │   Input:  [image + "Describe this building"]
    │   Output: [Historical context + Tamil verse + English translation]
    │
    └── Render text overlay in VR HUD
```

```bash
# Launch inference server for VR integration
./llama-server \
  --model outputs/madras-vision.gguf \
  --port 8080 \
  --n-gpu-layers 99 \
  --ctx-size 4096
```

---

## Repository Structure

```
madras-vision/
├── configs/                      # Training and inference configurations
│   ├── qlora_sft.yaml            #   Phase 2: SFT hyperparameters
│   └── dpo_alignment.yaml        #   Phase 3: DPO hyperparameters
├── data/
│   ├── visual_archives/          # Historical photographs
│   ├── sangam_texts/             # Tamil literary corpus
│   ├── sft_dataset/              # Curated SFT training data
│   ├── dpo_pairs/                # DPO preference pairs
│   └── test/                     # Held-out test images
├── docker/
│   └── Dockerfile.train          # Reproducible training environment
├── evaluation/
│   └── madras_bench/             # Custom benchmark suite
├── logs/                         # Training queue logs
├── models/
│   └── base/                     # Downloaded base model weights
├── outputs/                      # Training checkpoints and merged models
│   ├── sft-v1/
│   ├── dpo-v1/
│   ├── madras-vision-merged/
│   └── madras-vision.gguf
├── reports/                      # Evaluation results and analysis
├── scripts/
│   ├── tokenizer_eval.py         # Phase 1: Tokenizer analysis
│   ├── data_pipeline.py          # Dataset curation pipeline
│   ├── train_sft.py              # Phase 2: SFT training
│   ├── train_dpo.py              # Phase 3: DPO alignment
│   ├── merge_adapters.py         # Merge LoRA weights into base
│   ├── export_gguf.py            # GGUF export for VR integration
│   ├── test_inference.py         # Quick inference validation
│   ├── evaluate.py               # Full benchmark suite
│   ├── generate_report.py        # Evaluation report generator
│   └── train_queue.sh            # Sequential multi-project training
├── templates/
│   └── training-project/         # Template for new projects
│       ├── configs/
│       ├── data/
│       ├── scripts/
│       └── README.md
├── requirements.txt              # Pinned Python dependencies
├── README.md
└── LICENSE
```

---

## Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `CUDA out of memory` during SFT | Batch size too large or gradient checkpointing disabled | Reduce `per_device_train_batch_size` to 2, ensure `gradient_checkpointing: true` |
| `CUDA out of memory` at startup | Another process holding VRAM | Run `nvidia-smi` to check; kill orphan processes or stop other containers |
| Slow tokenizer on Tamil text | Token explosion from unrecognized ligatures | Run Phase 1 tokenizer eval; augment vocabulary if tokens/word > 3.0 |
| `docker: could not select device driver` | NVIDIA Container Toolkit missing | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |
| wandb sync fails | Offline container or auth issue | Run `wandb login` inside container, ensure network access |
| GGUF export produces oversized file | Quantization not applied | Verify `--quantization q4_k_m` flag in export command |
| DPO loss plateaus or diverges | Beta too high or weak preference signal | Lower `beta` to 0.05; audit that chosen/rejected pairs are clearly separated |
| vLLM fails to load model | Insufficient contiguous VRAM | Stop other GPU processes; restart with `CUDA_VISIBLE_DEVICES=0` |
| Training slower than expected | CPU bottleneck on data loading | Increase `dataloader_num_workers` to 4–8; ensure data is on NVMe |

---

## License

TBD — Research use. Historical image assets subject to their respective archive licenses.

---

## Acknowledgments

Built on the shoulders of the Tamil Sangam literary tradition, the archival work of the British Library and Madras Musings, and the open-source AI community (Meta, Unsloth, HuggingFace).

*"யாதும் ஊரே யாவரும் கேளிர்"* — Every place is our homeland, every person our kin. (*Purananuru* 192, Kaniyan Pungundranar)
