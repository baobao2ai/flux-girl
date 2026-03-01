# flux-girl

FLUX.1-dev fine-tuning pipeline for portrait image generation.

## Pipeline Overview

```
Phase 1: Data Collection
  → Scrape images/videos from online sources
  → Extract frames, filter, deduplicate
  → Auto-caption with Florence-2 / WD-tagger

Phase 2: Fine-Tuning
  → Base model: FLUX.1-dev
  → Method: LoRA via ai-toolkit (Ostris)
  → Hardware: RTX 4090 (24GB)

Phase 3: Deployment
  → Inference: ComfyUI + custom API
  → Test: prompt suite + quality metrics
```

## Project Structure

```
flux-girl/
├── data/
│   ├── raw/          # Downloaded images/video frames
│   ├── processed/    # Cleaned, resized, deduplicated
│   └── captions/     # Auto-generated captions (.txt per image)
├── training/
│   ├── configs/      # ai-toolkit YAML configs
│   ├── outputs/      # Trained LoRA weights
│   └── logs/         # Loss curves, sample images
├── inference/
│   ├── comfyui_workflows/   # ComfyUI JSON workflows
│   └── scripts/             # Python inference scripts
├── deploy/           # API server
├── scripts/          # Data pipeline scripts
└── docs/             # Notes, decisions, research
```

## Hardware

- GPU: NVIDIA RTX 4090 (24GB VRAM)
- FLUX.1-dev fits at ~20GB; LoRA training ~18-20GB

## Status

- [ ] Phase 1: Data collection pipeline
- [ ] Phase 2: Fine-tuning
- [ ] Phase 3: Deployment
