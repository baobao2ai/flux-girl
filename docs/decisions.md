# Decisions & Research Log

## 2026-02-28

### Base Model: FLUX.1-dev
- Best quality for realistic portraits
- 24GB VRAM fits on RTX 4090 (~20GB at bf16)
- Licensed for fine-tuning / research

### Fine-Tuning Method: LoRA
- Training tool: ai-toolkit by Ostris (best FLUX LoRA support)
- LoRA rank: TBD (start with rank 16-32)
- Steps: TBD (start with 1000-2000)
- Works on 4090 with gradient checkpointing + bf16

### Data Strategy
- Target: 200-500 high-quality curated images
- Auto-caption with Florence-2 (Microsoft) or JoyCaption
- Resolution: 1024x1024 (FLUX native)

### Deployment
- ComfyUI for visual workflow testing
- Custom FastAPI endpoint for programmatic access
