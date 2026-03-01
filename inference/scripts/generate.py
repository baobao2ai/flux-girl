#!/usr/bin/env python3
"""
generate.py — Run FLUX.1-schnell on RTX 4090.

Usage:
    python inference/scripts/generate.py --prompt "a beautiful portrait of a young woman"
    python inference/scripts/generate.py --prompt "..." --steps 4 --width 1024 --height 1024 --seed 42
"""

import argparse, os, time
from pathlib import Path
from huggingface_hub import login
import torch
from diffusers import FluxPipeline

# Auth
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

MODEL_ID   = "black-forest-labs/FLUX.1-schnell"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_pipeline():
    print(f"[+] Loading {MODEL_ID}...")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    # Enable sequential CPU offload — loads each component to GPU only when needed
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    print(f"[+] Loaded in {time.time()-t0:.1f}s (sequential CPU offload enabled)")
    return pipe

def generate(pipe, prompt, steps, width, height, seed, out):
    print(f"[+] Generating ({steps} steps, {width}x{height}, seed {seed})...")
    generator = torch.Generator("cpu").manual_seed(seed)
    t0 = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator,
        guidance_scale=0.0,
    ).images[0]
    elapsed = time.time() - t0
    out_path = OUTPUT_DIR / out
    image.save(out_path)
    print(f"[✓] Saved: {out_path} ({elapsed:.1f}s)")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps",  type=int, default=4)
    parser.add_argument("--width",  type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--out",    default=f"output_{int(time.time())}.png")
    args = parser.parse_args()
    pipe = load_pipeline()
    generate(pipe, args.prompt, args.steps, args.width, args.height, args.seed, args.out)

if __name__ == "__main__":
    main()
