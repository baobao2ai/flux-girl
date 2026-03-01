#!/usr/bin/env python3
"""
generate_and_post.py — Generate an image with FLUX.1-schnell and post to Discord.

Usage:
    python scripts/generate_and_post.py --prompt "a beautiful portrait of a young woman"
    python scripts/generate_and_post.py --prompt "..." --steps 4 --seed 123
"""

import argparse, os, time, sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import torch
from huggingface_hub import login
from diffusers import FluxPipeline

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN     = os.getenv("HF_TOKEN")
WEBHOOK_URL  = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1477473621558689802/gzZMO-WtQhgeDcpjc-VKuRz59Dq2_5r7c8QLAbnw7og1hL08wfe-zOHD9tNYD6ARQMl2")
MODEL_ID     = "black-forest-labs/FLUX.1-schnell"
OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if HF_TOKEN:
    login(token=HF_TOKEN)

def load_pipeline():
    print(f"[+] Loading {MODEL_ID}...")
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe

def generate(pipe, prompt, steps, width, height, seed):
    print(f"[+] Generating: {prompt[:70]}...")
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
    out_path = OUTPUT_DIR / f"output_{int(time.time())}.png"
    image.save(out_path)
    print(f"[✓] Generated in {elapsed:.1f}s → {out_path.name}")
    return out_path, elapsed

def post_to_discord(image_path: Path, prompt: str, elapsed: float, seed: int):
    msg = f"🎨 **FLUX.1-schnell** | `{elapsed:.1f}s` | seed `{seed}`\n> {prompt}"
    with open(image_path, "rb") as f:
        r = requests.post(
            WEBHOOK_URL,
            data={"content": msg},
            files={"file": (image_path.name, f, "image/png")},
            timeout=30,
        )
    if r.status_code in (200, 204):
        print(f"[✓] Posted to Discord!")
    else:
        print(f"[ERR] Discord returned {r.status_code}: {r.text[:100]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps",  type=int, default=4)
    parser.add_argument("--width",  type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    pipe = load_pipeline()
    img_path, elapsed = generate(pipe, args.prompt, args.steps, args.width, args.height, args.seed)
    post_to_discord(img_path, args.prompt, elapsed, args.seed)

if __name__ == "__main__":
    main()
