#!/usr/bin/env python3
"""Standalone CLI entrypoint for the Google ADK Skincare Specialist Agent."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw

from skincare_adk_agent.a2a_server import run_adk_specialist


def generate_sample_skin_image() -> str:
    """Generates a deterministic synthetic cheek skin closeup PNG (base64) with redness and dry flaking."""
    img = Image.new("RGB", (384, 384), color=(235, 195, 180))
    draw = ImageDraw.Draw(img)
    draw.ellipse([90, 110, 290, 270], fill=(225, 130, 125))
    draw.ellipse([130, 140, 250, 240], fill=(215, 105, 100))
    for x, y in [(150, 160), (180, 190), (210, 170), (165, 210), (200, 205)]:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(245, 230, 225))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Google ADK Skincare Specialist Agent.")
    parser.add_argument(
        "--prompt",
        default="Analyze my cheek redness and dry patches and recommend a Lumiere routine.",
        help="User query for the ADK Skincare Specialist Agent.",
    )
    parser.add_argument("--image", default=None, help="Optional path to a PNG/JPEG skincare image.")
    parser.add_argument("--json", action="store_true", help="Output structured JSON.")
    args = parser.parse_args()

    if args.image:
        img_bytes = Path(args.image).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        mime_type = "image/jpeg" if args.image.lower().endswith((".jpg", ".jpeg")) else "image/png"
    else:
        img_b64 = generate_sample_skin_image()
        mime_type = "image/png"

    resp_text, telemetry = asyncio.run(
        run_adk_specialist(
            user_text=args.prompt,
            image_b64=img_b64,
            mime_type=mime_type,
            context_id="cli-session",
        )
    )
    if args.json:
        print(
            json.dumps(
                {"response": resp_text, "cloud_run_nano_banana_telemetry": telemetry},
                indent=2,
            )
        )
    else:
        print(resp_text)


if __name__ == "__main__":
    main()
