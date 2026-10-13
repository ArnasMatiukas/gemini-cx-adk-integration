#!/usr/bin/env python3
"""Live 3-Turn End-to-End Verification for GECX -> ADK Agent Runtime -> Cloud Run API."""

from __future__ import annotations

import base64
import io
import uuid
from typing import Any

from PIL import Image, ImageDraw
import requests

from oci_builder import get_auth_headers


def _sample_skin_image_b64() -> str:
    img = Image.new("RGB", (320, 320), color=(235, 195, 180))
    draw = ImageDraw.Draw(img)
    draw.ellipse([70, 90, 250, 230], fill=(220, 115, 110))
    for x, y in [(130, 140), (160, 165), (185, 150)]:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(245, 235, 230))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _run_turn(session_name: str, inputs: list[dict[str, Any]], headers: dict[str, str]) -> tuple[str, list[str]]:
    r = requests.post(
        f"https://ces.googleapis.com/v1beta/{session_name}:runSession",
        headers=headers,
        json={"config": {"session": session_name}, "inputs": inputs},
        timeout=120,
    )
    r.raise_for_status()
    texts, tools = [], []
    for out in r.json().get("outputs", []):
        if out.get("text"):
            texts.append(out["text"])
        for msg in (out.get("diagnosticInfo") or {}).get("messages") or []:
            for chunk in msg.get("chunks") or []:
                if tc := chunk.get("toolCall"):
                    tools.append(tc.get("displayName") or tc.get("id") or "tool")
    return "\n".join(texts).strip(), tools


def run_e2e_test(
    project_id: str,
    ces_location: str,
    app_id: str,
    region: str,
    reasoning_engine_id: str,
    account: str | None = None,
) -> dict[str, Any]:
    headers = get_auth_headers(account)
    session_name = f"projects/{project_id}/locations/{ces_location}/apps/{app_id}/sessions/e2e-{uuid.uuid4().hex[:8]}"
    print(f"  -> Running E2E Session: {session_name}")

    # Turn 1: General FAQ (Handled directly by GECX, no delegation)
    _, t1_tools = _run_turn(session_name, [{"text": "What is Lumiere's shipping and return policy?"}], headers)
    print(f"  [Turn 1 - General FAQ] Tools called: {t1_tools}")
    assert "skincare_adk_agent_runtime_tool" not in t1_tools

    # Turn 2: Skin analysis requested WITHOUT an image (Conditional gate blocks ADK delegation)
    _, t2_tools = _run_turn(
        session_name,
        [{"text": "My cheeks are red and flaky. Can you analyze my skin and give me a routine?"}],
        headers,
    )
    print(f"  [Turn 2 - No Image Uploaded] Tools called: {t2_tools}")
    assert "skincare_adk_agent_runtime_tool" not in t2_tools or "request_skin_image_upload" in t2_tools

    # Turn 3: Skincare image uploaded (Condition met -> GECX -> ADK on Agent Runtime -> Cloud Run Gemini 3.8 Flash API)
    t3_text, t3_tools = _run_turn(
        session_name,
        [
            {"text": "Here is a close-up photo of my cheek redness. Please analyze it and recommend a routine."},
            {"image": {"mimeType": "image/png", "data": _sample_skin_image_b64()}},
        ],
        headers,
    )
    print(f"  [Turn 3 - Image Uploaded -> 3-Tier Delegation] Tools called: {t3_tools}")
    print(f"  [Turn 3 Response Preview]: {t3_text[:280]}...")
    assert "skincare_adk_agent_runtime_tool" in t3_tools

    r_sess = requests.get(
        f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/reasoningEngines/{reasoning_engine_id}/sessions",
        headers=headers,
        timeout=20,
    )
    sess_count = len(r_sess.json().get("sessions", [])) if r_sess.status_code == 200 else 0
    print(f"  -> Verified Vertex AI Agent Engine Managed Sessions: {sess_count}")
    return {"turn_1_tools": t1_tools, "turn_2_tools": t2_tools, "turn_3_tools": t3_tools, "agent_engine_sessions": sess_count}
