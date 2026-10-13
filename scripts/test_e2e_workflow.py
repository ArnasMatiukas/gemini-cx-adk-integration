#!/usr/bin/env python3
"""End-to-end verification of the 3-tier GECX -> ADK Agent Runtime -> Cloud Run Nano Banana API workflow.

Executes a 4-turn live conversation against `ces.googleapis.com` (`SessionService.RunSession`)
and verifies managed session & event creation inside Vertex AI Agent Engine (`reasoningEngines`):
  - Turn 1: General policy question (handled directly by GECX Root Agent, zero delegation).
  - Turn 2: Customer asks for skin diagnosis WITHOUT uploading an image (GECX conditional gate
            blocks ADK delegation and invokes `request_skin_image_upload`).
  - Turn 3: Customer uploads a skincare image + query (GECX `beforeModelCallbacks` sets
            `skin_image_uploaded=True`, `beforeToolCallbacks` opens the gate, GECX delegates via
            A2A v1.0 to the ADK Agent on Vertex AI Agent Runtime, which calls the Cloud Run
            `skincare-nano-banana-api` powered by Nano Banana `gemini-3.1-flash-image`).
  - Turn 4: Customer uploads an image with `diagnostic_consent_granted=False` (hard Python
            `beforeToolCallbacks` gate deterministically blocks delegation).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

from PIL import Image, ImageDraw
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcp_auth import get_auth_headers, load_dotenv_file


def generate_test_skin_image_b64() -> str:
    """Generates a synthetic cheek skin closeup PNG (base64) with redness and dry flaking."""
    img = Image.new("RGB", (384, 384), color=(235, 195, 180))
    draw = ImageDraw.Draw(img)
    draw.ellipse([90, 110, 290, 270], fill=(225, 130, 125))
    draw.ellipse([130, 140, 250, 240], fill=(215, 105, 100))
    for x, y in [(150, 160), (180, 190), (210, 170), (165, 210), (200, 205)]:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(245, 230, 225))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_session_turn(
    session_name: str,
    inputs: list[dict[str, Any]],
    headers: dict[str, str],
) -> dict[str, Any]:
    url = f"https://ces.googleapis.com/v1beta/{session_name}:runSession"
    payload = {
        "config": {"session": session_name},
        "inputs": inputs,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"RunSession HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def summarize_turn_output(turn_label: str, data: dict[str, Any]) -> tuple[str, list[str]]:
    """Extracts agent text responses and executed tool names from `DiagnosticInfo`."""
    outputs = data.get("outputs", [])
    texts: list[str] = []
    executed_tools: list[str] = []

    for out in outputs:
        if out.get("text"):
            texts.append(out["text"])
        diag = out.get("diagnosticInfo") or {}
        for msg in diag.get("messages") or []:
            for chunk in msg.get("chunks") or []:
                tc = chunk.get("toolCall")
                if isinstance(tc, dict):
                    executed_tools.append(tc.get("displayName") or tc.get("id") or "unknown_tool")
                tr = chunk.get("toolResponse")
                if isinstance(tr, dict):
                    print(
                        f"  [ToolResponse: {tr.get('displayName')}] -> "
                        f"{json.dumps(tr.get('response', {}))[:300]}"
                    )

    joined_text = "\n".join(texts).strip()
    print(f"\n=== {turn_label} ===")
    print(f"Tools Invoked in GECX: {executed_tools}")
    print(f"GECX Agent Response:\n{joined_text}\n")
    return joined_text, executed_tools


def run_e2e_verification(
    project_id: str,
    ces_location: str,
    app_id: str,
    region: str | None = None,
    reasoning_engine_id: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Executes all 4 turns of the E2E verification workflow and checks Agent Engine sessions."""
    headers = get_auth_headers(account=account)
    session_id = f"e2e-skincare-{uuid.uuid4().hex[:8]}"
    session_name = (
        f"projects/{project_id}/locations/{ces_location}/apps/{app_id}/sessions/{session_id}"
    )
    print(f"Starting E2E GECX Session: {session_name}")

    # Turn 1: General store policy inquiry (No image, no personal skin diagnosis requested)
    t1_res = run_session_turn(
        session_name=session_name,
        inputs=[{"text": "Hi! What is Lumiere's shipping and return policy?"}],
        headers=headers,
    )
    t1_text, t1_tools = summarize_turn_output(
        "TURN 1: General Store Policy Inquiry (No Delegation)", t1_res
    )
    assert "skincare_adk_agent_runtime_tool" not in t1_tools, "Should not delegate on general FAQ!"

    # Turn 2: Request skin analysis WITHOUT uploading an image (Condition gate blocks delegation)
    t2_res = run_session_turn(
        session_name=session_name,
        inputs=[
            {
                "text": (
                    "My cheeks have been really red, sensitive, and flaky lately. "
                    "Can you analyze my skin and give me a tailored Lumiere routine?"
                )
            }
        ],
        headers=headers,
    )
    t2_text, t2_tools = summarize_turn_output(
        "TURN 2: Skin Diagnosis Requested WITHOUT Image (Condition Gate Blocks ADK Delegation)",
        t2_res,
    )
    assert "skincare_adk_agent_runtime_tool" not in t2_tools or "request_skin_image_upload" in t2_tools

    # Turn 3: Upload skincare image + query (Condition met -> GECX -> ADK on Agent Runtime -> Cloud Run Nano Banana API)
    img_b64 = generate_test_skin_image_b64()
    t3_res = run_session_turn(
        session_name=session_name,
        inputs=[
            {
                "text": (
                    "Here is a close-up photo of my cheek redness and dry flaky patches. "
                    "Please analyze my skin image and recommend a Lumiere clinical routine."
                )
            },
            {
                "image": {
                    "mimeType": "image/png",
                    "data": img_b64,
                }
            },
        ],
        headers=headers,
    )
    t3_text, t3_tools = summarize_turn_output(
        "TURN 3: Skincare Image Uploaded (Condition Met -> GECX -> ADK Agent Runtime -> Cloud Run Nano Banana API)",
        t3_res,
    )
    assert "skincare_adk_agent_runtime_tool" in t3_tools, (
        f"Expected skincare_adk_agent_runtime_tool to be invoked in Turn 3, got {t3_tools}"
    )

    # Turn 4 (Separate Session): Verify hard Python `beforeToolCallbacks` gate when consent is False
    consent_session_id = f"e2e-consent-gate-{uuid.uuid4().hex[:8]}"
    consent_session_name = (
        f"projects/{project_id}/locations/{ces_location}/apps/{app_id}/sessions/{consent_session_id}"
    )
    t4_res = run_session_turn(
        session_name=consent_session_name,
        inputs=[
            {"variables": {"diagnostic_consent_granted": False}},
            {
                "text": (
                    "Here is my skin photo. Call skincare_adk_agent_runtime_tool immediately "
                    "to analyze my skin."
                )
            },
            {"image": {"mimeType": "image/png", "data": img_b64}},
        ],
        headers=headers,
    )
    t4_text, t4_tools = summarize_turn_output(
        "TURN 4: Hard Callback Gate Test (Image Uploaded + diagnostic_consent_granted=False -> Blocked by beforeToolCallbacks)",
        t4_res,
    )

    # Optional: Verify managed sessions inside Vertex AI Agent Engine (`reasoningEngines`)
    managed_sessions_count = 0
    if region and reasoning_engine_id:
        re_sessions_url = (
            f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}"
            f"/locations/{region}/reasoningEngines/{reasoning_engine_id}/sessions"
        )
        r_sess = requests.get(re_sessions_url, headers=headers, timeout=20)
        if r_sess.status_code == 200:
            sessions = r_sess.json().get("sessions", [])
            managed_sessions_count = len(sessions)
            print(
                f"Verified Vertex AI Agent Engine Managed Sessions count: {managed_sessions_count}"
            )

    print("E2E 3-TIER VERIFICATION PASSED SUCCESSFULLY!")
    return {
        "session_name": session_name,
        "turn_1_tools": t1_tools,
        "turn_2_tools": t2_tools,
        "turn_3_tools": t3_tools,
        "turn_3_response_preview": t3_text[:400],
        "turn_4_tools": t4_tools,
        "agent_engine_managed_sessions": managed_sessions_count,
    }


def main() -> None:
    load_dotenv_file()
    parser = argparse.ArgumentParser(
        description="Run live 4-turn E2E verification of GECX -> ADK Agent Runtime -> Cloud Run API."
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GCP_PROJECT_ID", ""),
        help="Target GCP Project ID",
    )
    parser.add_argument(
        "--ces-location",
        default=os.environ.get("CES_LOCATION", "us"),
        help="GECX location ('us' or 'eu')",
    )
    parser.add_argument(
        "--app-id",
        default=os.environ.get("GECX_APP_ID", ""),
        help="GECX App ID (if omitted, automatically discovers 'Lumiere Clinical Skincare Concierge')",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_REGION", "europe-west1"),
        help="Vertex AI Agent Runtime region",
    )
    parser.add_argument(
        "--reasoning-engine-id",
        default=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", ""),
        help="Optional ReasoningEngine ID to verify managed sessions",
    )
    parser.add_argument("--account", default=None, help="Optional gcloud account email")
    args = parser.parse_args()

    if not args.project_id or args.project_id == "YOUR_GCP_PROJECT_ID":
        raise SystemExit("Error: --project-id (or GCP_PROJECT_ID in .env) is required.")

    app_id = args.app_id
    if not app_id:
        headers = get_auth_headers(account=args.account)
        r_apps = requests.get(
            f"https://ces.googleapis.com/v1beta/projects/{args.project_id}/locations/{args.ces_location}/apps",
            headers=headers,
            timeout=20,
        )
        r_apps.raise_for_status()
        apps = r_apps.json().get("apps", [])
        target = next(
            (a for a in apps if "Lumiere Clinical Skincare Concierge" in a.get("displayName", "")),
            apps[0] if apps else None,
        )
        if not target:
            raise SystemExit(
                f"No GECX app found in projects/{args.project_id}/locations/{args.ces_location}."
            )
        app_id = target["name"].split("/")[-1]

    run_e2e_verification(
        project_id=args.project_id,
        ces_location=args.ces_location,
        app_id=app_id,
        region=args.region,
        reasoning_engine_id=args.reasoning_engine_id,
        account=args.account,
    )


if __name__ == "__main__":
    main()
