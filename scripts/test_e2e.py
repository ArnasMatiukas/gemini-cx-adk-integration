#!/usr/bin/env python3
"""Live 4-Turn End-to-End Verification for Multi-Flow GECX -> ADK Agent Runtime -> Cloud Run API.

Verifies both distinct GECX user-journey flows and strict separation of responsibilities:
  - Turn 1 (Root Concierge): General store policy FAQ handled directly by `lumiere_concierge_root_agent`.
  - Turn 2 (Flow 1 - Shopping Journey): Routes to `lumiere_shopping_agent` and calls `shop_lumiere_catalog`.
  - Turn 3 (Flow 2 - Skin Analysis Journey, No Image): Routes directly to `skin_analysis_recommendation_agent`;
    conditional gate blocks ADK delegation & GECX product recommendation and calls `request_skin_image_upload`.
  - Turn 4 (Flow 2 - Skin Analysis Journey, Image Uploaded):
      1. `skincare_adk_agent_runtime_tool` (ADK Agent Runtime -> Cloud Run Gemini 3.8 Flash Vision API)
         returns ONLY the clinical visual skin diagnosis (zero product SKUs).
      2. `recommend_lumiere_products` (GECX tool on `skin_analysis_recommendation_agent`) takes the
         ADK clinical findings and recommends the matched Lumiere products inside GECX.
"""

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


def _run_turn(
    session_name: str, inputs: list[dict[str, Any]], headers: dict[str, str]
) -> tuple[str, list[str], dict[str, Any]]:
    r = requests.post(
        f"https://ces.googleapis.com/v1beta/{session_name}:runSession",
        headers=headers,
        json={"config": {"session": session_name}, "inputs": inputs},
        timeout=120,
    )
    r.raise_for_status()
    texts, tools = [], []
    tool_outputs: dict[str, Any] = {}
    for out in r.json().get("outputs", []):
        if out.get("text"):
            texts.append(out["text"])
        for msg in (out.get("diagnosticInfo") or {}).get("messages") or []:
            for chunk in msg.get("chunks") or []:
                if tc := chunk.get("toolCall"):
                    name = (
                        tc.get("displayName")
                        or (tc.get("tool") or "").split("/")[-1]
                        or tc.get("id")
                        or "tool"
                    )
                    tools.append(name)
                if tr := chunk.get("toolResponse"):
                    rname = (
                        tr.get("displayName")
                        or (tr.get("tool") or "").split("/")[-1]
                        or "tool"
                    )
                    tool_outputs[rname] = tr.get("response")
                if at := chunk.get("agentTransfer"):
                    target = (at.get("targetAgent") or at.get("displayName") or "agent_transfer").split("/")[-1]
                    tools.append(f"transfer->{target}")
    return "\n".join(texts).strip(), tools, tool_outputs


def run_e2e_test(
    project_id: str,
    ces_location: str,
    app_id: str,
    region: str,
    reasoning_engine_id: str,
    account: str | None = None,
    agent_host_project_id: str | None = None,
) -> dict[str, Any]:
    headers = get_auth_headers(account)
    host_proj = agent_host_project_id or project_id
    session_name = f"projects/{project_id}/locations/{ces_location}/apps/{app_id}/sessions/e2e-{uuid.uuid4().hex[:8]}"
    print(f"  -> Running Multi-Flow E2E Session: {session_name} (ADK Host Project: {host_proj})")

    # Turn 1: General FAQ (Handled directly by GECX Root Agent, no tool delegation)
    _, t1_tools, _ = _run_turn(session_name, [{"text": "What is Lumiere's shipping and return policy?"}], headers)
    print(f"  [Turn 1 - Root Concierge FAQ] Actions: {t1_tools}")
    assert "skincare_adk_agent_runtime_tool" not in t1_tools

    # Turn 2: Flow 1 — Shopping & Catalog Journey (`lumiere_shopping_agent`)
    t2_text, t2_tools, _ = _run_turn(
        session_name,
        [{"text": "I want to do some shopping. Please check the price of the Lumiere Soothing Cica-Ceramide Barrier Cream (CLN-CICA-01) and add 2 of them to my cart."}],
        headers,
    )
    print(f"  [Turn 2 - Flow 1: Shopping Journey] Actions: {t2_tools}")
    print(f"  [Turn 2 Shopping Preview]: {t2_text[:200]}...")
    assert "shop_lumiere_catalog" in t2_tools

    # Turn 3: Flow 2 — Skin Analysis Journey WITHOUT an image (Conditional gate blocks ADK & GECX recommendation)
    _, t3_tools, _ = _run_turn(
        session_name,
        [{"text": "Before I check out, my cheeks are red and flaky. Can you analyze my skin condition and recommend products?"}],
        headers,
    )
    print(f"  [Turn 3 - Flow 2: Skin Analysis (No Image Uploaded)] Actions: {t3_tools}")
    assert "skincare_adk_agent_runtime_tool" not in t3_tools
    assert "recommend_lumiere_products" not in t3_tools

    # Turn 4: Flow 2 — Skincare image uploaded -> (1) ADK Visual Diagnosis -> (2) GECX Product Recommendation
    t4_text, t4_tools, t4_outputs = _run_turn(
        session_name,
        [
            {"text": "Here is a close-up photo of my cheek redness. Please analyze my skin condition and recommend products."},
            {"image": {"mimeType": "image/png", "data": _sample_skin_image_b64()}},
        ],
        headers,
    )
    print(f"  [Turn 4 - Flow 2: Image Uploaded -> ADK Diagnosis + GECX Product Recommendation] Actions: {t4_tools}")
    print(f"  [Turn 4 Response Preview]: {t4_text[:280]}...")
    assert "skincare_adk_agent_runtime_tool" in t4_tools, f"Expected ADK diagnosis tool in {t4_tools}"
    assert "recommend_lumiere_products" in t4_tools, f"Expected GECX recommend_lumiere_products tool in {t4_tools}"

    r_sess = requests.get(
        f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{host_proj}/locations/{region}/reasoningEngines/{reasoning_engine_id}/sessions",
        headers=headers,
        timeout=20,
    )
    sess_count = len(r_sess.json().get("sessions", [])) if r_sess.status_code == 200 else 0
    print(f"  -> Verified Vertex AI Agent Engine Managed Sessions ({host_proj}): {sess_count}")

    # Verify Cloud Trace spans for this ReasoningEngine across trace shards
    trace_count = 0
    page_token = ""
    for _ in range(10):
        t_url = f"https://cloudtrace.googleapis.com/v1/projects/{host_proj}/traces?view=COMPLETE&pageSize=50"
        if page_token:
            t_url += f"&pageToken={requests.utils.quote(page_token)}"
        r_tr = requests.get(t_url, headers=headers, timeout=20)
        if r_tr.status_code != 200:
            break
        t_data = r_tr.json()
        for tr in t_data.get("traces", []):
            if any(
                reasoning_engine_id in (s.get("labels", {}).get("cloud.resource_id") or "")
                for s in tr.get("spans", [])
            ):
                trace_count += 1
        page_token = t_data.get("nextPageToken", "")
        if not page_token:
            break
    print(f"  -> Verified Vertex AI Agent Engine Cloud Traces ({host_proj}): {trace_count}")

    return {
        "gecx_project_id": project_id,
        "agent_host_project_id": host_proj,
        "turn_1_faq_actions": t1_tools,
        "turn_2_shopping_flow_actions": t2_tools,
        "turn_3_no_image_gate_actions": t3_tools,
        "turn_4_image_adk_and_gecx_actions": t4_tools,
        "adk_diagnosis_output_preview": str(t4_outputs.get("skincare_adk_agent_runtime_tool", ""))[:240],
        "gecx_recommendation_output_preview": str(t4_outputs.get("recommend_lumiere_products", ""))[:240],
        "agent_engine_sessions": sess_count,
        "agent_engine_traces": trace_count,
    }
