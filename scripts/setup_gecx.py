#!/usr/bin/env python3
"""Tier 1 Provisioner: Gemini Enterprise for Customer Experience (GECX / ces.googleapis.com).

Configures the GECX Application, Root Agent (`lumiere_concierge_root_agent`),
Conditional Image Delegation Gate (`beforeModelCallbacks` + `beforeToolCallbacks`),
and Remote A2A Tool (`skincare_adk_agent_runtime_tool`).
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from oci_builder import get_auth_headers

BEFORE_MODEL_CALLBACK_CODE = """from typing import Any, List, Optional

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
  \"\"\"Sets skin_image_uploaded=True as soon as an uploaded image (inline_data) is present in the session.\"\"\"
  for content in (llm_request.contents or []):
    for part in (getattr(content, "parts", None) or []):
      inline_data = getattr(part, "inline_data", None)
      if inline_data and getattr(inline_data, "data", None):
        callback_context.variables["skin_image_uploaded"] = True
        return None
  return None
"""

BEFORE_TOOL_CALLBACK_CODE = """from typing import Any, Optional

def before_tool_callback(tool: Tool, input: dict[str, Any], callback_context: CallbackContext) -> Optional[dict[str, Any]]:
  \"\"\"Hard conditional gate: blocks ADK delegation until the customer uploads a skincare image.\"\"\"
  if tool.name == "skincare_adk_agent_runtime_tool" and not bool(callback_context.variables["skin_image_uploaded"]):
    return {
      "result": (
        "DELEGATION_BLOCKED: `skin_image_uploaded` is False. Do NOT delegate to the ADK agent yet. "
        "Call `request_skin_image_upload` and ask the customer to upload a clear photo first."
      )
    }
  return None
"""

REQUEST_IMAGE_UPLOAD_TOOL_CODE = """def request_skin_image_upload(reason: str = "visual_skin_analysis_required") -> dict:
  \"\"\"Prompts the customer to upload a photo of their skin concern or skincare product.\"\"\"
  return {
    "status": "awaiting_customer_image_upload",
    "reason": reason,
    "ui_action": "PROMPT_IMAGE_UPLOAD",
  }
"""

ROOT_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Clinical Skincare Concierge**, built on **Gemini Enterprise for Customer Experience (GECX)**.
</role>
<constraints>
1. **General Store Questions**: Answer shipping/return policy questions directly (complimentary 2-day cold-chain shipping over $50; 60-day clinical satisfaction guarantee).
2. **Conditional Delegation Rule (`skin_image_uploaded` Gate)**:
   - **If `skin_image_uploaded == false`** and the customer asks for a personal skin analysis or routine: do NOT call the ADK agent. Call `{@TOOL: request_skin_image_upload}` and ask them to upload a photo first.
   - **If `skin_image_uploaded == true`**: immediately call `{@TOOL: skincare_adk_agent_runtime_tool}` with `task` set to the customer's question, and relay the returned Gemini 3.8 Flash visual analysis (`analysis_id`, visible markers, active ingredients, and Lumiere SKUs).
</constraints>
"""


def setup_gecx_app(
    project_id: str,
    project_number: str,
    region: str,
    reasoning_engine_id: str,
    ces_location: str = "us",
    account: str | None = None,
) -> dict[str, Any]:
    """Creates or updates the GECX App, Tools, Root Agent, Version, and Deployment."""
    headers = get_auth_headers(account)
    ces_base = f"https://ces.googleapis.com/v1beta/projects/{project_id}/locations/{ces_location}"
    a2a_url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project_number}/locations/{region}/reasoningEngines/{reasoning_engine_id}/a2a/v1"
    )

    apps = requests.get(f"{ces_base}/apps", headers=headers, timeout=20).json().get("apps", [])
    display_name = "Lumiere Clinical Skincare Concierge (GECX + ADK + Cloud Run)"
    app_description = (
        "Customer Blueprint demonstrating Gemini Enterprise for Customer Experience (GECX) "
        "conditionally delegating uploaded skincare images to a Google ADK Agent on Vertex AI "
        "Agent Runtime, which invokes a Cloud Run Gemini 3.8 Flash Skincare Vision API."
    )
    app_obj = next((a for a in apps if "Lumiere Clinical Skincare Concierge" in a.get("displayName", "")), None)

    var_decls = [
        {
            "name": "skin_image_uploaded",
            "description": "True when the customer has uploaded a skincare image in the session.",
            "schema": {"type": "BOOLEAN", "default": False},
        }
    ]

    if app_obj is None:
        r_create = requests.post(
            f"{ces_base}/apps",
            headers=headers,
            json={
                "displayName": display_name,
                "description": app_description,
                "pinned": True,
                "modelSettings": {"model": "gemini-3.0-flash-001"},
                "variableDeclarations": var_decls,
                "globalInstruction": "SESSION STATE: skin_image_uploaded={skin_image_uploaded}",
                "loggingSettings": {"cloudLoggingSettings": {"enableCloudLogging": True}},
                "timeZoneSettings": {"timeZone": "America/Los_Angeles"},
            },
            timeout=30,
        )
        r_create.raise_for_status()
        op = r_create.json()
        while "operations/" in op.get("name", "") and not op.get("done"):
            time.sleep(2)
            op = requests.get(f"https://ces.googleapis.com/v1beta/{op['name']}", headers=headers, timeout=15).json()
        app_obj = op.get("response", op)

    app_name = app_obj["name"]
    existing_tools = {
        t.get("displayName"): t
        for t in requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, timeout=15).json().get("tools", [])
    }

    req_tool_payload = {
        "executionType": "SYNCHRONOUS",
        "pythonFunction": {
            "name": "request_skin_image_upload",
            "description": "Prompts the customer to upload a skincare image before visual analysis.",
            "pythonCode": REQUEST_IMAGE_UPLOAD_TOOL_CODE,
        },
    }
    if "request_skin_image_upload" not in existing_tools:
        r_t1 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=req_tool_payload, timeout=20)
    else:
        r_t1 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['request_skin_image_upload']['name']}", headers=headers, json=req_tool_payload, timeout=20)
    r_t1.raise_for_status()
    tool_req_name = r_t1.json()["name"]

    re_tool_payload = {
        "executionType": "SYNCHRONOUS",
        "timeout": "60s",
        "remoteAgentTool": {
            "name": "skincare_adk_agent_runtime_tool",
            "description": "Delegates skincare image analysis to the Google ADK Agent on Vertex AI Agent Runtime.",
            "agentCard": {
                "name": "skincare_specialist_adk_agent_runtime",
                "description": "ADK Skincare Specialist Agent on Vertex AI Agent Runtime.",
                "version": "1.0.0",
                "supportedInterfaces": [{"url": a2a_url, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
                "skills": [{"id": "skincare_visual_diagnosis", "name": "Skincare Image Analysis", "description": "Analyzes uploaded skin photos via Cloud Run Gemini 3.8 Flash API."}],
            },
        },
    }
    if "skincare_adk_agent_runtime_tool" not in existing_tools:
        r_t2 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=re_tool_payload, timeout=20)
    else:
        r_t2 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['skincare_adk_agent_runtime_tool']['name']}", headers=headers, json=re_tool_payload, timeout=20)
    r_t2.raise_for_status()
    tool_re_name = r_t2.json()["name"]

    agents = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, timeout=15).json().get("agents", [])
    root_ag = next((a for a in agents if a.get("displayName") == "lumiere_concierge_root_agent"), agents[0] if agents else None)
    ag_payload = {
        "displayName": "lumiere_concierge_root_agent",
        "description": (
            "Primary GECX Concierge Agent that enforces conditional delegation (skin_image_uploaded) "
            "before invoking the Google ADK Skincare Specialist Agent on Vertex AI Agent Runtime."
        ),
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": ROOT_AGENT_INSTRUCTION,
        "tools": [tool_req_name, tool_re_name],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
        "beforeToolCallbacks": [{"description": "Blocks ADK delegation if skin_image_uploaded is False.", "pythonCode": BEFORE_TOOL_CALLBACK_CODE}],
    }
    if root_ag is None:
        r_ag = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, json=ag_payload, timeout=20)
    else:
        r_ag = requests.patch(f"https://ces.googleapis.com/v1beta/{root_ag['name']}", headers=headers, json=ag_payload, timeout=20)
    r_ag.raise_for_status()
    root_ag_name = r_ag.json()["name"]

    requests.patch(
        f"https://ces.googleapis.com/v1beta/{app_name}?updateMask=rootAgent,variableDeclarations,globalInstruction,modelSettings,pinned,displayName,description",
        headers=headers,
        json={
            "name": app_name,
            "displayName": display_name,
            "description": app_description,
            "pinned": True,
            "rootAgent": root_ag_name,
            "variableDeclarations": var_decls,
            "modelSettings": {"model": "gemini-3.0-flash-001"},
            "globalInstruction": "SESSION STATE: skin_image_uploaded={skin_image_uploaded}",
        },
        timeout=20,
    ).raise_for_status()

    # Ensure published AppVersion and Deployment reflect Gemini 3.8 Flash (no legacy naming)
    ver_display = "v1.0 - GECX + ADK + Cloud Run Gemini 3.8 Flash Blueprint"
    r_vers = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/versions", headers=headers, timeout=15)
    versions = r_vers.json().get("appVersions", []) if r_vers.status_code == 200 else []
    clean_ver = next((v for v in versions if v.get("displayName") == ver_display), None)
    if clean_ver is None:
        r_vc = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/versions",
            headers=headers,
            json={
                "displayName": ver_display,
                "description": "Published release with conditional image delegation gate to ADK Agent Runtime + Cloud Run Gemini 3.8 Flash Vision API.",
            },
            timeout=20,
        )
        if r_vc.status_code == 200:
            clean_ver = r_vc.json()

    if clean_ver:
        ver_name = clean_ver["name"]
        deps = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/deployments", headers=headers, timeout=15).json().get("deployments", [])
        if deps:
            requests.patch(
                f"https://ces.googleapis.com/v1beta/{deps[0]['name']}?updateMask=appVersion",
                headers=headers,
                json={"name": deps[0]["name"], "appVersion": ver_name},
                timeout=20,
            )
        else:
            requests.post(
                f"https://ces.googleapis.com/v1beta/{app_name}/deployments",
                headers=headers,
                json={"displayName": "Lumiere Web & API Live Deployment", "appVersion": ver_name, "channelProfile": {"channelType": "WEB_UI"}},
                timeout=20,
            )
        for v in versions:
            if v["name"] != ver_name and ("nano" in json.dumps(v).lower() or "banana" in json.dumps(v).lower()):
                requests.delete(f"https://ces.googleapis.com/v1beta/{v['name']}", headers=headers, timeout=15)

    return {
        "location": ces_location,
        "gecx_app_id": app_name.split("/")[-1],
        "gecx_app_name": app_name,
        "gecx_console_url": f"https://ces.cloud.google.com/{app_name}",
        "reasoning_engine_a2a_url": a2a_url,
    }
