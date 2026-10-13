#!/usr/bin/env python3
"""Provisions the Gemini Enterprise for Customer Experience (GECX / ces.googleapis.com) Application.

Configures the complete 3-tier architecture in `projects/<PROJECT_ID>/locations/<CES_LOCATION>/apps`:
  Tier 1: Gemini Enterprise for Customer Experience (`lumiere_concierge_root_agent`)
          with a two-layer Conditional Delegation Gate (`beforeModelCallbacks` + `beforeToolCallbacks`
          + `<taskflow>` instructions checking `skin_image_uploaded`).
  Tier 2: Google ADK Skincare Specialist Agent (`gemini-3.8-flash`) deployed on
          Vertex AI Agent Runtime (`projects/<PROJECT_NUMBER>/locations/<REGION>/reasoningEngines/<ENGINE_ID>`)
  Tier 3: Cloud Run Skincare Image Analysis API (`skincare-nano-banana-api`)
          powered by Nano Banana 2 (`gemini-3.1-flash-image`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcp_auth import get_auth_headers, get_project_number, load_dotenv_file


def wait_for_ces_operation(op_name: str, headers: dict[str, str]) -> dict[str, Any]:
    """Polls a CES long-running operation until completion."""
    url = f"https://ces.googleapis.com/v1beta/{op_name}"
    for _ in range(90):
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("done"):
            if "error" in data:
                raise RuntimeError(f"CES Operation failed: {json.dumps(data['error'])}")
            return data.get("response", data)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for operation {op_name}")


BEFORE_MODEL_CALLBACK_CODE = """from typing import Any, List, Optional

def has_uploaded_image(contents: Optional[List[Any]]) -> bool:
  \"\"\"Checks if any turn in the conversation contains an uploaded image (inline_data).\"\"\"
  if not contents:
    return False
  for content in contents:
    parts = getattr(content, "parts", []) or []
    for part in parts:
      inline_data = getattr(part, "inline_data", None)
      if inline_data and getattr(inline_data, "data", None):
        return True
  return False

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
  \"\"\"Evaluates the GECX delegation condition before invoking the model.\"\"\"
  if has_uploaded_image(llm_request.contents):
    callback_context.variables["skin_image_uploaded"] = True
    callback_context.variables["delegation_gate_status"] = "CONDITION_MET_IMAGE_UPLOADED"
  return None
"""

BEFORE_TOOL_CALLBACK_CODE = """from typing import Any, Optional

def before_tool_callback(tool: Tool, input: dict[str, Any], callback_context: CallbackContext) -> Optional[dict[str, Any]]:
  \"\"\"Enforces the hard condition gate before delegating to the ADK Skincare Specialist Agent.\"\"\"
  protected_adk_tools = {
    "skincare_adk_agent_runtime_tool",
  }
  if tool.name in protected_adk_tools:
    image_uploaded = bool(callback_context.variables["skin_image_uploaded"])
    consent_granted = bool(callback_context.variables["diagnostic_consent_granted"])
    if not image_uploaded:
      callback_context.variables["delegation_gate_status"] = "BLOCKED_MISSING_IMAGE"
      return {
        "result": (
          "DELEGATION_CONDITION_NOT_MET: `skin_image_uploaded` is False. "
          "You MUST NOT delegate to the ADK Skincare Specialist Agent until the customer uploads "
          "a skin or skincare product photo. Call `request_skin_image_upload` and ask the customer "
          "to upload a clear photo first."
        )
      }
    if not consent_granted:
      callback_context.variables["delegation_gate_status"] = "BLOCKED_MISSING_CONSENT"
      return {
        "result": (
          "DELEGATION_CONDITION_NOT_MET: `diagnostic_consent_granted` is False. "
          "Please ask the customer to confirm consent for visual skin analysis before delegating."
        )
      }
    callback_context.variables["delegation_gate_status"] = f"DELEGATED_TO_{tool.name.upper()}"
  return None
"""

REQUEST_IMAGE_UPLOAD_TOOL_CODE = """def request_skin_image_upload(reason: str = "visual_skin_analysis_required") -> dict:
  \"\"\"Prompts the customer in the UI to upload a photo of their skin concern or skincare product.\"\"\"
  return {
    "status": "awaiting_customer_image_upload",
    "delegation_condition_met": False,
    "reason": reason,
    "ui_action": "PROMPT_IMAGE_UPLOAD",
    "instructions_for_agent": (
      "Ask the customer to upload a clear photo of their skin concern or skincare product "
      "so the clinical ADK specialist and Nano Banana vision engine can analyze it."
    ),
  }
"""

ROOT_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Clinical Skincare Concierge**, the primary customer experience agent built on **Gemini Enterprise for Customer Experience (GECX)**.
You handle general store inquiries, shipping/return policy questions, and orchestrate conditional delegation to the **Google ADK Skincare Specialist Agent** (which calls the **Cloud Run Nano Banana Skincare API**).
</role>

<persona>
Maintain a warm, objective, clinical-luxury concierge tone. Keep responses structured and concise.
</persona>

<constraints>
1. **Strict Conditional Delegation Rule (`skin_image_uploaded` Gate)**:
   - **Condition A — No Image Uploaded (`skin_image_uploaded == false`)**:
     - If the user asks a general question (store hours, shipping, returns, general ingredient questions), answer it directly without delegating.
     - If the user asks for a personalized skin analysis, diagnosis, or routine for their skin concern (e.g., redness, dryness, acne) **without uploading an image**, you **MUST NOT** delegate to the ADK agent yet. Instead, call `{@TOOL: request_skin_image_upload}` and ask the customer to upload a photo of their skin concern or skincare product so our clinical specialist can analyze it.
   - **Condition B — Skincare Image Uploaded (`skin_image_uploaded == true`)**:
     - As soon as the customer uploads an image of their skin or skincare product, the delegation condition is met!
     - Immediately call `{@TOOL: skincare_adk_agent_runtime_tool}` (passing the user's question/symptoms in the `task` parameter) to delegate visual analysis to the **Google ADK Skincare Specialist Agent on Vertex AI Agent Runtime**, which invokes the **Cloud Run Nano Banana (`gemini-3.1-flash-image`) API**.
     - Present the returned clinical visual findings (`analysis_id`, `model_used`, visible markers, active ingredients, and tailored Lumiere product SKUs + AM/PM routine) clearly to the customer.
</constraints>

<taskflow>
  <subtask name="GeneralAndPolicyInquiries">
    <step name="HandleStoreOrGeneralQuestion">
      <trigger>The user asks about shipping, returns, store locations, or general skincare concepts without requesting a personal visual diagnosis.</trigger>
      <action>Answer directly: Lumiere offers complimentary 2-day cold-chain shipping on orders over $50 and a 60-day clinical satisfaction guarantee.</action>
    </step>
  </subtask>
  <subtask name="ConditionalSkincareImageDelegation">
    <step name="GateCheckMissingImage">
      <trigger>The user asks to analyze their skin or recommend a routine for their skin symptoms, AND no image has been uploaded yet (`skin_image_uploaded == false`).</trigger>
      <action>
        1. Call `{@TOOL: request_skin_image_upload}` with `reason="Customer requested personal skin analysis without uploading an image"`.
        2. Politely inform the customer that visual analysis requires a photo and ask them to upload a clear image of their skin concern or product.
      </action>
    </step>
    <step name="DelegateUploadedImageToAdkAgent">
      <trigger>The user uploads a skin or skincare image (`skin_image_uploaded == true`).</trigger>
      <action>
        1. Immediately call `{@TOOL: skincare_adk_agent_runtime_tool}` with `task` set to the customer's skincare question and goals.
        2. Relay the complete Nano Banana visual assessment (`analysis_id`, detected visual markers, recommended active ingredients, Lumiere product SKUs, and AM/PM protocol) returned by the ADK specialist agent.
      </action>
    </step>
  </subtask>
</taskflow>
"""


def setup_gecx_blueprint_app(
    project_id: str,
    project_number: str,
    region: str,
    reasoning_engine_id: str,
    ces_location: str = "us",
    account: str | None = None,
) -> dict[str, Any]:
    """Creates or updates the GECX Application, Tools, Root Agent, Version, and Deployment."""
    headers = get_auth_headers(account=account)
    ces_base = f"https://ces.googleapis.com/v1beta/projects/{project_id}/locations/{ces_location}"
    reasoning_engine_a2a_url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project_number}/locations/{region}/"
        f"reasoningEngines/{reasoning_engine_id}/a2a/v1"
    )

    r_list = requests.get(f"{ces_base}/apps", headers=headers, timeout=20)
    r_list.raise_for_status()
    apps = r_list.json().get("apps", [])
    target_display_name = "Lumiere Clinical Skincare Concierge (GECX + ADK + Nano Banana)"
    app_obj = next((a for a in apps if a.get("displayName") == target_display_name), None)

    variable_declarations = [
        {
            "name": "skin_image_uploaded",
            "description": "True when the customer has uploaded a skin or skincare image in the session (enables ADK delegation).",
            "schema": {"type": "BOOLEAN", "default": False},
        },
        {
            "name": "diagnostic_consent_granted",
            "description": "True when the customer consents to AI cosmetic skin analysis.",
            "schema": {"type": "BOOLEAN", "default": True},
        },
        {
            "name": "delegation_gate_status",
            "description": "Tracks the current state of the GECX conditional delegation gate.",
            "schema": {"type": "STRING", "default": "AWAITING_IMAGE"},
        },
    ]

    if app_obj is None:
        print(f"Creating new GECX App in {ces_location}: {target_display_name}...")
        create_payload = {
            "displayName": target_display_name,
            "description": (
                "Customer Blueprint demonstrating Gemini Enterprise for Customer Experience (GECX) "
                "conditionally delegating uploaded skincare images to a Google ADK Agent on "
                "Vertex AI Agent Runtime, which invokes a Cloud Run Nano Banana "
                "(gemini-3.1-flash-image) Skincare Analysis API."
            ),
            "pinned": True,
            "modelSettings": {"model": "gemini-3.0-flash-001"},
            "variableDeclarations": variable_declarations,
            "globalInstruction": (
                "SESSION STATE: skin_image_uploaded={skin_image_uploaded}, "
                "diagnostic_consent_granted={diagnostic_consent_granted}, "
                "delegation_gate_status={delegation_gate_status}"
            ),
            "loggingSettings": {
                "cloudLoggingSettings": {"enableCloudLogging": True},
                "conversationLoggingSettings": {"retentionWindow": "31536000s"},
            },
            "timeZoneSettings": {"timeZone": "America/Los_Angeles"},
        }
        r_create = requests.post(f"{ces_base}/apps", headers=headers, json=create_payload, timeout=30)
        r_create.raise_for_status()
        op_data = r_create.json()
        app_obj = (
            wait_for_ces_operation(op_data["name"], headers)
            if "operations/" in op_data.get("name", "")
            else op_data
        )

    app_name = app_obj["name"]
    app_id = app_name.split("/")[-1]
    print(f"GECX App Resource ({ces_location}): {app_name}")

    r_tools = requests.get(
        f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, timeout=15
    )
    r_tools.raise_for_status()
    existing_tools = {t.get("displayName"): t for t in r_tools.json().get("tools", [])}

    # 1. Create or update `request_skin_image_upload` PythonFunction tool
    if "request_skin_image_upload" not in existing_tools:
        r_t1 = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/tools",
            headers=headers,
            json={
                "executionType": "SYNCHRONOUS",
                "pythonFunction": {
                    "name": "request_skin_image_upload",
                    "description": (
                        "Prompts the customer to upload a clear photo of their skin concern or "
                        "skincare product when they request skin analysis without providing an image."
                    ),
                    "pythonCode": REQUEST_IMAGE_UPLOAD_TOOL_CODE,
                },
            },
            timeout=20,
        )
        r_t1.raise_for_status()
        existing_tools["request_skin_image_upload"] = r_t1.json()
    tool_req_img_name = existing_tools["request_skin_image_upload"]["name"]

    # 2. Create or update `skincare_adk_agent_runtime_tool` (RemoteAgentTool -> Vertex AI Agent Runtime)
    re_tool_payload = {
        "executionType": "SYNCHRONOUS",
        "timeout": "60s",
        "remoteAgentTool": {
            "name": "skincare_adk_agent_runtime_tool",
            "description": (
                "Delegates visual skincare image analysis to the Google ADK Skincare Specialist Agent "
                "running on Vertex AI Agent Runtime (reasoningEngines), which invokes the Cloud Run "
                "Nano Banana (gemini-3.1-flash-image) API. Call ONLY when skin_image_uploaded is True."
            ),
            "agentCard": {
                "name": "skincare_specialist_adk_agent_runtime",
                "description": (
                    f"Google ADK Skincare Specialist Agent deployed on Vertex AI Agent Runtime "
                    f"(reasoningEngines/{reasoning_engine_id}) connected to Cloud Run Nano Banana API."
                ),
                "version": "1.0.0",
                "supportedInterfaces": [
                    {
                        "url": reasoning_engine_a2a_url,
                        "protocolBinding": "HTTP+JSON",
                        "protocolVersion": "1.0",
                    }
                ],
                "skills": [
                    {
                        "id": "skincare_visual_diagnosis",
                        "name": "Skincare Image Analysis via Nano Banana on Cloud Run",
                        "description": (
                            "Analyzes uploaded skin or skincare product photos using Nano Banana "
                            "on Cloud Run and recommends a tailored Lumiere skincare regimen."
                        ),
                        "tags": ["skincare", "dermatology", "nano-banana", "vision", "adk"],
                    }
                ],
            },
        },
    }
    if "skincare_adk_agent_runtime_tool" not in existing_tools:
        r_t2 = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/tools",
            headers=headers,
            json=re_tool_payload,
            timeout=20,
        )
        r_t2.raise_for_status()
        existing_tools["skincare_adk_agent_runtime_tool"] = r_t2.json()
    else:
        t2_name = existing_tools["skincare_adk_agent_runtime_tool"]["name"]
        r_t2 = requests.patch(
            f"https://ces.googleapis.com/v1beta/{t2_name}",
            headers=headers,
            json=re_tool_payload,
            timeout=20,
        )
        r_t2.raise_for_status()
        existing_tools["skincare_adk_agent_runtime_tool"] = r_t2.json()
    tool_re_name = existing_tools["skincare_adk_agent_runtime_tool"]["name"]

    # 3. Create or update Root Agent (`lumiere_concierge_root_agent`)
    r_agents = requests.get(
        f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, timeout=15
    )
    r_agents.raise_for_status()
    agents_list = r_agents.json().get("agents", [])
    root_agent_obj = next(
        (a for a in agents_list if a.get("displayName") == "lumiere_concierge_root_agent"),
        agents_list[0] if agents_list else None,
    )

    agent_payload = {
        "displayName": "lumiere_concierge_root_agent",
        "description": (
            "Primary GECX Concierge Agent that enforces conditional delegation (`skin_image_uploaded`) "
            "before invoking the Google ADK Skincare Specialist Agent on Vertex AI Agent Runtime."
        ),
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": ROOT_AGENT_INSTRUCTION,
        "tools": [tool_req_img_name, tool_re_name],
        "beforeModelCallbacks": [
            {
                "description": "Detects when a customer uploads a skincare image (inline_data) and sets skin_image_uploaded=True.",
                "pythonCode": BEFORE_MODEL_CALLBACK_CODE,
            }
        ],
        "beforeToolCallbacks": [
            {
                "description": "Deterministic condition gate that blocks ADK delegation if skin_image_uploaded is False.",
                "pythonCode": BEFORE_TOOL_CALLBACK_CODE,
            }
        ],
    }

    if root_agent_obj is None:
        r_ag = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/agents",
            headers=headers,
            json=agent_payload,
            timeout=20,
        )
        r_ag.raise_for_status()
        root_agent_obj = r_ag.json()
    else:
        ag_name = root_agent_obj["name"]
        r_ag = requests.patch(
            f"https://ces.googleapis.com/v1beta/{ag_name}",
            headers=headers,
            json=agent_payload,
            timeout=20,
        )
        r_ag.raise_for_status()
        root_agent_obj = r_ag.json()

    root_agent_name = root_agent_obj["name"]

    # 4. Patch App to set rootAgent, pinned=True, description, and variableDeclarations
    r_patch_app = requests.patch(
        f"https://ces.googleapis.com/v1beta/{app_name}?updateMask=rootAgent,variableDeclarations,globalInstruction,modelSettings,pinned,description",
        headers=headers,
        json={
            "name": app_name,
            "pinned": True,
            "description": (
                "Customer Blueprint demonstrating Gemini Enterprise for Customer Experience (GECX) "
                "conditionally delegating uploaded skincare images to a Google ADK Agent on "
                "Vertex AI Agent Runtime, which invokes a Cloud Run Nano Banana "
                "(gemini-3.1-flash-image) Skincare Analysis API."
            ),
            "rootAgent": root_agent_name,
            "variableDeclarations": variable_declarations,
            "modelSettings": {"model": "gemini-3.0-flash-001"},
            "globalInstruction": (
                "SESSION STATE: skin_image_uploaded={skin_image_uploaded}, "
                "diagnostic_consent_granted={diagnostic_consent_granted}, "
                "delegation_gate_status={delegation_gate_status}"
            ),
        },
        timeout=20,
    )
    r_patch_app.raise_for_status()

    # 5. Create an AppVersion if none exists so it shows a published version in CX Agent Studio
    r_vers = requests.get(
        f"https://ces.googleapis.com/v1beta/{app_name}/versions", headers=headers, timeout=15
    )
    versions = r_vers.json().get("appVersions", []) if r_vers.status_code == 200 else []
    if not versions:
        r_ver_create = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/versions",
            headers=headers,
            json={
                "displayName": "v1.0 - GECX + ADK + Nano Banana Blueprint",
                "description": "Published v1.0 release with conditional image delegation gate to ADK Agent Runtime + Cloud Run Nano Banana API.",
            },
            timeout=20,
        )
        if r_ver_create.status_code == 200:
            versions = [r_ver_create.json()]
    version_name = versions[0].get("name") if versions else None

    # 6. Create a Deployment (Channel Profile) if none exists
    r_deps = requests.get(
        f"https://ces.googleapis.com/v1beta/{app_name}/deployments", headers=headers, timeout=15
    )
    deployments = r_deps.json().get("deployments", []) if r_deps.status_code == 200 else []
    if not deployments and version_name:
        r_dep_create = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/deployments",
            headers=headers,
            json={
                "displayName": "Lumiere Web & API Live Deployment",
                "appVersion": version_name,
                "channelProfile": {
                    "channelType": "WEB_UI",
                },
            },
            timeout=20,
        )
        if r_dep_create.status_code == 200:
            deployments = [r_dep_create.json()]
    deployment_name = deployments[0].get("name") if deployments else None

    summary = {
        "location": ces_location,
        "gecx_app_id": app_id,
        "gecx_app_name": app_name,
        "gecx_console_url": f"https://ces.cloud.google.com/{app_name}",
        "gecx_root_agent": root_agent_name,
        "gecx_app_version": version_name,
        "gecx_deployment": deployment_name,
        "request_image_tool": tool_req_img_name,
        "adk_agent_runtime_tool": tool_re_name,
        "reasoning_engine_a2a_url": reasoning_engine_a2a_url,
    }
    print(f"GECX App Setup Complete ({ces_location}):\n", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    load_dotenv_file()
    parser = argparse.ArgumentParser(description="Provision GECX App and connect to ADK Agent Runtime.")
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GCP_PROJECT_ID", ""),
        help="Target GCP Project ID",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_REGION", "europe-west1"),
        help="Vertex AI Agent Runtime region",
    )
    parser.add_argument(
        "--reasoning-engine-id",
        default=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", ""),
        help="Numeric ReasoningEngine ID on Vertex AI Agent Runtime",
    )
    parser.add_argument(
        "--ces-location",
        default=os.environ.get("CES_LOCATION", "us"),
        help="GECX location ('us' or 'eu', or 'both')",
    )
    parser.add_argument("--account", default=None, help="Optional gcloud account email")
    args = parser.parse_args()

    if not args.project_id or args.project_id == "YOUR_GCP_PROJECT_ID":
        raise SystemExit("Error: --project-id (or GCP_PROJECT_ID in .env) is required.")
    if not args.reasoning_engine_id:
        raise SystemExit(
            "Error: --reasoning-engine-id (or GOOGLE_CLOUD_AGENT_ENGINE_ID in .env) is required."
        )

    project_number = get_project_number(args.project_id, account=args.account)
    locations = ["us", "eu"] if args.ces_location.lower() == "both" else [args.ces_location]
    for loc in locations:
        setup_gecx_blueprint_app(
            project_id=args.project_id,
            project_number=project_number,
            region=args.region,
            reasoning_engine_id=args.reasoning_engine_id,
            ces_location=loc,
            account=args.account,
        )


if __name__ == "__main__":
    main()
