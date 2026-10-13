"""Tier 2: Google ADK Skincare Specialist Agent (`root_agent`) and Cloud Run Tool."""

from __future__ import annotations

import contextvars
import os
from typing import Any

import google.auth.transport.requests
import google.oauth2.id_token
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types
import requests

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
ADK_MODEL = os.environ.get("ADK_MODEL_NAME", "gemini-3.8-flash")
SKINCARE_API_URL = os.environ.get("SKINCARE_API_URL", "http://localhost:8080")

# Force GenAI SDK inside ADK to use Vertex AI global endpoint
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
if PROJECT_ID:
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

CURRENT_IMAGE_B64: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_image_b64", default=None
)
CURRENT_IMAGE_MIME: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_image_mime", default="image/png"
)


def _fetch_id_token(audience: str) -> str | None:
    """Mints a Google-signed OIDC ID token for authenticated Cloud Run invocation."""
    try:
        return google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), audience
        )
    except Exception:
        try:
            r = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity",
                params={"audience": audience},
                headers={"Metadata-Flavor": "Google"},
                timeout=5,
            )
            return r.text.strip() if r.status_code == 200 else None
        except Exception:
            return None


def analyze_skincare_image_on_cloud_run(
    user_query: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Calls the Tier 3 Cloud Run Skincare Vision API (`gemini-3.8-flash`) to analyze the uploaded image."""
    image_b64 = CURRENT_IMAGE_B64.get()
    mime_type = CURRENT_IMAGE_MIME.get()
    if not image_b64:
        return {"status": "error", "message": "No skincare image uploaded in current session."}

    api_base = SKINCARE_API_URL.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if token := _fetch_id_token(api_base):
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.post(
        f"{api_base}/v1/analyze-skin",
        headers=headers,
        json={"image_base64": image_b64, "mime_type": mime_type, "user_query": user_query},
        timeout=90,
    )
    if resp.status_code != 200:
        return {"status": "error", "http_status": resp.status_code, "detail": resp.text[:400]}

    result: dict[str, Any] = resp.json()
    if tool_context is not None and tool_context.state is not None:
        tool_context.state["last_skincare_analysis"] = result
    return result


root_agent = Agent(
    name="skincare_specialist_adk_agent",
    model=ADK_MODEL,
    description="Clinical Skincare Specialist ADK Agent that delegates image analysis to the Cloud Run Vision API.",
    instruction=(
        "You are the Lumiere Clinical Skincare Specialist ADK Agent (`gemini-3.8-flash`), invoked via A2A "
        "from Gemini Enterprise for Customer Experience (GECX).\n"
        "1. Always call `analyze_skincare_image_on_cloud_run` first to analyze the customer's uploaded skin/product photo.\n"
        "2. Present the returned `analysis_id`, `model_used`, visual assessment markers, recommended active ingredients, "
        "matched Lumiere SKUs (`recommended_regimen`), and AM/PM routine concisely and objectively."
    ),
    tools=[analyze_skincare_image_on_cloud_run],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)
