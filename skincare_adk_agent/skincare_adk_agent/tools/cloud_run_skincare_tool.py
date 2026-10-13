"""ADK FunctionTool that invokes the Cloud Run Skincare Nano Banana API (`skincare-nano-banana-api`)."""

from __future__ import annotations

import contextvars
import logging
from typing import Any

import google.auth.transport.requests
import google.oauth2.id_token
from google.adk.tools import ToolContext
import requests

from skincare_adk_agent.config import CONFIG

logger = logging.getLogger("skincare_adk_agent.tools")

# ContextVars hold the active turn's uploaded skincare image extracted from A2A / ReasoningEngine request
CURRENT_IMAGE_B64: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_image_b64", default=None
)
CURRENT_IMAGE_MIME: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_image_mime", default="image/png"
)


def _fetch_cloud_run_id_token(audience: str) -> str | None:
    """Mints a Google-signed OIDC ID token to authenticate against the Cloud Run API."""
    try:
        auth_req = google.auth.transport.requests.Request()
        return google.oauth2.id_token.fetch_id_token(auth_req, audience)
    except Exception as exc:
        logger.warning("Could not fetch OIDC ID token via google.oauth2.id_token: %s", exc)
        try:
            r = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity",
                params={"audience": audience},
                headers={"Metadata-Flavor": "Google"},
                timeout=5,
            )
            if r.status_code == 200:
                return r.text.strip()
        except Exception as meta_exc:
            logger.warning("Metadata server ID token fallback failed: %s", meta_exc)
    return None


def analyze_skincare_image_on_cloud_run(
    user_query: str,
    customer_skin_goals: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Calls the Cloud Run Skincare Nano Banana API (`skincare-nano-banana-api`) to analyze the customer's uploaded skincare image.

    Args:
        user_query: The customer's skincare question, symptoms, or request context.
        customer_skin_goals: Comma-separated skincare goals or sensitivities mentioned by the customer.
        tool_context: Automatically injected by Google ADK runtime to access session state.

    Returns:
        Structured dictionary from the Cloud Run Nano Banana (`gemini-3.1-flash-image`) API
        containing `analysis_id`, `model_used`, `visual_assessment`, `recommended_regimen`,
        and `application_protocol`.
    """
    image_b64: str | None = None
    mime_type: str = "image/png"

    if tool_context is not None and tool_context.state is not None:
        image_b64 = tool_context.state.get("uploaded_image_base64")
        mime_type = tool_context.state.get("uploaded_image_mime_type", "image/png")

    if not image_b64:
        image_b64 = CURRENT_IMAGE_B64.get()
        mime_type = CURRENT_IMAGE_MIME.get()

    if not image_b64:
        return {
            "status": "error",
            "error_code": "NO_IMAGE_UPLOADED",
            "message": (
                "No skincare image was provided in the current session. "
                "Please ask the customer to upload a clear photo of their skin concern or skincare product."
            ),
        }

    api_base = CONFIG.skincare_api_url.rstrip("/")
    endpoint = f"{api_base}/v1/analyze-skin"

    headers = {"Content-Type": "application/json"}
    id_token = _fetch_cloud_run_id_token(api_base)
    if id_token:
        headers["Authorization"] = f"Bearer {id_token}"

    goals_list = [g.strip() for g in customer_skin_goals.split(",") if g.strip()]
    payload = {
        "image_base64": image_b64,
        "mime_type": mime_type,
        "user_query": user_query,
        "customer_skin_goals": goals_list,
    }

    logger.info("ADK Tool calling Cloud Run Skincare API: POST %s (query=%r)", endpoint, user_query)
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        return {
            "status": "error",
            "http_status": resp.status_code,
            "detail": resp.text[:500],
        }

    result: dict[str, Any] = resp.json()
    if tool_context is not None and tool_context.state is not None:
        tool_context.state["last_skincare_analysis"] = result
    return result
