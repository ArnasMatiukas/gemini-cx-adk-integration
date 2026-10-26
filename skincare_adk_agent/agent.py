"""Tier 2: Google ADK Clinical Skin Diagnosis Agent (`root_agent`) on Vertex AI Agent Runtime.

This ADK agent is strictly responsible for clinical visual skin analysis using `gemini-3.8-flash`
on Vertex AI with native Structured Outputs. It identifies what is wrong with the skin
(`analysis_id`, `apparent_skin_type`, `hydration_level`, `visible_markers`, `primary_concern_tags`,
`key_recommended_actives`, and `visual_summary`) and NEVER recommends actual commercial products or SKUs.
Product recommendations are handled by a separate GECX sub-agent (`lumiere_product_recommendation_agent`).
"""

from __future__ import annotations

import base64
import contextvars
import os
import time
import uuid
from typing import Any

from google import genai
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
ADK_MODEL = os.environ.get("ADK_MODEL_NAME", "gemini-3.8-flash")

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


class VisualAssessment(BaseModel):
    """Clinical visual skin assessment returned by Gemini 3.8 Flash (no commercial product SKUs)."""

    apparent_skin_type: str = Field(description="dry, oily, combination, sensitive, or normal")
    hydration_level: str = Field(description="dehydrated, moderate, or well_hydrated")
    visible_markers: list[str] = Field(
        description="Specific visual dermatology markers observed in the image (what is wrong with the skin)"
    )
    primary_concern_tags: list[str] = Field(
        description="Subset of: redness, dryness, barrier_impairment, blemishes, pores, uneven_texture, uv_protection"
    )
    key_recommended_actives: list[str] = Field(
        description="Generic dermatological active ingredients suited for the observed markers (no brand or product names)"
    )
    visual_summary: str = Field(
        description="Concise 2-sentence objective clinical summary of what is wrong with the skin in the image"
    )


def analyze_skin_condition_image(
    user_query: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Analyzes the uploaded skin image directly with `gemini-3.8-flash` on Vertex AI to identify what is wrong with the skin."""
    t0 = time.monotonic()
    image_b64 = CURRENT_IMAGE_B64.get()
    mime_type = CURRENT_IMAGE_MIME.get()
    if not image_b64:
        return {"status": "error", "message": "No skincare image uploaded in current session."}

    raw_b64 = image_b64.strip().split(",")[-1]
    try:
        img_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        return {"status": "error", "message": f"Invalid base64 image: {exc}"}

    client = genai.Client(vertexai=True, project=PROJECT_ID or None, location="global")
    prompt = (
        "You are a clinical dermatology vision analyzer. Inspect the uploaded skin image alongside "
        f"the user's reported symptoms ({user_query!r}) and return a structured clinical visual assessment "
        "identifying what is wrong with the skin (apparent skin type, hydration level, visible markers, "
        "primary concern tags, generic target active ingredients, and objective visual summary). "
        "Do NOT recommend or mention any commercial brand names, product names, or SKUs."
    )

    resp = client.models.generate_content(
        model=ADK_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=VisualAssessment,
        ),
    )
    assessment = VisualAssessment.model_validate_json(resp.text or "{}")

    result: dict[str, Any] = {
        "status": "ok",
        "analysis_id": f"skin-{uuid.uuid4().hex[:8]}",
        "model_used": ADK_MODEL,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "apparent_skin_type": assessment.apparent_skin_type,
        "hydration_level": assessment.hydration_level,
        "visible_markers": assessment.visible_markers,
        "primary_concern_tags": assessment.primary_concern_tags,
        "key_recommended_actives": assessment.key_recommended_actives,
        "visual_summary": assessment.visual_summary,
    }
    if tool_context is not None and tool_context.state is not None:
        tool_context.state["last_skincare_analysis"] = result
    return result


root_agent = Agent(
    name="skincare_specialist_adk_agent",
    model=ADK_MODEL,
    description=(
        "Clinical Dermatology Visual Diagnosis ADK Agent that analyzes uploaded skin images "
        "and identifies what is wrong with the skin without recommending actual products."
    ),
    instruction=(
        "You are the Clinical Dermatology Visual Diagnosis ADK Agent (`gemini-3.8-flash`), invoked via A2A "
        "from Gemini Enterprise for Customer Experience (GECX).\n"
        "1. Always call `analyze_skin_condition_image` first to analyze the customer's uploaded skin photo "
        "and identify what is wrong with the skin.\n"
        "2. Return ONLY the objective clinical visual diagnosis from the tool: `analysis_id`, `model_used`, "
        "`apparent_skin_type`, `hydration_level`, `visible_markers`, `primary_concern_tags`, "
        "`key_recommended_actives`, and `visual_summary`.\n"
        "3. STRICT BOUNDARY: Do NOT recommend or mention any commercial product names, SKUs, prices, or shopping "
        "regimens. You only identify what is wrong with the skin; product recommendations are handled by a "
        "separate GECX sub-agent (`lumiere_product_recommendation_agent`)."
    ),
    tools=[analyze_skin_condition_image],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)
