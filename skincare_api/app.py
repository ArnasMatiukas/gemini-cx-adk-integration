"""Tier 3: Cloud Run Skincare Image Analysis API powered by Gemini 3.8 Flash.

Receives an uploaded skincare image (base64) + user query from the Google ADK Agent,
analyzes visual dermatology markers using `gemini-3.8-flash` with native Structured Outputs,
and returns ONLY the clinical visual assessment (skin type, hydration, visible markers,
concern tags, and target active ingredients). Product matching and regimen recommendations
are owned exclusively by the Tier 1 GECX `skin_analysis_recommendation_agent`.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
VISION_MODEL = os.environ.get("VISION_MODEL", "gemini-3.8-flash")

app = FastAPI(title="Lumiere Skincare Vision API (Gemini 3.8 Flash)", version="2.0.0")


class VisualAssessment(BaseModel):
    """Clinical visual skin assessment returned by Gemini 3.8 Flash (no product SKUs)."""

    apparent_skin_type: str = Field(description="dry, oily, combination, sensitive, or normal")
    hydration_level: str = Field(description="dehydrated, moderate, or well_hydrated")
    visible_markers: list[str] = Field(description="Specific visual dermatology markers observed in the image")
    primary_concern_tags: list[str] = Field(
        description="Subset of: redness, dryness, barrier_impairment, blemishes, pores, uneven_texture, uv_protection"
    )
    key_recommended_actives: list[str] = Field(
        description="Generic dermatological active ingredients suited for the observed markers (no brand/product names)"
    )
    visual_summary: str = Field(description="Concise 2-sentence objective clinical summary of the image findings")


class SkinAnalysisRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/png"
    user_query: str = "Analyze the visual skin condition in this image."


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "skincare-vision-api", "model": VISION_MODEL}


@app.post("/v1/analyze-skin")
async def analyze_skin(req: SkinAnalysisRequest) -> dict[str, Any]:
    t0 = time.monotonic()
    raw_b64 = req.image_base64.strip().split(",")[-1]
    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    client = genai.Client(vertexai=True, project=PROJECT_ID or None, location=LOCATION)
    prompt = (
        "You are a clinical dermatology computer-vision engine. Analyze the uploaded skin photo "
        "and return ONLY objective visual dermatology markers, concern tags, and generic active "
        "ingredients. Do NOT recommend commercial product names or SKUs.\n"
        f"Customer context: {req.user_query}"
    )

    try:
        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=req.mime_type),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=VisualAssessment,
            ),
        )
        assessment = VisualAssessment.model_validate_json(response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{VISION_MODEL} call failed: {exc}") from exc

    return {
        "analysis_id": f"skin-{uuid.uuid4().hex[:8]}",
        "model_used": VISION_MODEL,
        "processing_time_ms": int((time.monotonic() - t0) * 1000),
        "visual_assessment": assessment.model_dump(),
    }
