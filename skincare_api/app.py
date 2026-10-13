"""Tier 3: Cloud Run Skincare Image Analysis API powered by Gemini 3.8 Flash.

Receives an uploaded skincare image (base64) + user query from the Google ADK Agent,
analyzes visual dermatology markers using `gemini-3.8-flash` with native Structured Outputs,
and returns a matched Lumiere clinical skincare regimen.
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

app = FastAPI(title="Lumiere Skincare Vision API (Gemini 3.8 Flash)", version="1.0.0")

SKINCARE_CATALOG: list[dict[str, Any]] = [
    {
        "sku": "LUM-CLN-001",
        "name": "Lumiere Barrier-Calm Ceramide Cleanser",
        "key_actives": ["Ceramide NP", "Panthenol (B5)", "Colloidal Oatmeal"],
        "concerns": ["redness", "dryness", "barrier_impairment", "sensitivity"],
        "price_usd": 34.00,
    },
    {
        "sku": "LUM-SRM-002",
        "name": "Lumiere Azelaic 10% + Niacinamide Clarifying Serum",
        "key_actives": ["10% Azelaic Acid", "5% Niacinamide", "Zinc PCA"],
        "concerns": ["redness", "blemishes", "acne", "uneven_tone", "pores"],
        "price_usd": 48.00,
    },
    {
        "sku": "LUM-MST-004",
        "name": "Lumiere Cica-Peptide Lipid Recovery Cream",
        "key_actives": ["Madecassoside (Centella)", "Copper Tripeptide-1", "Squalane"],
        "concerns": ["redness", "dryness", "barrier_impairment", "texture"],
        "price_usd": 58.00,
    },
    {
        "sku": "LUM-SPF-005",
        "name": "Lumiere Mineral Zinc SPF 50+ Invisible Fluid",
        "key_actives": ["Non-Nano Zinc Oxide 18%", "Bisabolol", "Ectoin"],
        "concerns": ["uv_protection", "redness", "hyperpigmentation"],
        "price_usd": 42.00,
    },
]


class VisualAssessment(BaseModel):
    """Structured output schema returned directly by Gemini 3.8 Flash."""

    apparent_skin_type: str = Field(description="dry, oily, combination, sensitive, or normal")
    hydration_level: str = Field(description="dehydrated, moderate, or well_hydrated")
    visible_markers: list[str] = Field(description="Specific visual markers observed in the image")
    primary_concern_tags: list[str] = Field(
        description="Subset of: redness, dryness, barrier_impairment, blemishes, texture, uv_protection"
    )
    key_recommended_actives: list[str] = Field(description="Recommended active ingredients")
    visual_summary: str = Field(description="Concise 2-sentence objective summary of the image findings")
    morning_routine: str = Field(description="Concise AM application steps")
    evening_routine: str = Field(description="Concise PM application steps")


class SkinAnalysisRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/png"
    user_query: str = "Analyze this skincare image and recommend a routine."


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
        "You are a cosmetic dermatology vision engine. Analyze the uploaded skin or skincare "
        f"product image and address the customer's query: {req.user_query}"
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

    tags = {t.lower().strip() for t in assessment.primary_concern_tags}
    matched = [
        item for item in SKINCARE_CATALOG if tags.intersection(item["concerns"])
    ] or SKINCARE_CATALOG[:3]
    if not any(p["sku"] == "LUM-SPF-005" for p in matched):
        matched.append(SKINCARE_CATALOG[-1])

    return {
        "analysis_id": f"skin-{uuid.uuid4().hex[:8]}",
        "model_used": VISION_MODEL,
        "processing_time_ms": int((time.monotonic() - t0) * 1000),
        "visual_assessment": assessment.model_dump(),
        "recommended_regimen": matched,
    }
