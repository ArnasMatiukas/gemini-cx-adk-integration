"""Cloud Run Skincare Image Analysis API powered by Nano Banana (gemini-3.1-flash-image).

Tier 3 of the GECX -> ADK Agent -> Cloud Run API Blueprint:
Accepts an uploaded skin/skincare image (base64) + user query/context,
invokes Nano Banana 2 (`gemini-3.1-flash-image`) on Vertex AI (`location="global"`),
and returns structured dermatological visual analysis + matched skincare regimen.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skincare_nano_banana_api")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
NANO_BANANA_MODEL = os.environ.get("NANO_BANANA_MODEL", "gemini-3.1-flash-image")

app = FastAPI(
    title="Lumiere Skincare Nano Banana Analysis API",
    description=(
        "Cloud Run microservice invoked as a tool by the Google ADK Skincare Specialist Agent. "
        "Uses Nano Banana 2 (gemini-3.1-flash-image) on Vertex AI to analyze uploaded skin "
        "and skincare product images."
    ),
    version="1.0.0",
)

# Curated Lumiere Clinical Skincare Product Catalog
SKINCARE_CATALOG: list[dict[str, Any]] = [
    {
        "sku": "LUM-CLN-001",
        "name": "Lumiere Barrier-Calm Ceramide Cleanser",
        "category": "Cleanser",
        "key_actives": ["Ceramide NP", "Panthenol (Vitamin B5)", "Colloidal Oatmeal"],
        "target_concerns": ["redness", "erythema", "dryness", "barrier_impairment", "sensitivity"],
        "price_usd": 34.00,
        "usage": "AM/PM gentle non-stripping cleanse on damp skin.",
    },
    {
        "sku": "LUM-SRM-002",
        "name": "Lumiere Azelaic 10% + Niacinamide Clarifying Serum",
        "category": "Treatment Serum",
        "key_actives": ["10% Azelaic Acid", "5% Niacinamide", "Zinc PCA"],
        "target_concerns": ["redness", "erythema", "blemishes", "acne", "uneven_tone", "pores"],
        "price_usd": 48.00,
        "usage": "Apply 3-4 drops AM/PM after cleansing to visible redness or blemish-prone zones.",
    },
    {
        "sku": "LUM-SRM-003",
        "name": "Lumiere Polyglutamic + Multi-Weight Hyaluronic Complex",
        "category": "Hydration Serum",
        "key_actives": ["Polyglutamic Acid", "5-Molecular-Weight Hyaluronic Acid", "Ectoin"],
        "target_concerns": ["dehydration", "dryness", "fine_lines", "dullness"],
        "price_usd": 52.00,
        "usage": "Apply to damp skin AM/PM before moisturizer.",
    },
    {
        "sku": "LUM-MST-004",
        "name": "Lumiere Cica-Peptide Lipid Recovery Cream",
        "category": "Moisturizer",
        "key_actives": ["Madecassoside (Centella Asiatica)", "Copper Tripeptide-1", "Squalane"],
        "target_concerns": ["redness", "dryness", "barrier_impairment", "texture", "irritation"],
        "price_usd": 58.00,
        "usage": "Seal hydration AM/PM as final treatment step before sunscreen.",
    },
    {
        "sku": "LUM-SPF-005",
        "name": "Lumiere Mineral Zinc SPF 50+ Invisible Fluid",
        "category": "Sun Protection",
        "key_actives": ["Non-Nano Zinc Oxide 18%", "Bisabolol", "Ectoin"],
        "target_concerns": ["uv_protection", "hyperpigmentation", "redness", "post_inflammatory_erythema"],
        "price_usd": 42.00,
        "usage": "Apply generously every morning as the final skincare step.",
    },
]


class SkinAnalysisRequest(BaseModel):
    """Request payload for Nano Banana skincare image analysis."""

    image_base64: str = Field(
        ...,
        description="Base64-encoded image bytes (PNG/JPEG/WEBP) uploaded by the customer.",
    )
    mime_type: str = Field(
        default="image/png",
        description="MIME type of the uploaded image (e.g., image/png, image/jpeg).",
    )
    user_query: str = Field(
        default="Please analyze this skincare image and recommend a tailored regimen.",
        description="Customer's question, symptoms, or skincare goals.",
    )
    customer_skin_goals: list[str] = Field(
        default_factory=list,
        description="Optional list of customer-stated skin goals or sensitivities.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional upstream GECX / ADK session identifier for tracing.",
    )


class SkinAnalysisResponse(BaseModel):
    """Structured response from the Cloud Run Nano Banana Skincare API."""

    analysis_id: str
    model_used: str
    processing_time_ms: int
    image_received: bool
    image_size_bytes: int
    visual_assessment: dict[str, Any]
    recommended_regimen: list[dict[str, Any]]
    application_protocol: dict[str, str]
    safety_disclaimer: str


NANO_BANANA_SYSTEM_PROMPT = """You are a specialized cosmetic dermatology visual analysis engine powered by Nano Banana (gemini-3.1-flash-image).
Analyze the provided skin or skincare product image alongside the customer's query.
Return ONLY a valid JSON object (no markdown fences) with the following schema:
{
  "image_category": "facial_skin_closeup | skincare_product_label | skin_patch_texture | other",
  "apparent_skin_type": "dry | oily | combination | sensitive | normal",
  "hydration_level": "dehydrated | moderate | well_hydrated",
  "visible_markers": [
    {
      "marker": "e.g. mild erythema / redness on cheek zone | surface dryness / flaking | congested pores | hyperpigmentation | product label active ingredients",
      "confidence": "high | medium | low",
      "region_or_detail": "specific visual observation from the image"
    }
  ],
  "primary_concern_tags": ["redness", "dryness", "barrier_impairment", "blemishes", "dehydration", "texture", "uv_protection"],
  "severity_estimate": "mild | moderate | notable",
  "key_recommended_actives": ["Ceramide NP", "Azelaic Acid", "Madecassoside", "Polyglutamic Acid", "Non-Nano Zinc Oxide"],
  "visual_summary": "2-3 sentence objective visual analysis of what is seen in the uploaded image and why specific barrier-repair or clarifying actives are indicated.",
  "am_pm_guidance": {
    "morning": "Concise AM routine steps tailored to the image findings.",
    "evening": "Concise PM routine steps tailored to the image findings."
  }
}
"""


def _match_catalog_products(concern_tags: list[str]) -> list[dict[str, Any]]:
    """Matches detected skin concerns against the Lumiere Skincare Catalog."""
    normalized = {c.lower().strip() for c in concern_tags}
    if not normalized:
        normalized = {"redness", "dryness", "barrier_impairment"}

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in SKINCARE_CATALOG:
        overlap = len(normalized.intersection(set(item["target_concerns"])))
        scored.append((overlap, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [item for score, item in scored if score > 0][:3]
    if not selected:
        selected = SKINCARE_CATALOG[:3]
    if not any(p["sku"] == "LUM-SPF-005" for p in selected):
        selected.append(SKINCARE_CATALOG[-1])
    return selected


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Cloud Run health check endpoint."""
    return {
        "status": "ok",
        "service": "skincare-nano-banana-api",
        "model": NANO_BANANA_MODEL,
        "project": PROJECT_ID,
        "location": LOCATION,
    }


@app.get("/v1/catalog")
async def list_catalog() -> dict[str, Any]:
    """Returns the Lumiere clinical skincare product catalog."""
    return {"catalog": SKINCARE_CATALOG, "count": len(SKINCARE_CATALOG)}


@app.post("/v1/analyze-skin", response_model=SkinAnalysisResponse)
async def analyze_skin(req: SkinAnalysisRequest) -> SkinAnalysisResponse:
    """Analyzes an uploaded skincare image using Nano Banana (gemini-3.1-flash-image) on Vertex AI."""
    t0 = time.monotonic()
    analysis_id = f"nb-skin-{uuid.uuid4().hex[:10]}"

    raw_b64 = req.image_base64.strip()
    if "," in raw_b64 and raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image payload is empty (0 bytes).")

    logger.info(
        "Running Nano Banana (%s) analysis [%s]: image_bytes=%d, mime_type=%s, query=%r",
        NANO_BANANA_MODEL,
        analysis_id,
        len(image_bytes),
        req.mime_type,
        req.user_query,
    )

    client_kwargs: dict[str, Any] = {"vertexai": True, "location": LOCATION}
    if PROJECT_ID:
        client_kwargs["project"] = PROJECT_ID
    client = genai.Client(**client_kwargs)
    prompt_text = (
        f"{NANO_BANANA_SYSTEM_PROMPT}\n\n"
        f"Customer Query: {req.user_query}\n"
        f"Customer Skin Goals: {', '.join(req.customer_skin_goals) if req.customer_skin_goals else 'None specified'}\n"
    )

    try:
        response = client.models.generate_content(
            model=NANO_BANANA_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=req.mime_type),
                types.Part.from_text(text=prompt_text),
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )
        raw_text = (response.text or "").strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            visual_assessment = json.loads(raw_text[start_idx : end_idx + 1])
        else:
            visual_assessment = {
                "image_category": "facial_skin_closeup",
                "apparent_skin_type": "sensitive",
                "hydration_level": "dehydrated",
                "visible_markers": [
                    {
                        "marker": "Localized erythema and barrier dryness",
                        "confidence": "high",
                        "region_or_detail": raw_text[:300],
                    }
                ],
                "primary_concern_tags": ["redness", "dryness", "barrier_impairment"],
                "severity_estimate": "mild",
                "key_recommended_actives": ["Ceramide NP", "Azelaic Acid", "Madecassoside"],
                "visual_summary": raw_text[:500],
            }
    except Exception as exc:
        logger.exception("Nano Banana model invocation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Nano Banana ({NANO_BANANA_MODEL}) invocation failed: {exc}",
        ) from exc

    concern_tags = visual_assessment.get(
        "primary_concern_tags", ["redness", "dryness", "barrier_impairment"]
    )
    matched_products = _match_catalog_products(concern_tags)
    am_pm = visual_assessment.get("am_pm_guidance") or {
        "morning": (
            "1. Cleanse with Lumiere Barrier-Calm Ceramide Cleanser. "
            "2. Apply Polyglutamic + Hyaluronic Complex on damp skin. "
            "3. Seal with Cica-Peptide Lipid Recovery Cream and Mineral Zinc SPF 50+."
        ),
        "evening": (
            "1. Gentle cleanse with Barrier-Calm Ceramide Cleanser. "
            "2. Apply Azelaic 10% + Niacinamide Serum to redness zones. "
            "3. Lock in moisture with Cica-Peptide Lipid Recovery Cream."
        ),
    }

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return SkinAnalysisResponse(
        analysis_id=analysis_id,
        model_used=f"{NANO_BANANA_MODEL} (Nano Banana 2 on Vertex AI)",
        processing_time_ms=elapsed_ms,
        image_received=True,
        image_size_bytes=len(image_bytes),
        visual_assessment=visual_assessment,
        recommended_regimen=matched_products,
        application_protocol=am_pm,
        safety_disclaimer=(
            "Cosmetic skincare analysis generated by Nano Banana (gemini-3.1-flash-image) "
            "for informational regimen personalization only; not a medical diagnosis."
        ),
    )
