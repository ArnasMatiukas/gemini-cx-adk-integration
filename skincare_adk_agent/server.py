"""Tier 2: FastAPI A2A v1.0 Server for Vertex AI Agent Runtime (`reasoningEngines`).

Exposes the A2A v1.0 endpoints required by GECX (`POST /api/a2a/v1/message:send`),
persists sessions via `VertexAiSessionService` (populating the Agent Engine Sessions tab),
and exports OpenTelemetry spans (populating the Agent Engine Traces tab).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import google.auth
import google.auth.transport.requests
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.genai import types

from skincare_adk_agent.agent import (
    ADK_MODEL,
    CURRENT_IMAGE_B64,
    CURRENT_IMAGE_MIME,
    PROJECT_ID,
    SKINCARE_API_URL,
    root_agent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skincare_adk_agent")

AGENT_ENGINE_ID = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID") or os.environ.get("AGENT_ENGINE_ID", "")
AGENT_ENGINE_LOCATION = (
    os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
    or os.environ.get("AGENT_ENGINE_LOCATION", "europe-west1")
)
PROJECT_NUMBER = os.environ.get("GCP_PROJECT_NUMBER", "")


def _setup_telemetry() -> Any:
    """Exports OpenTelemetry traces so spans appear in Vertex AI Agent Engine -> Traces."""
    if not AGENT_ENGINE_ID or not PROJECT_ID:
        return None
    try:
        import opentelemetry.sdk.resources as otel_resource
        import opentelemetry.sdk.trace as otel_trace
        import opentelemetry.sdk.trace.export as otel_export
        import opentelemetry.trace as trace_api
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        resource_id = (
            f"//aiplatform.googleapis.com/projects/{PROJECT_ID}"
            f"/locations/{AGENT_ENGINE_LOCATION}/reasoningEngines/{AGENT_ENGINE_ID}"
        )
        urn_proj = PROJECT_NUMBER or PROJECT_ID
        resource = otel_resource.Resource.create(
            attributes={
                "gcp.project_id": PROJECT_ID,
                "cloud.account.id": PROJECT_ID,
                "cloud.provider": "gcp",
                "cloud.platform": "gcp.agent_engine",
                "service.name": AGENT_ENGINE_ID,
                "cloud.region": AGENT_ENGINE_LOCATION,
                "cloud.resource_id": resource_id,
                "gen_ai.main_agent.id": (
                    f"urn:agent:projects-{urn_proj}:projects:{urn_proj}"
                    f":locations:{AGENT_ENGINE_LOCATION}:aiplatform:reasoningEngines:{AGENT_ENGINE_ID}"
                ),
            }
        )
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        otlp_exporter = OTLPSpanExporter(
            session=google.auth.transport.requests.AuthorizedSession(creds),
            endpoint="https://telemetry.googleapis.com/v1/traces",
            headers={"x-goog-user-project": PROJECT_ID},
        )
        provider = otel_trace.TracerProvider(resource=resource)
        provider.add_span_processor(otel_export.BatchSpanProcessor(otlp_exporter))
        provider.add_span_processor(
            otel_export.BatchSpanProcessor(CloudTraceSpanExporter(project_id=PROJECT_ID))
        )
        trace_api.set_tracer_provider(provider)
        return provider
    except Exception as exc:
        logger.warning("Telemetry setup skipped: %s", exc)
        return None


TRACER_PROVIDER = _setup_telemetry()

if AGENT_ENGINE_ID and PROJECT_ID:
    APP_NAME = AGENT_ENGINE_ID
    SESSION_SERVICE = VertexAiSessionService(
        project=PROJECT_ID, location=AGENT_ENGINE_LOCATION, agent_engine_id=AGENT_ENGINE_ID
    )
else:
    APP_NAME = "skincare_adk_agent"
    SESSION_SERVICE = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=SESSION_SERVICE,
    artifact_service=InMemoryArtifactService(),
    memory_service=InMemoryMemoryService(),
)

CONTEXT_TO_SESSION_ID: dict[str, str] = {}
app = FastAPI(title="Lumiere Skincare Specialist ADK Agent", version="1.0.0")


def extract_a2a_input(payload: dict[str, Any]) -> tuple[str, str | None, str, str]:
    """Extracts `(user_text, image_b64, mime_type, context_id)` from a GECX A2A v1.0 request."""
    msg = payload.get("message") or payload.get("request") or payload
    context_id = msg.get("contextId") or f"ctx-{uuid.uuid4().hex[:8]}"
    text_parts: list[str] = []
    image_b64: str | None = None
    mime_type = "image/png"

    for part in msg.get("parts") or []:
        if isinstance(part, dict):
            if part.get("text"):
                text_parts.append(str(part["text"]))
            if part.get("raw"):
                image_b64 = str(part["raw"])
                mime_type = str(part.get("mediaType") or "image/png")

    # Inspect GECX conversation history (`metadata.gecx_a2a_agent_context`) for uploaded images
    ctx_raw = (msg.get("metadata") or {}).get("gecx_a2a_agent_context")
    if ctx_raw and not image_b64:
        try:
            ctx_obj = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
            for turn in reversed(ctx_obj.get("history") or []):
                for chunk in reversed(turn.get("chunks") or []):
                    img = chunk.get("image") if isinstance(chunk, dict) else None
                    if isinstance(img, dict) and img.get("data"):
                        image_b64 = str(img["data"])
                        mime_type = str(img.get("mimeType") or "image/png")
                        break
                if image_b64:
                    break
        except Exception as exc:
            logger.warning("Failed to parse gecx_a2a_agent_context: %s", exc)

    user_text = "\n".join(text_parts).strip() or "Analyze my uploaded skincare image."
    return user_text, image_b64, mime_type, context_id


@app.get("/")
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "adk_model": ADK_MODEL, "skincare_api_url": SKINCARE_API_URL}


@app.get("/.well-known/agent-card.json")
@app.get("/api/a2a/v1/card")
async def agent_card(request: Request) -> dict[str, Any]:
    return {
        "name": "skincare_specialist_adk_agent",
        "description": "Google ADK Skincare Specialist Agent calling Cloud Run Vision API.",
        "version": "1.0.0",
        "supportedInterfaces": [
            {"url": str(request.base_url).rstrip("/"), "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
        ],
        "skills": [{"id": "skincare_visual_diagnosis", "name": "Skincare Image Analysis"}],
    }


@app.post("/message:send")
@app.post("/api/a2a/v1/message:send")
async def handle_a2a_send(request: Request) -> JSONResponse:
    """Handles A2A v1.0 `message:send` delegation from GECX."""
    payload = await request.json()
    user_text, image_b64, mime_type, context_id = extract_a2a_input(payload)

    user_id = "gecx_customer"
    session_id = CONTEXT_TO_SESSION_ID.get(context_id)
    if not session_id:
        session = await runner.session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            state={"gecx_context_id": context_id, "has_uploaded_image": bool(image_b64)},
        )
        session_id = session.id
        CONTEXT_TO_SESSION_ID[context_id] = session_id

    tok_b64 = CURRENT_IMAGE_B64.set(image_b64)
    tok_mime = CURRENT_IMAGE_MIME.set(mime_type)
    try:
        prompt = (
            f"{user_text}\n[Customer uploaded a {mime_type} skincare image. Call `analyze_skincare_image_on_cloud_run` now.]"
            if image_b64
            else user_text
        )
        chunks: list[str] = []
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                chunks.extend(p.text for p in event.content.parts if getattr(p, "text", None))
        response_text = "\n".join(chunks).strip()
    finally:
        CURRENT_IMAGE_B64.reset(tok_b64)
        CURRENT_IMAGE_MIME.reset(tok_mime)
        if TRACER_PROVIDER is not None:
            try:
                TRACER_PROVIDER.force_flush(timeout_millis=5000)
            except Exception:
                pass

    return JSONResponse(
        {
            "message": {
                "messageId": f"msg-{uuid.uuid4().hex[:10]}",
                "contextId": context_id,
                "role": "ROLE_AGENT",
                "parts": [{"text": response_text}],
            }
        }
    )
