"""A2A v1.0 & Vertex AI Agent Runtime HTTP Server for the Google ADK Skincare Specialist Agent.

Supports BOTH Vertex AI Agent Runtime (`reasoningEngines`) and standalone HTTP execution:
1. Vertex AI Agent Runtime (`reasoningEngines`):
   - `POST /api/a2a/v1/message:send` & `POST /api/a2a/message:send` (A2A Unary)
   - `POST /api/a2a/v1/message:stream` & `POST /api/a2a/message:stream` (A2A Streaming SSE)
   - `POST /api/reasoning_engine` (Native ReasoningEngine `:query`)
   - `GET /api/a2a/v1/card` & `GET /.well-known/agent-card.json`
2. Standard A2A HTTP endpoints:
   - `POST /message:send` & `POST /v1/message:send` (A2A Unary)
   - `POST /message:stream` & `POST /v1/message:stream` (A2A Streaming SSE)
   - `GET /.well-known/agent-card.json` & `GET /health`
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import google.auth
import google.auth.transport.requests
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.genai import types

from skincare_adk_agent.agent import root_agent
from skincare_adk_agent.config import CONFIG
from skincare_adk_agent.tools.cloud_run_skincare_tool import (
    CURRENT_IMAGE_B64,
    CURRENT_IMAGE_MIME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skincare_adk_agent.a2a_server")

AGENT_ENGINE_ID = CONFIG.agent_engine_id
AGENT_ENGINE_LOCATION = CONFIG.agent_engine_location
PROJECT_ID = CONFIG.project_id
PROJECT_NUMBER = CONFIG.project_number

# If Vertex AI Agent Runtime sets GOOGLE_CLOUD_PROJECT to the numeric project number,
# prefer the explicit alphanumeric GCP_PROJECT_ID injected via deploymentSpec.env.
if os.environ.get("GCP_PROJECT_ID"):
    PROJECT_ID = os.environ["GCP_PROJECT_ID"]
elif PROJECT_ID.isdigit() and not PROJECT_NUMBER:
    PROJECT_NUMBER = PROJECT_ID

if AGENT_ENGINE_ID:
    os.environ.setdefault("GOOGLE_CLOUD_AGENT_ENGINE_ID", AGENT_ENGINE_ID)
os.environ.setdefault("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION", AGENT_ENGINE_LOCATION)
os.environ["GOOGLE_CLOUD_LOCATION"] = CONFIG.location
os.environ.setdefault("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT")


def _setup_agent_engine_telemetry() -> Any:
    """Configures OpenTelemetry trace export so spans appear in Vertex AI Agent Engine -> Traces."""
    if not AGENT_ENGINE_ID or not PROJECT_ID:
        logger.info(
            "Skipping Vertex AI Agent Engine telemetry setup (AGENT_ENGINE_ID=%r, PROJECT_ID=%r).",
            AGENT_ENGINE_ID,
            PROJECT_ID,
        )
        return None
    try:
        import opentelemetry.sdk.resources as otel_resource
        import opentelemetry.sdk.trace as otel_trace
        import opentelemetry.sdk.trace.export as otel_export
        import opentelemetry.trace as trace_api
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        cloud_resource_id = (
            f"//aiplatform.googleapis.com/projects/{PROJECT_ID}"
            f"/locations/{AGENT_ENGINE_LOCATION}/reasoningEngines/{AGENT_ENGINE_ID}"
        )
        urn_project = PROJECT_NUMBER or PROJECT_ID
        agent_urn = (
            f"urn:agent:projects-{urn_project}:projects:{urn_project}"
            f":locations:{AGENT_ENGINE_LOCATION}:aiplatform:reasoningEngines:{AGENT_ENGINE_ID}"
        )
        resource = otel_resource.Resource.create(
            attributes={
                "gcp.project_id": PROJECT_ID,
                "cloud.account.id": PROJECT_ID,
                "cloud.provider": "gcp",
                "cloud.platform": "gcp.agent_engine",
                "service.name": AGENT_ENGINE_ID,
                "service.instance.id": f"{uuid.uuid4().hex}-{os.getpid()}",
                "cloud.region": AGENT_ENGINE_LOCATION,
                "cloud.resource_id": cloud_resource_id,
                "cloud.resource.id": cloud_resource_id,
                "gen_ai.main_agent.id": agent_urn,
            }
        )
        creds, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/trace.append",
            ]
        )
        auth_session = google.auth.transport.requests.AuthorizedSession(creds)
        otlp_exporter = OTLPSpanExporter(
            session=auth_session,
            endpoint="https://telemetry.googleapis.com/v1/traces",
            headers={
                "x-goog-user-project": PROJECT_ID,
                "User-Agent": "Vertex-Agent-Engine/1.0 OTel-OTLP-Exporter-Python/1.0",
            },
        )
        cloud_trace_exporter = CloudTraceSpanExporter(
            project_id=PROJECT_ID,
            resource_regex=".*",
        )
        tracer_provider = otel_trace.TracerProvider(resource=resource)
        tracer_provider.add_span_processor(otel_export.BatchSpanProcessor(otlp_exporter))
        tracer_provider.add_span_processor(otel_export.BatchSpanProcessor(cloud_trace_exporter))
        trace_api.set_tracer_provider(tracer_provider)

        try:
            from opentelemetry.instrumentation import google_genai

            google_genai.GoogleGenAiSdkInstrumentor().instrument()
        except Exception as inst_exc:
            logger.debug("GoogleGenAiSdkInstrumentor not enabled: %s", inst_exc)

        logger.info("Configured Vertex AI Agent Engine OpenTelemetry tracing for %s", cloud_resource_id)
        return tracer_provider
    except Exception as exc:
        logger.warning("OpenTelemetry setup skipped: %s", exc)
        return None


TRACER_PROVIDER = _setup_agent_engine_telemetry()

# Use VertexAiSessionService when GOOGLE_CLOUD_AGENT_ENGINE_ID is configured so sessions & events
# populate the Vertex AI Agent Engine -> Sessions tab; fallback to InMemorySessionService locally.
USE_IN_MEMORY_SESSIONS = (
    os.environ.get("ADK_USE_IN_MEMORY_SESSIONS", "").lower() in ("1", "true")
    or not AGENT_ENGINE_ID
    or not PROJECT_ID
)
if USE_IN_MEMORY_SESSIONS:
    APP_NAME = "skincare_adk_agent"
    SESSION_SERVICE = InMemorySessionService()
else:
    APP_NAME = AGENT_ENGINE_ID
    SESSION_SERVICE = VertexAiSessionService(
        project=PROJECT_ID,
        location=AGENT_ENGINE_LOCATION,
        agent_engine_id=AGENT_ENGINE_ID,
    )

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=SESSION_SERVICE,
    artifact_service=InMemoryArtifactService(),
    memory_service=InMemoryMemoryService(),
)

# Cache uploaded images and mapped Vertex AI session IDs by GECX context_id
SESSION_IMAGE_CACHE: dict[str, tuple[str, str]] = {}
CONTEXT_TO_VERTEX_SESSION_ID: dict[str, str] = {}

app = FastAPI(
    title="Lumiere Skincare Specialist ADK Agent (A2A + Vertex AI Agent Runtime)",
    description=(
        "Google ADK Agent running gemini-3.8-flash that receives delegated skincare image "
        "requests from Gemini Enterprise for Customer Experience (GECX / ces.googleapis.com) "
        "and invokes the Cloud Run Nano Banana Skincare API."
    ),
    version="1.0.0",
)


def extract_a2a_input(payload: dict[str, Any]) -> tuple[str, str | None, str, str]:
    """Extracts `(user_text, image_b64, mime_type, context_id)` from an A2A v1.0 / v0.3 request.

    Inspects both:
    1. `message.parts` / `request.content` for inline images (`raw` in A2A v1.0, `file.fileWithBytes` in v0.3)
       sent by `RemoteA2aAgent` (sub-agent handoff).
    2. `message.metadata["gecx_a2a_agent_context"]` (`history[].chunks[].image`) sent by BOTH
       `RemoteAgentTool` (tool call) and `RemoteA2aAgent` (sub-agent handoff).
    """
    if "params" in payload and isinstance(payload["params"], dict):
        payload = payload["params"]

    msg = payload.get("message") or payload.get("request") or payload
    context_id = (
        msg.get("contextId")
        or msg.get("context_id")
        or payload.get("contextId")
        or f"ctx-{uuid.uuid4().hex[:10]}"
    )

    text_parts: list[str] = []
    image_b64: str | None = None
    mime_type: str = "image/png"

    parts = msg.get("parts") or msg.get("content") or []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("text"):
            text_parts.append(str(part["text"]))
        if part.get("raw"):
            image_b64 = str(part["raw"])
            mime_type = str(part.get("mediaType") or part.get("mimeType") or "image/png")
        elif isinstance(part.get("file"), dict) and part["file"].get("fileWithBytes"):
            image_b64 = str(part["file"]["fileWithBytes"])
            mime_type = str(part["file"].get("mimeType") or "image/png")
        elif isinstance(part.get("inlineData"), dict) and part["inlineData"].get("data"):
            image_b64 = str(part["inlineData"]["data"])
            mime_type = str(part["inlineData"].get("mimeType") or "image/png")

    metadata = msg.get("metadata") or payload.get("metadata") or {}
    if isinstance(metadata, dict):
        ctx_raw = metadata.get("gecx_a2a_agent_context") or metadata.get("ces_a2a_agent_context")
        if ctx_raw:
            try:
                ctx_obj = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
                if isinstance(ctx_obj, dict):
                    history = ctx_obj.get("history") or []
                    for turn in reversed(history):
                        if not isinstance(turn, dict):
                            continue
                        for chunk in reversed(turn.get("chunks") or []):
                            if not isinstance(chunk, dict):
                                continue
                            img_obj = chunk.get("image")
                            if isinstance(img_obj, dict) and img_obj.get("data") and not image_b64:
                                image_b64 = str(img_obj["data"])
                                mime_type = str(
                                    img_obj.get("mimeType")
                                    or img_obj.get("mime_type")
                                    or "image/png"
                                )
                            elif chunk.get("text") and not text_parts:
                                text_parts.append(str(chunk["text"]))
                    variables = ctx_obj.get("variables") or {}
                    if isinstance(variables, dict) and not image_b64:
                        for k in ("image_base64", "skin_image", "uploaded_image"):
                            if variables.get(k):
                                image_b64 = str(variables[k])
                                break
            except Exception as exc:
                logger.warning("Failed to parse gecx_a2a_agent_context: %s", exc)

    if image_b64:
        SESSION_IMAGE_CACHE[context_id] = (image_b64, mime_type)
    elif context_id in SESSION_IMAGE_CACHE:
        image_b64, mime_type = SESSION_IMAGE_CACHE[context_id]

    user_text = "\n".join(text_parts).strip()
    if not user_text:
        user_text = "Please analyze my uploaded skincare image and recommend a Lumiere routine."

    return user_text, image_b64, mime_type, context_id


async def run_adk_specialist(
    user_text: str,
    image_b64: str | None,
    mime_type: str,
    context_id: str,
) -> tuple[str, dict[str, Any]]:
    """Executes the Google ADK `root_agent` (`gemini-3.8-flash`) with managed Vertex AI Session & Trace telemetry."""
    user_id = "gecx_customer"
    initial_state: dict[str, Any] = {
        "gecx_context_id": context_id,
        "has_uploaded_image": bool(image_b64),
        "uploaded_image_mime_type": mime_type,
        "uploaded_image_bytes": len(image_b64) if image_b64 else 0,
    }
    if USE_IN_MEMORY_SESSIONS and image_b64:
        initial_state["uploaded_image_base64"] = image_b64

    target_session_id = CONTEXT_TO_VERTEX_SESSION_ID.get(context_id)
    session = None
    if target_session_id:
        try:
            session = await runner.session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=target_session_id
            )
        except Exception as exc:
            logger.warning("Could not load existing session %s: %s", target_session_id, exc)
            session = None

    if session is None:
        create_kwargs: dict[str, Any] = {
            "app_name": APP_NAME,
            "user_id": user_id,
            "state": initial_state,
        }
        if USE_IN_MEMORY_SESSIONS:
            create_kwargs["session_id"] = context_id
        session = await runner.session_service.create_session(**create_kwargs)
        target_session_id = session.id
        CONTEXT_TO_VERTEX_SESSION_ID[context_id] = target_session_id
        logger.info(
            "Created managed ADK session id=%s for gecx_context_id=%s",
            target_session_id,
            context_id,
        )

    tok_b64 = CURRENT_IMAGE_B64.set(image_b64)
    tok_mime = CURRENT_IMAGE_MIME.set(mime_type)
    try:
        prompt_with_status = user_text
        if image_b64:
            prompt_with_status = (
                f"{user_text}\n\n"
                f"[System Context: A customer skincare image ({mime_type}, base64 length={len(image_b64)}) "
                f"is loaded in session state. Call `analyze_skincare_image_on_cloud_run` now.]"
            )
        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_with_status)],
        )
        final_text_chunks: list[str] = []
        async for event in runner.run_async(
            user_id=user_id,
            session_id=target_session_id,
            new_message=new_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for p in event.content.parts:
                    if getattr(p, "text", None):
                        final_text_chunks.append(p.text)

        final_response = "\n".join(final_text_chunks).strip()
        updated_session = await runner.session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=target_session_id
        )
        last_analysis = (
            updated_session.state.get("last_skincare_analysis", {})
            if updated_session and updated_session.state
            else {}
        )
        if not final_response and last_analysis:
            final_response = (
                f"Nano Banana Visual Analysis ({last_analysis.get('analysis_id')}): "
                f"{last_analysis.get('visual_assessment', {}).get('visual_summary', '')}"
            )
        return final_response, last_analysis
    finally:
        CURRENT_IMAGE_B64.reset(tok_b64)
        CURRENT_IMAGE_MIME.reset(tok_mime)
        if TRACER_PROVIDER is not None:
            try:
                TRACER_PROVIDER.force_flush(timeout_millis=5000)
            except Exception as flush_exc:
                logger.debug("Trace flush warning: %s", flush_exc)


def build_a2a_v10_message_object(
    response_text: str,
    context_id: str,
    last_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Formats the response as an A2A v1.0 SendMessageResponse / StreamResponse payload."""
    agent_ctx_metadata = {
        "variables": {
            "skincare_analysis_completed": bool(last_analysis),
            "skincare_analysis_id": str(last_analysis.get("analysis_id", "")),
            "skincare_model_used": str(last_analysis.get("model_used", "gemini-3.1-flash-image")),
            "apparent_skin_type": str(
                last_analysis.get("visual_assessment", {}).get("apparent_skin_type", "")
            ),
        },
        "sessionMetadata": {"closed": False},
    }
    return {
        "message": {
            "messageId": f"msg-{uuid.uuid4().hex[:12]}",
            "contextId": context_id,
            "role": "ROLE_AGENT",
            "parts": [{"text": response_text}],
            "metadata": {
                "gecx_a2a_agent_context": json.dumps(agent_ctx_metadata),
            },
        }
    }


@app.get("/")
@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint for Vertex AI Agent Runtime and local verification."""
    return {
        "status": "ok",
        "service": "skincare-adk-agent",
        "adk_model": CONFIG.model_name,
        "cloud_run_skincare_api": CONFIG.skincare_api_url,
        "agent_engine_id": AGENT_ENGINE_ID,
        "session_service": type(SESSION_SERVICE).__name__,
    }


@app.get("/.well-known/agent-card.json")
@app.get("/api/a2a/v1/card")
async def agent_card(request: Request) -> dict[str, Any]:
    """Returns the A2A v1.0 Agent Card."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "name": "skincare_specialist_adk_agent",
        "description": (
            "Google ADK Clinical Skincare Specialist Agent (gemini-3.8-flash) that delegates "
            "skincare image diagnostics to the Cloud Run Nano Banana API (gemini-3.1-flash-image)."
        ),
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
        },
        "defaultInputModes": ["text/plain", "image/png", "image/jpeg"],
        "defaultOutputModes": ["text/plain"],
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
    }


@app.post("/message:send")
@app.post("/v1/message:send")
@app.post("/api/a2a/v1/message:send")
@app.post("/api/a2a/message:send")
async def handle_a2a_send(request: Request) -> JSONResponse:
    """Handles unary A2A `message:send` requests from GECX (`RemoteAgentTool` / `RemoteA2aAgent`)."""
    payload = await request.json()
    logger.info("Received A2A message:send on %s", request.url.path)

    user_text, image_b64, mime_type, context_id = extract_a2a_input(payload)
    logger.info(
        "Extracted A2A input: context_id=%s, has_image=%s, text=%r",
        context_id,
        bool(image_b64),
        user_text[:120],
    )

    response_text, last_analysis = await run_adk_specialist(
        user_text=user_text,
        image_b64=image_b64,
        mime_type=mime_type,
        context_id=context_id,
    )
    resp_obj = build_a2a_v10_message_object(response_text, context_id, last_analysis)

    if payload.get("jsonrpc") == "2.0":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id", "1"),
                "result": resp_obj,
            }
        )
    return JSONResponse(resp_obj)


@app.post("/message:stream")
@app.post("/v1/message:stream")
@app.post("/api/a2a/v1/message:stream")
@app.post("/api/a2a/message:stream")
async def handle_a2a_stream(request: Request) -> StreamingResponse:
    """Handles streaming SSE A2A `message:stream` requests from GECX (`RemoteA2aAgent`)."""
    payload = await request.json()
    logger.info("Received A2A message:stream on %s", request.url.path)

    user_text, image_b64, mime_type, context_id = extract_a2a_input(payload)
    logger.info(
        "Extracted A2A stream input: context_id=%s, has_image=%s, text=%r",
        context_id,
        bool(image_b64),
        user_text[:120],
    )

    response_text, last_analysis = await run_adk_specialist(
        user_text=user_text,
        image_b64=image_b64,
        mime_type=mime_type,
        context_id=context_id,
    )
    resp_obj = build_a2a_v10_message_object(response_text, context_id, last_analysis)

    async def event_generator():
        data_str = json.dumps(resp_obj, separators=(",", ":"))
        yield f"data: {data_str}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/reasoning_engine")
async def handle_reasoning_engine_query(request: Request) -> JSONResponse:
    """Handles native Vertex AI Agent Runtime (`reasoningEngines:query`) requests."""
    payload = await request.json()
    input_data = payload.get("input") or payload
    user_text = (
        input_data.get("prompt")
        or input_data.get("query")
        or input_data.get("user_query")
        or "Please analyze my uploaded skincare image."
    )
    image_b64 = input_data.get("image_base64")
    mime_type = input_data.get("mime_type", "image/png")
    context_id = input_data.get("context_id") or f"re-{uuid.uuid4().hex[:10]}"

    response_text, last_analysis = await run_adk_specialist(
        user_text=user_text,
        image_b64=image_b64,
        mime_type=mime_type,
        context_id=context_id,
    )
    return JSONResponse(
        {
            "output": {
                "response": response_text,
                "context_id": context_id,
                "cloud_run_nano_banana_telemetry": last_analysis,
            }
        }
    )
