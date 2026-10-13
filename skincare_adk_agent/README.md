# Tier 2: Google ADK Skincare Specialist Agent (`skincare_adk_agent`)

This component implements **Tier 2** of the **GECX $\rightarrow$ ADK Agent Runtime $\rightarrow$ Cloud Run API** reference architecture.

It is built with the **Google Agent Development Kit (ADK)** (`google-adk`) running `gemini-3.8-flash` and is deployed as a custom container on **Vertex AI Agent Runtime (`reasoningEngines`)**.

---

## Architecture & Responsibilities

1. **A2A v1.0 Protocol Server (`a2a_server.py`)**:
   - Exposes the Agent-to-Agent (A2A v1.0) HTTP/JSON endpoints required by Vertex AI Agent Runtime (`POST /api/a2a/v1/message:send`, `POST /api/a2a/v1/message:stream`, `GET /api/a2a/v1/card`) and native `:query` (`POST /api/reasoning_engine`).
   - Extracts customer-uploaded skincare images (`image/png`, `image/jpeg`) from **both** A2A inline message parts (`raw` / `file.fileWithBytes`) and the GECX conversation history (`message.metadata.gecx_a2a_agent_context.history[].chunks[].image`).
2. **ADK `root_agent` (`agent.py`) & Cloud Run Tool (`tools/cloud_run_skincare_tool.py`)**:
   - Runs `gemini-3.8-flash` (`location="global"`) with thinking enabled (`thinking_budget=8192`).
   - Invokes the `analyze_skincare_image_on_cloud_run` `FunctionTool`, which automatically mints a Google-signed OIDC ID token and calls the **Tier 3 Cloud Run Skincare API (`POST /v1/analyze-skin`)** powered by Nano Banana (`gemini-3.1-flash-image`).
3. **Managed Vertex AI Agent Engine Sessions (`VertexAiSessionService`)**:
   - When `GOOGLE_CLOUD_AGENT_ENGINE_ID` is set in the runtime environment, `a2a_server.py` uses `VertexAiSessionService` so every customer conversation creates a persistent session and event timeline in the **Google Cloud Console $\rightarrow$ Vertex AI $\rightarrow$ Agent Engine $\rightarrow$ Sessions** tab.
   -Note: Because Vertex AI Agent Engine enforces a `64 KB` state size limit on `CreateSession`, the raw base64 image is passed to the tool via a thread-safe `contextvars.ContextVar` while lightweight image metadata and the structured Nano Banana analysis response (`last_skincare_analysis`) are persisted in session state.
4. **Managed OpenTelemetry Tracing (`_setup_agent_engine_telemetry`)**:
   - Exports ADK agent/LLM/tool spans to both `https://telemetry.googleapis.com/v1/traces` (`OTLPSpanExporter`) and Cloud Trace (`CloudTraceSpanExporter`) with the resource attributes expected by the **Vertex AI Agent Engine $\rightarrow$ Traces** dashboard (`cloud.platform="gcp.agent_engine"`, `cloud.resource_id`, and `gen_ai.main_agent.id`).

---

## Running Unit Tests Locally

```bash
PYTHONPATH=skincare_adk_agent python3 -m unittest discover -s skincare_adk_agent/tests -v
```
