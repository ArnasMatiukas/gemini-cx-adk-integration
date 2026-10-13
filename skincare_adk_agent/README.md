# Tier 2: Google ADK Skincare Specialist Agent (`skincare_adk_agent`)

Deployed on **Vertex AI Agent Runtime (`reasoningEngines`)** and invoked via **A2A v1.0** by **Gemini Enterprise for Customer Experience (GECX)**.

## Files

- **[`agent.py`](agent.py)**: Defines the Google ADK `root_agent` (`gemini-3.8-flash`) and the `analyze_skincare_image_on_cloud_run` `FunctionTool` (which mints an OIDC ID token and calls Tier 3 Cloud Run).
- **[`server.py`](server.py)**: FastAPI A2A v1.0 server (`POST /api/a2a/v1/message:send`) that:
  1. Extracts the customer's uploaded image from GECX's `metadata.gecx_a2a_agent_context`.
  2. Persists conversations via `VertexAiSessionService` so sessions appear in the **Vertex AI Agent Engine $\rightarrow$ Sessions** console tab.
  3. Exports OpenTelemetry spans so traces appear in the **Vertex AI Agent Engine $\rightarrow$ Traces** console tab.
