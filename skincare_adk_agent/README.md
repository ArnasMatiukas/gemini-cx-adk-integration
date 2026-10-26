# Tier 2: Google ADK Skincare Specialist Agent (`skincare_adk_agent`)

Deployed on **Vertex AI Agent Runtime (`reasoningEngines`)** and invoked via **A2A v1.0** by **Gemini Enterprise for Customer Experience (GECX)**.

## Files

- **[`agent.py`](agent.py)**: Defines the Google ADK `root_agent` (`gemini-3.8-flash`) and the `analyze_skin_condition_image` `FunctionTool`, which analyzes the uploaded skin image directly on Vertex AI (`gemini-3.8-flash` with Pydantic `VisualAssessment` structured output) and identifies what is wrong with the skin (`analysis_id`, `apparent_skin_type`, `hydration_level`, `visible_markers`, `primary_concern_tags`, `key_recommended_actives`, `visual_summary`) **without recommending any actual products or SKUs**. Product recommendations are handled by a separate GECX sub-agent (`lumiere_product_recommendation_agent`).
- **[`server.py`](server.py)**: FastAPI A2A v1.0 server (`POST /api/a2a/v1/message:send`) that:
  1. Extracts the customer's uploaded image from GECX's `metadata.gecx_a2a_agent_context`.
  2. Persists conversations via `VertexAiSessionService` so sessions appear in the **Vertex AI Agent Engine $\rightarrow$ Sessions** console tab.
  3. Exports OpenTelemetry spans so traces appear in the **Vertex AI Agent Engine $\rightarrow$ Traces** console tab.
