# Tier 3: Cloud Run Skincare Vision API (`skincare_api`)

A minimal FastAPI microservice on **Google Cloud Run** that analyzes customer-uploaded skin and product photos using **Gemini 3.8 Flash (`gemini-3.8-flash`)** on Vertex AI.

## How It Works

1. The **Google ADK Agent** (Tier 2) calls `POST /v1/analyze-skin` with an OIDC ID token and `{ "image_base64": "...", "mime_type": "image/png", "user_query": "..." }`.
2. [`app.py`](app.py) invokes `gemini-3.8-flash` (`location="global"`) with native **Structured Outputs** (`response_mime_type="application/json"`, `response_schema=VisualAssessment`).
3. Detected concern tags are matched against `SKINCARE_CATALOG` and returned as JSON (`analysis_id`, `model_used`, `visual_assessment`, `recommended_regimen`).

## Local Run

```bash
pip install -r skincare_api/requirements.txt
GCP_PROJECT_ID=your-project-id python3 -m uvicorn skincare_api.app:app --port 8080
```
