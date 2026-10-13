# Tier 3: Cloud Run Skincare Image Analysis API (`skincare_api`)

This microservice implements **Tier 3** of the **GECX $\rightarrow$ ADK Agent Runtime $\rightarrow$ Cloud Run API** reference architecture.

It runs as a serverless FastAPI application on **Google Cloud Run** and is invoked as an authenticated tool by the **Google ADK Skincare Specialist Agent** (Tier 2).

---

## How It Works

1. **Authenticated Tool Invocation**:
   - The ADK Agent on Vertex AI Agent Runtime mints an OIDC ID token for the Cloud Run service URL and sends a `POST /v1/analyze-skin` request containing the customer's uploaded skincare image (`image_base64`, `mime_type`) and question (`user_query`).
2. **Multimodal Visual Analysis with Nano Banana 2 (`gemini-3.1-flash-image`)**:
   - The API decodes the image bytes and invokes `gemini-3.1-flash-image` on Vertex AI (`location="global"`) using the `google-genai` SDK.
   - It extracts structured cosmetic dermatology observations (`apparent_skin_type`, `hydration_level`, `visible_markers`, `primary_concern_tags`, `severity_estimate`, `key_recommended_actives`, and `visual_summary`).
3. **Catalog Regimen Matching**:
   - Detected concern tags are matched against the curated **Lumiere Clinical Skincare Catalog** (`SKINCARE_CATALOG`) to return tailored SKUs and an AM/PM application protocol.

---

## Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check returning active model (`gemini-3.1-flash-image`) and GCP project configuration. |
| `GET` | `/v1/catalog` | Returns the curated Lumiere clinical skincare product catalog. |
| `POST` | `/v1/analyze-skin` | Analyzes a base64-encoded skin or product image with `gemini-3.1-flash-image` and returns a structured `SkinAnalysisResponse`. |

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GCP_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` | *(Required at runtime)* | Your Google Cloud Project ID used for Vertex AI `genai.Client(vertexai=True)`. |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex AI endpoint location for `gemini-3.1-flash-image`. |
| `NANO_BANANA_MODEL` | `gemini-3.1-flash-image` | Multimodal vision model used for skin/product image analysis. |

---

## Running Locally

```bash
pip install -r skincare_api/requirements.txt
export GCP_PROJECT_ID="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="global"
python3 -m uvicorn skincare_api.app:app --host 0.0.0.0 --port 8080
```
