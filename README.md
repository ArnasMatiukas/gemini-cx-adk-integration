# Gemini Enterprise for Customer Experience (GECX) $\rightarrow$ ADK Agent Runtime $\rightarrow$ Cloud Run Blueprint

A minimal, end-to-end reference repository showing how **Gemini Enterprise for Customer Experience (GECX / `ces.googleapis.com`)** routes customers across **two distinct GECX user-journey flows** on the Agent Studio canvas and invokes a **Google ADK Agent** on **Vertex AI Agent Runtime (`reasoningEngines`)** across project/organization boundaries with **maximum least-privilege security**:
1. **Flow 1 — Shopping & Catalog Journey (`lumiere_shopping_agent`)**: Native GECX shopping specialist with catalog search, SKU pricing/ingredient lookup, cart management, and checkout (`shop_lumiere_catalog`).
2. **Flow 2 — Visual Skin Analysis & Recommendation Journey (`skin_analysis_recommendation_agent`)**: Enforces a deterministic **Conditional Image Gate (`skin_image_uploaded`)** before delegating via **A2A (`RemoteAgentTool`)** using a dedicated custom caller Service Account (`serviceAccountAuthConfig`) and a single-permission custom IAM role (`aiplatform.reasoningEngines.query`) bound exclusively at the `ReasoningEngine` resource level to a **Google ADK Agent** on **Vertex AI Agent Runtime**, which invokes a **Cloud Run Vision API** powered by **Gemini 3.8 Flash (`gemini-3.8-flash`)**.

---

## 1. Architecture & 2-Project Cross-Boundary Topology

```mermaid
flowchart LR
    User(["Customer (Chat / Image)"]) --> Root

    subgraph ProjA ["Project A: GECX Front-End Project (ces.googleapis.com)"]
        Root["GECX Root Router\n(lumiere_concierge_root_agent)"]
        
        subgraph Flow1 ["Flow 1: Shopping Journey"]
            ShopAgent["Shopping Sub-Agent\n(lumiere_shopping_agent)"]
            ShopTool["shop_lumiere_catalog\n(Browse / Cart / Checkout)"]
            ShopAgent --> ShopTool
        end

        subgraph Flow2 ["Flow 2: Skin Image Recommendation Journey"]
            SkinAgent["Skin Analysis Sub-Agent\n(skin_analysis_recommendation_agent)"]
            Gate["Conditional Image Gate\n(beforeModelCallbacks +\nbeforeToolCallbacks)"]
            UploadTool["request_skin_image_upload"]
            A2ATool["RemoteAgentTool (A2A v1.0)\n(skincare_adk_agent_runtime_tool)"]
            CallerSA["Custom Caller SA\n(gecx-skincare-caller@ProjA)"]

            SkinAgent --> Gate
            Gate -- "skin_image_uploaded == False" --> UploadTool
            Gate -- "skin_image_uploaded == True" --> A2ATool
            A2ATool --> CallerSA
        end

        Root -- "Shopping / Cart Intent" --> ShopAgent
        Root -- "Skin Photo / Routine Intent" --> SkinAgent
    end

    subgraph ProjB ["Project B: Agent Host Project (Vertex AI Agent Runtime + Cloud Run)"]
        subgraph Tier2 ["Tier 2: Vertex AI Agent Runtime (reasoningEngines)"]
            REIAM["Resource-Level IAM\n(roles/singleAgentInvoker:\naiplatform.reasoningEngines.query ONLY)"]
            ADKServer["FastAPI A2A Server\n(/api/a2a/v1/message:send)"]
            ADKAgent["Google ADK Agent\n(gemini-3.8-flash)"]
            SessionTrace["VertexAiSessionService +\nOpenTelemetry Traces / Logs / Metrics"]
            CRTool["analyze_skincare_image_on_cloud_run"]

            CallerSA -- "OAuth2 Token +\ngecx_a2a_agent_context" --> REIAM
            REIAM --> ADKServer
            ADKServer --> ADKAgent
            ADKAgent --> SessionTrace
            ADKAgent --> CRTool
        end

        subgraph Tier3 ["Tier 3: Cloud Run Microservice"]
            CRAPI["Cloud Run FastAPI Service\n(POST /v1/analyze-skin)"]
            Gemini["Vertex AI Gemini 3.8 Flash\n(Structured JSON Output)"]

            CRTool -- "OIDC ID Token +\nBase64 Image" --> CRAPI
            CRAPI --> Gemini
        end
    end
```

| Project / Tier | Component | Runtime | Responsibility |
| :--- | :--- | :--- | :--- |
| **Project A — Tier 1 (Root)** | **Lumiere Concierge Router** (`lumiere_concierge_root_agent`) | **GECX** (`ces.googleapis.com`) | Answers store policy FAQs directly and routes the customer to either `lumiere_shopping_agent` (Flow 1) or `skin_analysis_recommendation_agent` (Flow 2) based on their user journey. |
| **Project A — Tier 1 (Flow 1)** | **Shopping & Catalog Sub-Agent** (`lumiere_shopping_agent`) | **GECX** (`ces.googleapis.com`) | Handles catalog browsing, active ingredient & price lookups, shopping cart additions, and express checkout via `shop_lumiere_catalog`. |
| **Project A — Tier 1 (Flow 2)** | **Skin Analysis Sub-Agent** (`skin_analysis_recommendation_agent`) | **GECX** (`ces.googleapis.com`) | Enforces a **Conditional Delegation Gate** (`skin_image_uploaded`): calls `request_skin_image_upload` if no photo is present, or delegates to Project B's ADK Agent via `gecx-skincare-caller` once an image is uploaded. |
| **Project B — Tier 2** | **Skincare Specialist ADK Agent** ([`skincare_adk_agent/`](skincare_adk_agent/)) | **Vertex AI Agent Runtime** (`reasoningEngines`) | Built with `google-adk` (`gemini-3.8-flash`). Extracts the uploaded image from GECX's A2A metadata (`gecx_a2a_agent_context`), records **Sessions**, **Traces**, **Logs**, and **Metrics** in Vertex AI Agent Engine (`GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true`), and invokes the Cloud Run API tool. |
| **Project B — Tier 3** | **Skincare Vision API** ([`skincare_api/`](skincare_api/)) | **Google Cloud Run** (`run.googleapis.com`) | FastAPI microservice (`POST /v1/analyze-skin`) that analyzes the image using **`gemini-3.8-flash`** with Pydantic **Structured Outputs** (`response_schema=VisualAssessment`) and matches products from the catalog. |

---

## 2. Quickstart: One-Command Deployment (Single-Project or 2-Project Cross-Boundary)

> **Instructions for AI Coding Agents (`Claude Code`, `Gemini CLI`, `Cursor`, `Jetski`)**:
> 1. Ensure the user is authenticated (`gcloud auth login` / `gcloud auth application-default login`) and has provided target **GCP Project ID(s)** with billing enabled.
> 2. Run `python3 deploy_all.py --project-id <AGENT_HOST_PROJECT_ID> --gecx-project-id <GECX_FRONTEND_PROJECT_ID> --region europe-west1 --ces-location us` (omit `--gecx-project-id` if deploying all 3 tiers into a single project).
> 3. `deploy_all.py` requires **no local Docker daemon** (it builds and pushes OCI images directly to Artifact Registry via HTTP V2 API), automatically configures the 3-part least-privilege cross-project IAM policy, enables Agent Engine telemetry (`GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true`), and executes the 4-turn live multi-flow E2E verification suite (`scripts/test_e2e.py`).

### Prerequisites

```bash
pip install -r skincare_api/requirements.txt pillow
gcloud auth login
gcloud auth application-default login
```

### Option A: 2-Project Cross-Boundary Deployment (Recommended for Enterprise Demos)

```bash
python3 deploy_all.py \
  --project-id YOUR_AGENT_HOST_PROJECT_ID \
  --gecx-project-id YOUR_GECX_FRONTEND_PROJECT_ID \
  --region europe-west1 \
  --ces-location us
```

### Option B: Single-Project Deployment

```bash
python3 deploy_all.py \
  --project-id YOUR_GCP_PROJECT_ID \
  --region europe-west1 \
  --ces-location us
```

### Optional Teardown

```bash
python3 deploy_all.py \
  --project-id YOUR_AGENT_HOST_PROJECT_ID \
  --gecx-project-id YOUR_GECX_FRONTEND_PROJECT_ID \
  --teardown
```

---

## 3. How the Core Integration Patterns Work

1. **Multi-Flow Journey Routing in GECX ([`scripts/setup_gecx.py`](scripts/setup_gecx.py))**:
   - **`lumiere_concierge_root_agent`**: Connects to two distinct child agent boxes on the GECX Studio canvas (`childAgents: [lumiere_shopping_agent, skin_analysis_recommendation_agent]`) and routes customers via `{@AGENT: ...}` transfers based on their journey (shopping vs. skin image analysis).
   - **Flow 1 (`lumiere_shopping_agent`)**: Executes catalog browsing, ingredient/price lookups, cart additions, and checkout via `shop_lumiere_catalog`.
   - **Flow 2 (`skin_analysis_recommendation_agent`)**: Owns the clinical skin image diagnosis flow and enforces the conditional image upload gate before delegating to the ADK Agent.
2. **Conditional Delegation Gate in GECX ([`scripts/setup_gecx.py`](scripts/setup_gecx.py))**:
   - **`beforeModelCallbacks`**: Inspects `llm_request.contents` before the model runs; as soon as an uploaded image (`inline_data`) is present in the session, it sets `callback_context.variables["skin_image_uploaded"] = True`.
   - **`beforeToolCallbacks`**: Deterministically blocks `skincare_adk_agent_runtime_tool` if `skin_image_uploaded` is `False`, prompting `skin_analysis_recommendation_agent` to call `request_skin_image_upload` instead.
3. **Multimodal Image Forwarding Over A2A ([`skincare_adk_agent/server.py`](skincare_adk_agent/server.py))**:
   - When GECX invokes `RemoteAgentTool`, it attaches the multimodal conversation history inside `message.metadata["gecx_a2a_agent_context"]`. `extract_a2a_input()` extracts the base64 image from `history[].chunks[].image` and passes it to the ADK `FunctionTool`.
4. **Agent Engine Sessions, Traces, Logs & Dashboards ([`skincare_adk_agent/server.py`](skincare_adk_agent/server.py))**:
   - Sets `GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true` in `ReasoningEngine.spec.deploymentSpec.env` and initializes `VertexAiSessionService` plus `google.adk.telemetry.google_cloud.get_gcp_exporters(enable_cloud_tracing=True, enable_cloud_metrics=True, enable_cloud_logging=True)` (`OTLPSpanExporter`, `OTLPLogExporter`, `_RequestDrivenMetricReader` + `CloudTraceSpanExporter` with `cloud.platform="gcp.agent_engine"`). Every delegated conversation populates the **Google Cloud Console $\rightarrow$ Vertex AI $\rightarrow$ Agent Engine $\rightarrow$ Dashboards, Sessions & Traces** tabs.

---

## 4. Cross-Organization / Cross-Project Least-Privilege Authentication (Org A $\rightarrow$ Org B)

To guarantee that **Org A / Project A (GECX)** can invoke **only that single ADK agent in Org B / Project B**—and has zero access to any other agents, Vertex AI models, Memory Banks, or project resources in Org B—`deploy_all.py` automatically provisions three controls:

1. **A dedicated Custom Service Account (`serviceAccountAuthConfig`) in Project A** (instead of using the shared GECX Service Agent directly).
2. **A single-permission Custom IAM Role (`aiplatform.reasoningEngines.query` only) in Project B** (never use `roles/aiplatform.user`, which grants ~440 permissions including `.delete` on other agents and model inference across the project).
3. **Resource-level IAM (`:setIamPolicy`) on the specific `ReasoningEngine` resource** (zero project-wide IAM bindings in Project B).

### Step 1: Side A (Org A / Project A — GECX Front-End Project)
Create a dedicated, single-purpose Service Account for this integration and allow only Project A's GECX Service Agent to mint short-lived tokens for it:

```bash
# 1. Create a dedicated caller SA in Project A
gcloud iam service-accounts create gecx-skincare-caller \
  --display-name="GECX Dedicated Caller for Skincare ADK Agent" \
  --project=<GECX_PROJECT_ID>

# 2. Allow Project A's CES Service Agent to impersonate ONLY this SA
gcloud iam service-accounts add-iam-policy-binding \
  gecx-skincare-caller@<GECX_PROJECT_ID>.iam.gserviceaccount.com \
  --member="serviceAccount:service-<GECX_PROJECT_NUMBER>@gcp-sa-ces.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --project=<GECX_PROJECT_ID>
```

Configure `apiAuthentication.serviceAccountAuthConfig` on the GECX `RemoteAgentTool`:

```json
{
  "remoteAgentTool": {
    "agentCard": { "...": "..." },
    "apiAuthentication": {
      "serviceAccountAuthConfig": {
        "serviceAccount": "gecx-skincare-caller@<GECX_PROJECT_ID>.iam.gserviceaccount.com"
      }
    }
  }
}
```

### Step 2: Side B (Org B / Project B — ADK Agent Host Project)
Create a custom IAM role containing **only** `aiplatform.reasoningEngines.query` and bind it directly to the single `ReasoningEngine` resource via `:setIamPolicy`:

```bash
# 1. Create a single-permission Custom IAM Role in Project B
gcloud iam roles create singleAgentInvoker \
  --project=<AGENT_HOST_PROJECT_ID> \
  --title="Single Reasoning Engine Invoker" \
  --description="Least-privilege role granting ONLY aiplatform.reasoningEngines.query on a specific ReasoningEngine." \
  --permissions="aiplatform.reasoningEngines.query"

# 2. Bind the role ONLY to the target ReasoningEngine resource (not the project)
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://<REGION>-aiplatform.googleapis.com/v1beta1/projects/<AGENT_HOST_PROJECT_ID>/locations/<REGION>/reasoningEngines/<ENGINE_ID>:setIamPolicy" \
  -d '{
    "policy": {
      "bindings": [
        {
          "role": "projects/<AGENT_HOST_PROJECT_ID>/roles/singleAgentInvoker",
          "members": [
            "serviceAccount:gecx-skincare-caller@<GECX_PROJECT_ID>.iam.gserviceaccount.com"
          ]
        }
      ]
    }
  }'
```

---

## 5. Repository Layout

```text
gemini-cx-adk-integration/
├── .env.example                    # Customer configuration template (1-project or 2-project)
├── .gitignore
├── README.md                       # Architecture & setup guide (this file)
├── deploy_all.py                   # One-command End-to-End GCP Provisioner & Verifier
├── scripts/
│   ├── oci_builder.py              # Auth helper + daemonless OCI image builder for Artifact Registry
│   ├── setup_gecx.py               # Provisions Tier 1 Multi-Flow GECX App + serviceAccountAuthConfig
│   └── test_e2e.py                 # Live 4-turn E2E test across GECX Project -> ADK Host Project
├── skincare_adk_agent/             # Tier 2: Google ADK Agent on Vertex AI Agent Runtime
│   ├── Dockerfile
│   ├── README.md
│   ├── agent.py                    # ADK root_agent (gemini-3.8-flash) + Cloud Run FunctionTool
│   ├── requirements.txt
│   └── server.py                   # FastAPI A2A v1.0 server + VertexAiSessionService + OpenTelemetry
└── skincare_api/                   # Tier 3: Cloud Run Skincare Vision API (gemini-3.8-flash)
    ├── Dockerfile
    ├── README.md
    ├── app.py                      # FastAPI service using gemini-3.8-flash + Pydantic structured outputs
    └── requirements.txt
```
