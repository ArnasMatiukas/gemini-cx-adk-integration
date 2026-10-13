# Provisioning & Verification Scripts (`scripts/`)

This directory contains the modular automation scripts used by [`deploy_all.py`](../deploy_all.py) to build, deploy, configure, and test the 3-tier **Gemini Enterprise for Customer Experience (GECX) $\rightarrow$ Google ADK Agent on Vertex AI Agent Runtime $\rightarrow$ Cloud Run Skincare API** architecture.

---

## Script Reference

| Script | Purpose |
| :--- | :--- |
| [`gcp_auth.py`](gcp_auth.py) | Zero-config credential helper that loads `.env`, resolves numeric `project_number` via Cloud Resource Manager, and obtains OAuth2 Bearer tokens from ADC (`GOOGLE_APPLICATION_CREDENTIALS`), `~/.config/gcloud/credentials.db`, or `gcloud auth print-access-token`. |
| [`push_oci_image.py`](push_oci_image.py) | Daemonless OCI container image builder & Artifact Registry V2 uploader. Pulls the 43 base layers of `gcr.io/distroless/python3-debian12:latest` (`linux/amd64`), packs Python 3.11 `site-packages` and application code into deterministic tar+gzip layers, and publishes the manifest directly to Artifact Registry without needing Docker or Cloud Build. |
| [`setup_gecx_app.py`](setup_gecx_app.py) | Idempotently provisions the GECX (`ces.googleapis.com`) Application, `request_skin_image_upload` PythonFunction tool, `skincare_adk_agent_runtime_tool` (`RemoteAgentTool` pointing to the Vertex AI Agent Runtime A2A v1.0 endpoint), `lumiere_concierge_root_agent` (with `beforeModelCallbacks`, `beforeToolCallbacks`, and `<taskflow>` instructions), `AppVersion`, and `Deployment`. |
| [`test_e2e_workflow.py`](test_e2e_workflow.py) | Executes a 4-turn live conversation against `ces.googleapis.com` (`SessionService.RunSession`) verifying: (1) General FAQ without delegation, (2) Blocked delegation when no image is uploaded, (3) Full 3-tier A2A delegation when a skincare image is uploaded, (4) Blocked delegation when `diagnostic_consent_granted=False`, and (5) Managed session persistence in Vertex AI Agent Engine (`VertexAiSessionService`). |

---

## Standalone Usage

While [`deploy_all.py`](../deploy_all.py) runs all steps automatically, each script can also be executed individually:

```bash
# Update only the GECX App configuration in 'us' or 'eu'
python3 scripts/setup_gecx_app.py \
  --project-id YOUR_GCP_PROJECT_ID \
  --region europe-west1 \
  --reasoning-engine-id YOUR_REASONING_ENGINE_ID \
  --ces-location us

# Run the live 4-turn E2E verification test against your deployed GECX App
python3 scripts/test_e2e_workflow.py \
  --project-id YOUR_GCP_PROJECT_ID \
  --ces-location us \
  --region europe-west1 \
  --reasoning-engine-id YOUR_REASONING_ENGINE_ID
```
