#!/usr/bin/env python3
"""End-to-End Provisioner & Verifier for the 3-Tier GECX + ADK Agent Runtime + Cloud Run API Blueprint.

Provisions all infrastructure and application components from scratch in any customer GCP project:
  1. Enables required Google Cloud APIs (`run`, `artifactregistry`, `aiplatform`, `ces`, `cloudtrace`, `telemetry`).
  2. Generates GCP Service Identities and binds least-privilege IAM roles (`aiplatform.user`, `run.invoker`,
     `cloudtrace.agent`, `telemetry.tracesWriter`, `artifactregistry.reader`).
  3. Creates the Artifact Registry Docker repository (`gecx-skincare-repo`).
  4. Stages Python 3.11 `manylinux2014_x86_64` wheels and pushes both OCI container images
     (`skincare-nano-banana-api:latest` and `skincare-adk-agent:latest`) without requiring a local Docker daemon.
  5. Deploys Tier 3: Cloud Run Skincare Image Analysis API (`skincare-nano-banana-api`) powered by
     Nano Banana 2 (`gemini-3.1-flash-image`).
  6. Deploys Tier 2: Google ADK Skincare Specialist Agent (`gemini-3.8-flash`) on Vertex AI Agent Runtime
     (`reasoningEngines`) with `VertexAiSessionService` and OpenTelemetry Agent Engine Tracing.
  7. Provisions Tier 1: Gemini Enterprise for Customer Experience (`ces.googleapis.com`) Application,
     Root Agent, Two-Layer Conditional Delegation Gate (`beforeModelCallbacks` + `beforeToolCallbacks`),
     Remote A2A Tool (`skincare_adk_agent_runtime_tool`), AppVersion, and Web Deployment.
  8. Executes the 4-turn live End-to-End Verification Suite (`scripts/test_e2e_workflow.py`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gcp_auth import get_auth_headers, get_project_number, load_dotenv_file
from push_oci_image import build_and_push
from setup_gecx_app import setup_gecx_blueprint_app
from test_e2e_workflow import run_e2e_verification

REQUIRED_APIS = [
    "serviceusage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "compute.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "ces.googleapis.com",
    "cloudtrace.googleapis.com",
    "telemetry.googleapis.com",
]


def enable_gcp_apis(project_id: str, project_number: str, account: str | None = None) -> None:
    """Enables all required GCP APIs via Service Usage batchEnable."""
    print(f"\n[Step 1/8] Enabling required GCP APIs in project '{project_id}' ({project_number})...")
    headers = get_auth_headers(account=account)
    url = f"https://serviceusage.googleapis.com/v1/projects/{project_number}/services:batchEnable"
    r = requests.post(url, headers=headers, json={"serviceIds": REQUIRED_APIS}, timeout=60)
    if r.status_code != 200:
        # Fallback to enabling individually if batchEnable reports already-enabled or quota nuances
        for api in REQUIRED_APIS:
            requests.post(
                f"https://serviceusage.googleapis.com/v1/projects/{project_number}/services/{api}:enable",
                headers=headers,
                json={},
                timeout=30,
            )
        time.sleep(5)
        return

    op = r.json()
    op_name = op.get("name", "")
    if op_name and not op.get("done"):
        for _ in range(60):
            r_op = requests.get(
                f"https://serviceusage.googleapis.com/v1/{op_name}",
                headers=headers,
                timeout=20,
            )
            if r_op.status_code == 200 and r_op.json().get("done"):
                break
            time.sleep(3)
    print("  -> Required GCP APIs enabled.")


def ensure_service_identities_and_iam(
    project_id: str,
    project_number: str,
    account: str | None = None,
) -> list[str]:
    """Generates GCP service identities (P4SA/P6SA) and binds required project IAM roles."""
    print(f"\n[Step 2/8] Provisioning Service Identities and IAM bindings in '{project_id}'...")
    headers = get_auth_headers(account=account)
    for service in ("aiplatform.googleapis.com", "ces.googleapis.com", "run.googleapis.com"):
        url = (
            f"https://serviceusage.googleapis.com/v1beta1/projects/{project_number}"
            f"/services/{service}:generateServiceIdentity"
        )
        requests.post(url, headers=headers, json={}, timeout=20)

    candidate_service_accounts = [
        f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-aiplatform.iam.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-ces.iam.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@serverless-robot-prod.iam.gserviceaccount.com",
    ]
    roles_to_grant = [
        "roles/aiplatform.user",
        "roles/run.invoker",
        "roles/cloudtrace.agent",
        "roles/telemetry.tracesWriter",
        "roles/artifactregistry.reader",
    ]

    r_get = requests.post(
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:getIamPolicy",
        headers=headers,
        json={},
        timeout=20,
    )
    r_get.raise_for_status()
    policy = r_get.json()
    bindings: list[dict[str, Any]] = policy.get("bindings", [])

    # Apply each member individually so a not-yet-created service agent doesn't reject the whole policy update
    for member in candidate_service_accounts:
        test_bindings = [dict(b, members=list(b.get("members", []))) for b in bindings]
        changed = False
        for role in roles_to_grant:
            role_entry = next((b for b in test_bindings if b.get("role") == role), None)
            if role_entry is None:
                role_entry = {"role": role, "members": []}
                test_bindings.append(role_entry)
            if member not in role_entry["members"]:
                role_entry["members"].append(member)
                changed = True
        if not changed:
            continue
        r_set = requests.post(
            f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:setIamPolicy",
            headers=headers,
            json={"policy": {**policy, "bindings": test_bindings}},
            timeout=20,
        )
        if r_set.status_code == 200:
            policy = r_set.json()
            bindings = policy.get("bindings", [])
            print(f"  -> Bound IAM roles for {member}")

    return candidate_service_accounts


def ensure_artifact_registry_repo(
    project_id: str,
    project_number: str,
    region: str,
    repo_name: str,
    account: str | None = None,
) -> None:
    """Creates the Docker Artifact Registry repository if needed and binds repository-level reader IAM."""
    print(f"\n[Step 3/8] Ensuring Artifact Registry repository '{repo_name}' in {region}...")
    headers = get_auth_headers(account=account)
    repo_res = f"projects/{project_id}/locations/{region}/repositories/{repo_name}"
    repo_url = f"https://artifactregistry.googleapis.com/v1/{repo_res}"
    r_get = requests.get(repo_url, headers=headers, timeout=20)
    if r_get.status_code == 200:
        print(f"  -> Artifact Registry repository '{repo_name}' already exists.")
    else:
        create_url = (
            f"https://artifactregistry.googleapis.com/v1/projects/{project_id}"
            f"/locations/{region}/repositories?repositoryId={repo_name}"
        )
        r_create = requests.post(
            create_url,
            headers=headers,
            json={
                "format": "DOCKER",
                "description": "Container repository for GECX + ADK Agent Runtime + Cloud Run Skincare API",
            },
            timeout=30,
        )
        r_create.raise_for_status()
        op = r_create.json()
        op_name = op.get("name", "")
        if op_name and not op.get("done"):
            for _ in range(30):
                r_op = requests.get(
                    f"https://artifactregistry.googleapis.com/v1/{op_name}",
                    headers=headers,
                    timeout=15,
                )
                if r_op.status_code == 200 and r_op.json().get("done"):
                    break
                time.sleep(2)
        print(f"  -> Created Artifact Registry repository '{repo_name}'.")

    # Bind roles/artifactregistry.reader directly on the repository for Vertex AI Agent Runtime P6SA
    reader_members = [
        f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-aiplatform.iam.gserviceaccount.com",
    ]
    requests.post(
        f"{repo_url}:setIamPolicy",
        headers=headers,
        json={
            "policy": {
                "bindings": [
                    {
                        "role": "roles/artifactregistry.reader",
                        "members": reader_members,
                    }
                ]
            }
        },
        timeout=20,
    )


def stage_python311_site_packages(cache_dir: Path = Path("/tmp/py311_adk_deps")) -> Path:
    """Stages Linux x86_64 Python 3.11 wheels for the Distroless Python 3.11 container."""
    marker = cache_dir / "google" / "adk"
    otel_marker = cache_dir / "opentelemetry" / "exporter" / "otlp"
    if marker.exists() and otel_marker.exists():
        print(f"  -> Reusing cached Python 3.11 site-packages at {cache_dir}")
        return cache_dir

    print(f"  -> Downloading Python 3.11 Linux x86_64 wheels into {cache_dir}...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    req_file = REPO_ROOT / "skincare_adk_agent" / "requirements.txt"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(cache_dir),
        "--python-version",
        "3.11",
        "--platform",
        "manylinux2014_x86_64",
        "--only-binary=:all:",
        "--upgrade",
        "-r",
        str(req_file),
    ]
    subprocess.run(cmd, check=True)
    return cache_dir


def deploy_cloud_run_skincare_api(
    project_id: str,
    project_number: str,
    region: str,
    service_name: str,
    image_uri: str,
    nano_banana_model: str = "gemini-3.1-flash-image",
    account: str | None = None,
) -> str:
    """Deploys Tier 3 Cloud Run Skincare API (`skincare-nano-banana-api`) and returns its HTTPS URL."""
    print(f"\n[Step 5/8] Deploying Tier 3 Cloud Run Skincare API '{service_name}' in {region}...")
    headers = get_auth_headers(account=account)
    parent = f"projects/{project_id}/locations/{region}"
    service_res_name = f"{parent}/services/{service_name}"
    get_url = f"https://run.googleapis.com/v2/{service_res_name}"

    service_body = {
        "ingress": "INGRESS_TRAFFIC_ALL",
        "template": {
            "scaling": {"minInstanceCount": 0, "maxInstanceCount": 5},
            "timeout": "120s",
            "containers": [
                {
                    "image": image_uri,
                    "resources": {
                        "limits": {"cpu": "2", "memory": "1Gi"},
                    },
                    "env": [
                        {"name": "GCP_PROJECT_ID", "value": project_id},
                        {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id},
                        {"name": "GOOGLE_CLOUD_LOCATION", "value": "global"},
                        {"name": "NANO_BANANA_MODEL", "value": nano_banana_model},
                    ],
                }
            ],
        },
    }

    r_check = requests.get(get_url, headers=headers, timeout=20)
    if r_check.status_code == 200:
        r_deploy = requests.patch(get_url, headers=headers, json=service_body, timeout=30)
    else:
        r_deploy = requests.post(
            f"https://run.googleapis.com/v2/{parent}/services?serviceId={service_name}",
            headers=headers,
            json=service_body,
            timeout=30,
        )
    r_deploy.raise_for_status()
    op_name = r_deploy.json().get("name", "")
    if op_name:
        for _ in range(60):
            r_op = requests.get(
                f"https://run.googleapis.com/v2/{op_name}", headers=headers, timeout=20
            )
            r_op.raise_for_status()
            if r_op.json().get("done"):
                break
            time.sleep(3)

    r_svc = requests.get(get_url, headers=headers, timeout=20)
    r_svc.raise_for_status()
    service_uri = r_svc.json()["uri"]

    # Bind Cloud Run invoker IAM policy for Vertex AI Agent Runtime and Compute Service Accounts
    invoker_members = [
        f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com",
    ]
    requests.post(
        f"https://run.googleapis.com/v2/{service_res_name}:setIamPolicy",
        headers=headers,
        json={
            "policy": {
                "bindings": [
                    {
                        "role": "roles/run.invoker",
                        "members": invoker_members,
                    }
                ]
            }
        },
        timeout=20,
    )
    print(f"  -> Cloud Run Skincare API deployed at: {service_uri}")
    return service_uri


def wait_for_reasoning_engine_op(
    region: str,
    op_name: str,
    headers: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    """Polls a Vertex AI ReasoningEngine LRO until completion. Returns `(succeeded, data)`."""
    url = f"https://{region}-aiplatform.googleapis.com/v1beta1/{op_name}"
    for _ in range(120):
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("done"):
            if "error" in data:
                return False, data["error"]
            return True, data.get("response", data)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for ReasoningEngine operation {op_name}")


def deploy_adk_agent_on_agent_runtime(
    project_id: str,
    project_number: str,
    region: str,
    display_name: str,
    image_uri: str,
    skincare_api_url: str,
    adk_model_name: str = "gemini-3.8-flash",
    account: str | None = None,
) -> str:
    """Deploys Tier 2 Google ADK Agent on Vertex AI Agent Runtime (`reasoningEngines`) and returns its ID."""
    print(
        f"\n[Step 6/8] Deploying Tier 2 ADK Agent '{display_name}' on Vertex AI Agent Runtime ({region})..."
    )
    headers = get_auth_headers(account=account)
    base_url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}"
        f"/locations/{region}/reasoningEngines"
    )

    r_list = requests.get(base_url, headers=headers, timeout=20)
    r_list.raise_for_status()
    engines = r_list.json().get("reasoningEngines", [])
    existing = next((e for e in engines if e.get("displayName") == display_name), None)

    def _build_env_list(engine_id: str = "") -> list[dict[str, str]]:
        env_vars = [
            {"name": "GCP_PROJECT_ID", "value": project_id},
            {"name": "GCP_PROJECT_NUMBER", "value": project_number},
            {"name": "AGENT_ENGINE_LOCATION", "value": region},
            {"name": "ADK_MODEL_LOCATION", "value": "global"},
            {"name": "ADK_MODEL_NAME", "value": adk_model_name},
            {"name": "SKINCARE_API_URL", "value": skincare_api_url},
            {"name": "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "value": "true"},
            {"name": "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "value": "SPAN_AND_EVENT"},
        ]
        if engine_id:
            env_vars.append({"name": "AGENT_ENGINE_ID", "value": engine_id})
        return env_vars

    class_methods = [
        {"name": "query", "api_mode": ""},
        {"name": "handle_a2a", "api_mode": "a2a_extension"},
    ]

    reasoning_engine_id = ""
    if existing is None:
        create_payload = {
            "displayName": display_name,
            "description": (
                "Google ADK Skincare Specialist Agent (gemini-3.8-flash) deployed on Vertex AI Agent Runtime "
                "with A2A v1.0 protocol, VertexAiSessionService, and OpenTelemetry Tracing."
            ),
            "spec": {
                "serviceAccount": f"{project_number}-compute@developer.gserviceaccount.com",
                "agentFramework": "google-adk",
                "classMethods": class_methods,
                "containerSpec": {"imageUri": image_uri},
                "deploymentSpec": {"env": _build_env_list()},
            },
        }
        for attempt in range(1, 4):
            print(f"  -> Creating ReasoningEngine (attempt {attempt}/3)...")
            r_create = requests.post(base_url, headers=headers, json=create_payload, timeout=30)
            r_create.raise_for_status()
            op_name = r_create.json()["name"]
            ok, res = wait_for_reasoning_engine_op(region, op_name, headers)
            if ok:
                reasoning_engine_id = res["name"].split("/")[-1]
                break
            print(
                f"  -> First-time Agent Runtime tenant warm-up returned: {res}. "
                f"Re-applying IAM bindings for newly provisioned P6SA and retrying in 60s..."
            )
            ensure_service_identities_and_iam(project_id, project_number, account=account)
            time.sleep(60)
        if not reasoning_engine_id:
            raise RuntimeError(f"Failed to create ReasoningEngine after 3 attempts: {res}")
    else:
        reasoning_engine_id = existing["name"].split("/")[-1]
        print(f"  -> Found existing ReasoningEngine ID: {reasoning_engine_id}")

    # Patch ReasoningEngine with its own AGENT_ENGINE_ID so VertexAiSessionService
    # and OpenTelemetry Agent Engine Tracing attach to the exact ReasoningEngine resource.
    print(
        f"  -> Configuring AGENT_ENGINE_ID={reasoning_engine_id} on ReasoningEngine "
        f"for managed Sessions & Traces..."
    )
    patch_url = (
        f"{base_url}/{reasoning_engine_id}"
        f"?updateMask=spec.containerSpec,spec.deploymentSpec.env,spec.classMethods"
    )
    patch_payload = {
        "spec": {
            "classMethods": class_methods,
            "containerSpec": {"imageUri": image_uri},
            "deploymentSpec": {"env": _build_env_list(reasoning_engine_id)},
        }
    }
    r_patch = requests.patch(patch_url, headers=headers, json=patch_payload, timeout=30)
    r_patch.raise_for_status()
    op_name = r_patch.json()["name"]
    ok, res = wait_for_reasoning_engine_op(region, op_name, headers)
    if not ok:
        raise RuntimeError(f"Failed to patch ReasoningEngine {reasoning_engine_id}: {res}")

    print(
        f"  -> Vertex AI Agent Runtime ADK Agent ready: "
        f"projects/{project_number}/locations/{region}/reasoningEngines/{reasoning_engine_id}"
    )
    return reasoning_engine_id


def main() -> None:
    load_dotenv_file()
    parser = argparse.ArgumentParser(
        description="End-to-End Provisioner for GECX -> ADK Agent Runtime -> Cloud Run Skincare API."
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GCP_PROJECT_ID", ""),
        help="Target Google Cloud Project ID",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_REGION", "europe-west1"),
        help="GCP region for Cloud Run, Artifact Registry, and Vertex AI Agent Runtime",
    )
    parser.add_argument(
        "--ces-location",
        default=os.environ.get("CES_LOCATION", "us"),
        help="GECX (ces.googleapis.com) location ('us', 'eu', or 'both')",
    )
    parser.add_argument(
        "--artifact-repo",
        default=os.environ.get("ARTIFACT_REPO_NAME", "gecx-skincare-repo"),
        help="Artifact Registry Docker repository name",
    )
    parser.add_argument(
        "--api-service-name",
        default=os.environ.get("SKINCARE_API_SERVICE_NAME", "skincare-nano-banana-api"),
        help="Cloud Run Skincare API service name",
    )
    parser.add_argument(
        "--adk-display-name",
        default=os.environ.get("ADK_AGENT_DISPLAY_NAME", "lumiere-skincare-adk-agent-runtime"),
        help="Vertex AI Agent Runtime (ReasoningEngine) display name",
    )
    parser.add_argument(
        "--skip-e2e-test",
        action="store_true",
        help="Skip running the live 4-turn E2E verification after deployment",
    )
    parser.add_argument("--account", default=None, help="Optional gcloud account email")
    args = parser.parse_args()

    if not args.project_id or args.project_id == "YOUR_GCP_PROJECT_ID":
        raise SystemExit(
            "Error: Please specify --project-id <YOUR_GCP_PROJECT_ID> or set GCP_PROJECT_ID in .env"
        )

    project_id = args.project_id
    region = args.region
    repo_name = args.artifact_repo
    account = args.account

    project_number = get_project_number(project_id, account=account)
    print("=" * 80)
    print(f"PROVISIONING 3-TIER GECX + ADK AGENT RUNTIME + CLOUD RUN BLUEPRINT")
    print(f"  Project ID     : {project_id}")
    print(f"  Project Number : {project_number}")
    print(f"  Region         : {region}")
    print(f"  GECX Location  : {args.ces_location}")
    print("=" * 80)

    # Step 1: Enable APIs
    enable_gcp_apis(project_id, project_number, account=account)

    # Step 2: Provision Service Identities & IAM
    ensure_service_identities_and_iam(project_id, project_number, account=account)

    # Step 3: Ensure Artifact Registry Repo
    ensure_artifact_registry_repo(project_id, project_number, region, repo_name, account=account)

    # Step 4: Stage Python 3.11 dependencies & push OCI images
    print("\n[Step 4/8] Building & pushing OCI container images to Artifact Registry...")
    site_packages_dir = stage_python311_site_packages()
    api_image_uri = build_and_push(
        project_id=project_id,
        region=region,
        repo_name=repo_name,
        target_image=args.api_service_name,
        target_tag="latest",
        entrypoint_module="skincare_api.app:app",
        site_packages_dir=site_packages_dir,
        app_mappings=[(REPO_ROOT / "skincare_api", "app/skincare_api")],
        account=account,
    )
    adk_image_uri = build_and_push(
        project_id=project_id,
        region=region,
        repo_name=repo_name,
        target_image="skincare-adk-agent",
        target_tag="latest",
        entrypoint_module="skincare_adk_agent.a2a_server:app",
        site_packages_dir=site_packages_dir,
        app_mappings=[
            (
                REPO_ROOT / "skincare_adk_agent" / "skincare_adk_agent",
                "app/skincare_adk_agent",
            )
        ],
        mount_from_image=args.api_service_name,
        account=account,
    )

    # Step 5: Deploy Cloud Run Skincare Nano Banana API
    skincare_api_url = deploy_cloud_run_skincare_api(
        project_id=project_id,
        project_number=project_number,
        region=region,
        service_name=args.api_service_name,
        image_uri=api_image_uri,
        account=account,
    )

    # Step 6: Deploy ADK Agent on Vertex AI Agent Runtime (`reasoningEngines`)
    reasoning_engine_id = deploy_adk_agent_on_agent_runtime(
        project_id=project_id,
        project_number=project_number,
        region=region,
        display_name=args.adk_display_name,
        image_uri=adk_image_uri,
        skincare_api_url=skincare_api_url,
        account=account,
    )

    # Re-apply Cloud Run invoker IAM in case the ReasoningEngine P6SA was created during Step 6
    ensure_service_identities_and_iam(project_id, project_number, account=account)

    # Step 7: Provision GECX Application(s)
    print(f"\n[Step 7/8] Provisioning Tier 1 GECX Application in '{args.ces_location}'...")
    locations = ["us", "eu"] if args.ces_location.lower() == "both" else [args.ces_location]
    gecx_summaries: list[dict[str, Any]] = []
    for loc in locations:
        summary = setup_gecx_blueprint_app(
            project_id=project_id,
            project_number=project_number,
            region=region,
            reasoning_engine_id=reasoning_engine_id,
            ces_location=loc,
            account=account,
        )
        gecx_summaries.append(summary)

    # Step 8: Run Live End-to-End Verification
    primary_gecx = gecx_summaries[0]
    e2e_result: dict[str, Any] = {}
    if not args.skip_e2e_test:
        print("\n[Step 8/8] Running 4-Turn Live End-to-End Verification Suite...")
        e2e_result = run_e2e_verification(
            project_id=project_id,
            ces_location=primary_gecx["location"],
            app_id=primary_gecx["gecx_app_id"],
            region=region,
            reasoning_engine_id=reasoning_engine_id,
            account=account,
        )

    final_output = {
        "project_id": project_id,
        "project_number": project_number,
        "region": region,
        "tier_3_cloud_run_skincare_api_url": skincare_api_url,
        "tier_2_vertex_agent_runtime_id": reasoning_engine_id,
        "tier_2_vertex_agent_runtime_resource": (
            f"projects/{project_number}/locations/{region}/reasoningEngines/{reasoning_engine_id}"
        ),
        "tier_2_vertex_console_url": (
            f"https://console.cloud.google.com/vertex-ai/agents/agent-engines"
            f"/locations/{region}/agent-engines/{reasoning_engine_id}/sessions?project={project_id}"
        ),
        "tier_1_gecx_apps": gecx_summaries,
        "e2e_verification": e2e_result,
    }
    print("\n" + "=" * 80)
    print("DEPLOYMENT & E2E VERIFICATION COMPLETE!")
    print("=" * 80)
    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    main()
