#!/usr/bin/env python3
"""End-to-End Provisioner & Verifier for the 3-Tier GECX + ADK Agent Runtime + Cloud Run Blueprint.

Supports both Single-Project deployment and 2-Project Cross-Boundary Least-Privilege deployment:
  - Agent Host Project (`--project-id`): Hosts Tier 3 Cloud Run Vision API + Tier 2 ADK Agent Runtime.
  - GECX Front-End Project (`--gecx-project-id`): Hosts Tier 1 GECX Multi-Flow App (`ces.googleapis.com`)
    authenticating to the single ReasoningEngine via a dedicated custom Service Account + 1-permission
    custom IAM role (`aiplatform.reasoningEngines.query`) bound exclusively at the ReasoningEngine resource level.
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

from oci_builder import build_and_push_oci, get_auth_headers, get_project_number, load_dotenv
from setup_gecx import setup_gecx_app
from test_e2e import run_e2e_test

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
    "logging.googleapis.com",
    "monitoring.googleapis.com",
]


def enable_apis(project_number: str, headers: dict[str, str], apis: list[str] | None = None) -> None:
    target_apis = apis or REQUIRED_APIS
    print(f"[1/7] Enabling required GCP APIs on project #{project_number}...")
    r = requests.post(
        f"https://serviceusage.googleapis.com/v1/projects/{project_number}/services:batchEnable",
        headers=headers,
        json={"serviceIds": target_apis},
        timeout=60,
    )
    if r.status_code == 200 and (op_name := r.json().get("name")) and not r.json().get("done"):
        for _ in range(40):
            if requests.get(f"https://serviceusage.googleapis.com/v1/{op_name}", headers=headers, timeout=20).json().get("done"):
                break
            time.sleep(3)


def configure_iam(project_id: str, project_number: str, headers: dict[str, str]) -> None:
    print(f"[2/7] Provisioning Service Identities and runtime IAM bindings in Agent Host project '{project_id}'...")
    for svc in ("aiplatform.googleapis.com", "run.googleapis.com", "ces.googleapis.com"):
        requests.post(
            f"https://serviceusage.googleapis.com/v1beta1/projects/{project_number}/services/{svc}:generateServiceIdentity",
            headers=headers,
            json={},
            timeout=20,
        )
    members = [
        f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com",
        f"serviceAccount:service-{project_number}@gcp-sa-aiplatform.iam.gserviceaccount.com",
    ]
    roles = [
        "roles/aiplatform.user",
        "roles/run.invoker",
        "roles/cloudtrace.agent",
        "roles/telemetry.tracesWriter",
        "roles/logging.logWriter",
        "roles/monitoring.metricWriter",
        "roles/serviceusage.serviceUsageConsumer",
        "roles/artifactregistry.reader",
    ]
    policy = requests.post(
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:getIamPolicy",
        headers=headers,
        json={},
        timeout=20,
    ).json()
    bindings = policy.get("bindings", [])
    for member in members:
        updated = [dict(b, members=list(b.get("members", []))) for b in bindings]
        changed = False
        for role in roles:
            entry = next((b for b in updated if b.get("role") == role), None)
            if entry is None:
                entry = {"role": role, "members": []}
                updated.append(entry)
            if member not in entry["members"]:
                entry["members"].append(member)
                changed = True
        if changed:
            r_set = requests.post(
                f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:setIamPolicy",
                headers=headers,
                json={"policy": {**policy, "bindings": updated}},
                timeout=20,
            )
            if r_set.status_code == 200:
                policy = r_set.json()
                bindings = policy.get("bindings", [])


def configure_least_privilege_cross_project_auth(
    agent_host_project_id: str,
    gecx_project_id: str,
    gecx_project_number: str,
    region: str,
    engine_id: str,
    headers: dict[str, str],
) -> str:
    """Enforces the 3-part Maximum Least-Privilege Security Policy for GECX -> ADK Agent Runtime:
    1. Dedicated Custom Service Account (`gecx-skincare-caller`) in the GECX Project, impersonated
       exclusively by the GECX Service Agent (`service-<GECX_PROJ_NUM>@gcp-sa-ces.iam.gserviceaccount.com`).
    2. Single-permission Custom IAM Role (`singleAgentInvoker` with `aiplatform.reasoningEngines.query` only)
       in the Agent Host Project.
    3. Resource-level IAM binding (`:setIamPolicy`) on the single `ReasoningEngine` resource ONLY
       (zero project-level IAM bindings for GECX in the Agent Host Project).
    """
    print(f"  -> Configuring Maximum Least-Privilege Security ({gecx_project_id} -> {agent_host_project_id})...")
    requests.post(
        f"https://serviceusage.googleapis.com/v1beta1/projects/{gecx_project_number}/services/ces.googleapis.com:generateServiceIdentity",
        headers=headers,
        json={},
        timeout=20,
    )

    sa_account_id = "gecx-skincare-caller"
    caller_sa_email = f"{sa_account_id}@{gecx_project_id}.iam.gserviceaccount.com"
    sa_resource = f"projects/{gecx_project_id}/serviceAccounts/{caller_sa_email}"
    if requests.get(f"https://iam.googleapis.com/v1/{sa_resource}", headers=headers, timeout=20).status_code != 200:
        r_sa = requests.post(
            f"https://iam.googleapis.com/v1/projects/{gecx_project_id}/serviceAccounts",
            headers=headers,
            json={
                "accountId": sa_account_id,
                "serviceAccount": {
                    "displayName": "GECX Dedicated Caller for Skincare ADK Agent",
                    "description": "Least-privilege custom SA used by GECX RemoteAgentTool to invoke a single ReasoningEngine.",
                },
            },
            timeout=20,
        )
        r_sa.raise_for_status()
        time.sleep(5)

    # 1. Allow ONLY the GECX project's CES Service Agent to mint tokens for `gecx-skincare-caller`
    ces_p4sa = f"serviceAccount:service-{gecx_project_number}@gcp-sa-ces.iam.gserviceaccount.com"
    r_sa_iam = requests.post(
        f"https://iam.googleapis.com/v1/{sa_resource}:setIamPolicy",
        headers=headers,
        json={
            "policy": {
                "bindings": [
                    {
                        "role": "roles/iam.serviceAccountTokenCreator",
                        "members": [ces_p4sa],
                    }
                ]
            }
        },
        timeout=20,
    )
    r_sa_iam.raise_for_status()

    # 2. Create/update single-permission custom IAM role (`aiplatform.reasoningEngines.query` only) in Agent Host Project
    role_id = "singleAgentInvoker"
    role_name = f"projects/{agent_host_project_id}/roles/{role_id}"
    role_body = {
        "title": "Single Reasoning Engine Invoker",
        "description": "Least-privilege role granting ONLY aiplatform.reasoningEngines.query on a specific ReasoningEngine.",
        "includedPermissions": ["aiplatform.reasoningEngines.query"],
        "stage": "GA",
    }
    r_role_get = requests.get(f"https://iam.googleapis.com/v1/{role_name}", headers=headers, timeout=20)
    if r_role_get.status_code != 200:
        r_role = requests.post(
            f"https://iam.googleapis.com/v1/projects/{agent_host_project_id}/roles",
            headers=headers,
            json={"roleId": role_id, "role": role_body},
            timeout=20,
        )
        r_role.raise_for_status()
    else:
        requests.patch(
            f"https://iam.googleapis.com/v1/{role_name}?updateMask=title,description,includedPermissions,stage",
            headers=headers,
            json=role_body,
            timeout=20,
        ).raise_for_status()

    # 3. Bind `singleAgentInvoker` ONLY on the specific ReasoningEngine resource (never project-wide)
    re_iam_url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{agent_host_project_id}/locations/{region}/reasoningEngines/{engine_id}:setIamPolicy"
    )
    r_re_iam = requests.post(
        re_iam_url,
        headers=headers,
        json={
            "policy": {
                "bindings": [
                    {
                        "role": role_name,
                        "members": [f"serviceAccount:{caller_sa_email}"],
                    }
                ]
            }
        },
        timeout=20,
    )
    r_re_iam.raise_for_status()

    # Ensure zero project-level IAM bindings exist in Agent Host Project for the GECX caller SA or CES P4SA
    host_policy = requests.post(
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{agent_host_project_id}:getIamPolicy",
        headers=headers,
        json={},
        timeout=20,
    ).json()
    disallowed_project_members = {f"serviceAccount:{caller_sa_email}", ces_p4sa}
    cleaned_bindings = []
    modified = False
    for b in host_policy.get("bindings", []):
        kept = [m for m in b.get("members", []) if m not in disallowed_project_members]
        if len(kept) != len(b.get("members", [])):
            modified = True
        if kept:
            cleaned_bindings.append({**b, "members": kept})
    if modified:
        requests.post(
            f"https://cloudresourcemanager.googleapis.com/v1/projects/{agent_host_project_id}:setIamPolicy",
            headers=headers,
            json={"policy": {**host_policy, "bindings": cleaned_bindings}},
            timeout=20,
        )

    print(f"  -> Bound '{role_name}' exclusively on ReasoningEngine '{engine_id}' for '{caller_sa_email}'")
    return caller_sa_email


def ensure_artifact_repo(project_id: str, project_number: str, region: str, repo_name: str, headers: dict[str, str]) -> None:
    print(f"[3/7] Ensuring Artifact Registry repository '{repo_name}' in {region}...")
    repo_url = f"https://artifactregistry.googleapis.com/v1/projects/{project_id}/locations/{region}/repositories/{repo_name}"
    if requests.get(repo_url, headers=headers, timeout=20).status_code != 200:
        r_c = requests.post(
            f"https://artifactregistry.googleapis.com/v1/projects/{project_id}/locations/{region}/repositories?repositoryId={repo_name}",
            headers=headers,
            json={"format": "DOCKER"},
            timeout=30,
        )
        r_c.raise_for_status()
        if (op_name := r_c.json().get("name")) and not r_c.json().get("done"):
            for _ in range(30):
                if requests.get(f"https://artifactregistry.googleapis.com/v1/{op_name}", headers=headers, timeout=15).json().get("done"):
                    break
                time.sleep(2)
    requests.post(
        f"{repo_url}:setIamPolicy",
        headers=headers,
        json={
            "policy": {
                "bindings": [
                    {
                        "role": "roles/artifactregistry.reader",
                        "members": [
                            f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com",
                            f"serviceAccount:service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com",
                            f"serviceAccount:service-{project_number}@gcp-sa-aiplatform.iam.gserviceaccount.com",
                        ],
                    }
                ]
            }
        },
        timeout=20,
    )


def stage_py311_deps(cache_dir: Path = Path("/tmp/py311_adk_deps")) -> Path:
    if (cache_dir / "google" / "adk").exists() and (cache_dir / "opentelemetry" / "exporter" / "otlp").exists():
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--target", str(cache_dir),
            "--python-version", "3.11", "--platform", "manylinux2014_x86_64",
            "--only-binary=:all:", "--upgrade", "-r", str(REPO_ROOT / "skincare_adk_agent" / "requirements.txt"),
        ],
        check=True,
    )
    return cache_dir


def deploy_cloud_run_api(
    project_id: str, project_number: str, region: str, service_name: str, image_uri: str, model_name: str, headers: dict[str, str]
) -> str:
    print(f"[5/7] Deploying Tier 3 Cloud Run Skincare Vision API '{service_name}' ({model_name})...")
    parent = f"projects/{project_id}/locations/{region}"
    svc_url = f"https://run.googleapis.com/v2/{parent}/services/{service_name}"
    body = {
        "ingress": "INGRESS_TRAFFIC_ALL",
        "template": {
            "scaling": {"minInstanceCount": 0, "maxInstanceCount": 5},
            "timeout": "120s",
            "containers": [
                {
                    "image": image_uri,
                    "resources": {"limits": {"cpu": "2", "memory": "1Gi"}},
                    "env": [
                        {"name": "GCP_PROJECT_ID", "value": project_id},
                        {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id},
                        {"name": "GOOGLE_CLOUD_LOCATION", "value": "global"},
                        {"name": "VISION_MODEL", "value": model_name},
                    ],
                }
            ],
        },
    }
    r = (
        requests.patch(svc_url, headers=headers, json=body, timeout=30)
        if requests.get(svc_url, headers=headers, timeout=20).status_code == 200
        else requests.post(f"https://run.googleapis.com/v2/{parent}/services?serviceId={service_name}", headers=headers, json=body, timeout=30)
    )
    r.raise_for_status()
    if op_name := r.json().get("name"):
        for _ in range(60):
            if requests.get(f"https://run.googleapis.com/v2/{op_name}", headers=headers, timeout=20).json().get("done"):
                break
            time.sleep(3)
    uri = requests.get(svc_url, headers=headers, timeout=20).json()["uri"]
    requests.post(
        f"{svc_url}:setIamPolicy",
        headers=headers,
        json={
            "policy": {
                "bindings": [
                    {
                        "role": "roles/run.invoker",
                        "members": [
                            f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com",
                            f"serviceAccount:service-{project_number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com",
                        ],
                    }
                ]
            }
        },
        timeout=20,
    )
    print(f"  -> Cloud Run API live at: {uri}")
    return uri


def _wait_re_op(region: str, op_name: str, headers: dict[str, str]) -> tuple[bool, dict[str, Any]]:
    for _ in range(120):
        data = requests.get(f"https://{region}-aiplatform.googleapis.com/v1beta1/{op_name}", headers=headers, timeout=20).json()
        if data.get("done"):
            return ("error" not in data), data.get("error") or data.get("response", data)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {op_name}")


def deploy_adk_agent_runtime(
    project_id: str, project_number: str, region: str, display_name: str, image_uri: str, api_url: str, model_name: str, headers: dict[str, str]
) -> str:
    print(f"[6/7] Deploying Tier 2 ADK Agent '{display_name}' on Vertex AI Agent Runtime...")
    base_url = f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/reasoningEngines"
    engines = requests.get(base_url, headers=headers, timeout=20).json().get("reasoningEngines", [])
    existing = next((e for e in engines if e.get("displayName") == display_name), None)

    def _env_list(engine_id: str = "") -> list[dict[str, str]]:
        env = [
            {"name": "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "value": "true"},
            {"name": "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "value": "true"},
            {"name": "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "value": "true"},
            {"name": "GCP_PROJECT_ID", "value": project_id},
            {"name": "GCP_PROJECT_NUMBER", "value": project_number},
            {"name": "AGENT_ENGINE_LOCATION", "value": region},
            {"name": "ADK_MODEL_NAME", "value": model_name},
            {"name": "SKINCARE_API_URL", "value": api_url},
        ]
        if engine_id:
            env.append({"name": "AGENT_ENGINE_ID", "value": engine_id})
        return env

    class_methods = [{"name": "query", "api_mode": ""}, {"name": "handle_a2a", "api_mode": "a2a_extension"}]
    engine_desc = (
        "Google ADK Skincare Specialist Agent (gemini-3.8-flash) deployed on Vertex AI "
        "Agent Runtime calling the Cloud Run Skincare Vision API (gemini-3.8-flash)."
    )
    engine_id = existing["name"].split("/")[-1] if existing else ""

    if not engine_id:
        payload = {
            "displayName": display_name,
            "description": engine_desc,
            "spec": {
                "serviceAccount": f"{project_number}-compute@developer.gserviceaccount.com",
                "agentFramework": "google-adk",
                "classMethods": class_methods,
                "containerSpec": {"imageUri": image_uri},
                "deploymentSpec": {"env": _env_list()},
            },
        }
        for attempt in range(1, 4):
            r_c = requests.post(base_url, headers=headers, json=payload, timeout=30)
            r_c.raise_for_status()
            ok, res = _wait_re_op(region, r_c.json()["name"], headers)
            if ok:
                engine_id = res["name"].split("/")[-1]
                break
            print(f"  -> Tenant warm-up retry ({attempt}/3); re-applying IAM and waiting 60s...")
            configure_iam(project_id, project_number, headers)
            time.sleep(60)
        if not engine_id:
            raise RuntimeError(f"ReasoningEngine creation failed: {res}")

    r_p = requests.patch(
        f"{base_url}/{engine_id}?updateMask=description,spec.containerSpec,spec.deploymentSpec.env,spec.classMethods",
        headers=headers,
        json={
            "description": engine_desc,
            "spec": {
                "classMethods": class_methods,
                "containerSpec": {"imageUri": image_uri},
                "deploymentSpec": {"env": _env_list(engine_id)},
            },
        },
        timeout=30,
    )
    r_p.raise_for_status()
    ok, res = _wait_re_op(region, r_p.json()["name"], headers)
    if not ok:
        raise RuntimeError(f"ReasoningEngine update failed: {res}")
    print(f"  -> ADK Agent Runtime live (Telemetry Enabled): projects/{project_number}/locations/{region}/reasoningEngines/{engine_id}")
    return engine_id


def teardown_gecx_apps(project_id: str, headers: dict[str, str]) -> None:
    for loc in ("us", "eu"):
        r_apps = requests.get(f"https://ces.googleapis.com/v1beta/projects/{project_id}/locations/{loc}/apps", headers=headers, timeout=20)
        for app in r_apps.json().get("apps", []):
            if "Lumiere" not in app.get("displayName", ""):
                continue
            app_name = app["name"]
            for dep in requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/deployments", headers=headers, timeout=15).json().get("deployments", []):
                requests.delete(f"https://ces.googleapis.com/v1beta/{dep['name']}", headers=headers, timeout=15)
            for ver in requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/versions", headers=headers, timeout=15).json().get("appVersions", []):
                requests.delete(f"https://ces.googleapis.com/v1beta/{ver['name']}", headers=headers, timeout=15)
            r_del = requests.delete(f"https://ces.googleapis.com/v1beta/{app_name}?force=true", headers=headers, timeout=20)
            print(f"  -> Deleted GECX App {app_name} (HTTP {r_del.status_code})")


def teardown_stack(
    project_id: str,
    gecx_project_id: str,
    region: str,
    repo_name: str,
    service_name: str,
    adk_display_name: str,
    headers: dict[str, str],
) -> None:
    """Deletes all GECX, Vertex AI Agent Runtime, Cloud Run, and Artifact Registry resources for this blueprint."""
    for p in dict.fromkeys([gecx_project_id, project_id]):
        print(f"Tearing down GECX apps in project '{p}'...")
        teardown_gecx_apps(p, headers)

    print(f"Tearing down Agent Host resources in project '{project_id}'...")
    r_re = requests.get(f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/reasoningEngines", headers=headers, timeout=20)
    for re_obj in r_re.json().get("reasoningEngines", []):
        if re_obj.get("displayName") == adk_display_name or "lumiere" in re_obj.get("displayName", "").lower():
            re_name = re_obj["name"]
            r_del_re = requests.delete(f"https://{region}-aiplatform.googleapis.com/v1beta1/{re_name}?force=true", headers=headers, timeout=20)
            print(f"  -> Deleted ReasoningEngine {re_name} (HTTP {r_del_re.status_code})")

    r_del_cr = requests.delete(f"https://run.googleapis.com/v2/projects/{project_id}/locations/{region}/services/{service_name}", headers=headers, timeout=20)
    print(f"  -> Deleted Cloud Run service {service_name} (HTTP {r_del_cr.status_code})")

    r_del_ar = requests.delete(f"https://artifactregistry.googleapis.com/v1/projects/{project_id}/locations/{region}/repositories/{repo_name}", headers=headers, timeout=20)
    print(f"  -> Deleted Artifact Registry repository {repo_name} (HTTP {r_del_ar.status_code})")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Deploy or Teardown GECX + ADK Agent Runtime + Cloud Run Blueprint.")
    parser.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID", ""), help="Agent Host Project ID (Cloud Run + ADK Agent Runtime)")
    parser.add_argument("--gecx-project-id", default=os.environ.get("GECX_PROJECT_ID", ""), help="Optional separate GECX Front-End Project ID (defaults to --project-id)")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "europe-west1"))
    parser.add_argument("--ces-location", default=os.environ.get("CES_LOCATION", "us"))
    parser.add_argument("--artifact-repo", default=os.environ.get("ARTIFACT_REPO_NAME", "gecx-skincare-repo"))
    parser.add_argument("--api-service-name", default=os.environ.get("SKINCARE_API_SERVICE_NAME", "skincare-vision-api"))
    parser.add_argument("--adk-display-name", default=os.environ.get("ADK_AGENT_DISPLAY_NAME", "lumiere-skincare-adk-agent-runtime"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemini-3.8-flash"))
    parser.add_argument("--teardown", action="store_true", help="Delete all blueprint resources from the project(s)")
    parser.add_argument("--account", default=None)
    args = parser.parse_args()

    if not args.project_id or args.project_id == "YOUR_GCP_PROJECT_ID":
        raise SystemExit("Error: specify --project-id <AGENT_HOST_PROJECT_ID> or set GCP_PROJECT_ID in .env")

    gecx_project_id = args.gecx_project_id or args.project_id
    headers = get_auth_headers(args.account)
    if args.teardown:
        teardown_stack(args.project_id, gecx_project_id, args.region, args.artifact_repo, args.api_service_name, args.adk_display_name, headers)
        return

    project_number = get_project_number(args.project_id, args.account)
    gecx_project_number = get_project_number(gecx_project_id, args.account) if gecx_project_id != args.project_id else project_number

    enable_apis(project_number, headers)
    if gecx_project_id != args.project_id:
        enable_apis(gecx_project_number, headers, apis=["serviceusage.googleapis.com", "cloudresourcemanager.googleapis.com", "iam.googleapis.com", "ces.googleapis.com", "logging.googleapis.com"])
        # Remove any legacy GECX apps from the Agent Host project so roles are cleanly separated
        teardown_gecx_apps(args.project_id, headers)

    configure_iam(args.project_id, project_number, headers)
    ensure_artifact_repo(args.project_id, project_number, args.region, args.artifact_repo, headers)

    print("[4/7] Building and pushing OCI container images to Artifact Registry...")
    sp_dir = stage_py311_deps()
    api_img = build_and_push_oci(
        args.project_id, args.region, args.artifact_repo, args.api_service_name,
        "skincare_api.app:app", sp_dir, [(REPO_ROOT / "skincare_api" / "app.py", "app/skincare_api/app.py")], account=args.account,
    )
    adk_img = build_and_push_oci(
        args.project_id, args.region, args.artifact_repo, "skincare-adk-agent",
        "skincare_adk_agent.server:app", sp_dir,
        [(REPO_ROOT / "skincare_adk_agent" / "agent.py", "app/skincare_adk_agent/agent.py"),
         (REPO_ROOT / "skincare_adk_agent" / "server.py", "app/skincare_adk_agent/server.py")],
        mount_from=args.api_service_name, account=args.account,
    )

    api_url = deploy_cloud_run_api(args.project_id, project_number, args.region, args.api_service_name, api_img, args.model, get_auth_headers(args.account))
    engine_id = deploy_adk_agent_runtime(args.project_id, project_number, args.region, args.adk_display_name, adk_img, api_url, args.model, get_auth_headers(args.account))

    caller_sa = configure_least_privilege_cross_project_auth(
        args.project_id, gecx_project_id, gecx_project_number, args.region, engine_id, get_auth_headers(args.account)
    )

    print(f"[7/7] Provisioning Tier 1 GECX Application in '{gecx_project_id}' ({args.ces_location}) and running E2E verification...")
    locations = ["us", "eu"] if args.ces_location.lower() == "both" else [args.ces_location]
    gecx_list = [
        setup_gecx_app(gecx_project_id, project_number, args.region, engine_id, loc, caller_service_account=caller_sa, account=args.account)
        for loc in locations
    ]
    e2e_info = run_e2e_test(
        gecx_project_id,
        gecx_list[0]["location"],
        gecx_list[0]["gecx_app_id"],
        args.region,
        engine_id,
        args.account,
        agent_host_project_id=args.project_id,
    )

    print("\nDEPLOYMENT & E2E VERIFICATION SUCCEEDED:")
    print(
        json.dumps(
            {
                "agent_host_project_id": args.project_id,
                "gecx_frontend_project_id": gecx_project_id,
                "cloud_run_api": api_url,
                "agent_runtime_id": engine_id,
                "least_privilege_caller_sa": caller_sa,
                "least_privilege_custom_role": f"projects/{args.project_id}/roles/singleAgentInvoker",
                "gecx_apps": gecx_list,
                "e2e_test": e2e_info,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
