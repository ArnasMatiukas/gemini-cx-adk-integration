#!/usr/bin/env python3
"""End-to-End Provisioner & Verifier for the 3-Tier GECX + ADK Agent Runtime + Cloud Run Blueprint."""

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
]


def enable_apis(project_number: str, headers: dict[str, str]) -> None:
    print("[1/7] Enabling required GCP APIs...")
    r = requests.post(
        f"https://serviceusage.googleapis.com/v1/projects/{project_number}/services:batchEnable",
        headers=headers,
        json={"serviceIds": REQUIRED_APIS},
        timeout=60,
    )
    if r.status_code == 200 and (op_name := r.json().get("name")) and not r.json().get("done"):
        for _ in range(40):
            if requests.get(f"https://serviceusage.googleapis.com/v1/{op_name}", headers=headers, timeout=20).json().get("done"):
                break
            time.sleep(3)


def configure_iam(project_id: str, project_number: str, headers: dict[str, str]) -> None:
    print("[2/7] Provisioning Service Identities and project IAM bindings...")
    for svc in ("aiplatform.googleapis.com", "ces.googleapis.com", "run.googleapis.com"):
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
        f"serviceAccount:service-{project_number}@gcp-sa-ces.iam.gserviceaccount.com",
    ]
    roles = [
        "roles/aiplatform.user",
        "roles/run.invoker",
        "roles/cloudtrace.agent",
        "roles/telemetry.tracesWriter",
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
    print(f"  -> ADK Agent Runtime live: projects/{project_number}/locations/{region}/reasoningEngines/{engine_id}")
    return engine_id


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Deploy GECX + ADK Agent Runtime + Cloud Run Blueprint.")
    parser.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID", ""))
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "europe-west1"))
    parser.add_argument("--ces-location", default=os.environ.get("CES_LOCATION", "us"))
    parser.add_argument("--artifact-repo", default=os.environ.get("ARTIFACT_REPO_NAME", "gecx-skincare-repo"))
    parser.add_argument("--api-service-name", default=os.environ.get("SKINCARE_API_SERVICE_NAME", "skincare-vision-api"))
    parser.add_argument("--adk-display-name", default=os.environ.get("ADK_AGENT_DISPLAY_NAME", "lumiere-skincare-adk-agent-runtime"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemini-3.8-flash"))
    parser.add_argument("--account", default=None)
    args = parser.parse_args()

    if not args.project_id or args.project_id == "YOUR_GCP_PROJECT_ID":
        raise SystemExit("Error: specify --project-id <YOUR_GCP_PROJECT_ID> or set GCP_PROJECT_ID in .env")

    headers = get_auth_headers(args.account)
    project_number = get_project_number(args.project_id, args.account)

    enable_apis(project_number, headers)
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

    print(f"[7/7] Provisioning Tier 1 GECX Application ({args.ces_location}) and running E2E verification...")
    locations = ["us", "eu"] if args.ces_location.lower() == "both" else [args.ces_location]
    gecx_list = [
        setup_gecx_app(args.project_id, project_number, args.region, engine_id, loc, args.account)
        for loc in locations
    ]
    e2e_info = run_e2e_test(args.project_id, gecx_list[0]["location"], gecx_list[0]["gecx_app_id"], args.region, engine_id, args.account)

    print("\nDEPLOYMENT & E2E VERIFICATION SUCCEEDED:")
    print(json.dumps({"cloud_run_api": api_url, "agent_runtime_id": engine_id, "gecx_apps": gecx_list, "e2e_test": e2e_info}, indent=2))


if __name__ == "__main__":
    main()
