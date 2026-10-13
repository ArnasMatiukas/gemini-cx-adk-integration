#!/usr/bin/env python3
"""Build and push a Python 3.11 Distroless OCI image directly to Artifact Registry via HTTP V2 API.

Works in any customer environment without requiring a local Docker daemon or Cloud Build:
1. Fetches the 43 base layers of `gcr.io/distroless/python3-debian12:latest` (`linux/amd64`)
   directly from `https://gcr.io/v2/distroless/python3-debian12/...` (and caches/mounts them in the
   customer's Artifact Registry repository).
2. Builds deterministic tar+gzip layers for `/app/site-packages` and `/app/<package>`.
3. Uploads the layers, OCI config, and OCI manifest to Artifact Registry.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gcp_auth import get_auth_headers, load_dotenv_file

DISTROLESS_REPO = "distroless/python3-debian12"
DISTROLESS_TAG = "latest"
LOCAL_BASE_CACHE = Path("/tmp/distroless_py311_cache")


def create_layer_blobs(dir_mappings: list[tuple[Path, str]]) -> tuple[bytes, str, str]:
    """Creates an uncompressed tar and gzipped tar from `(local_path, container_arcname)` pairs.

    Returns:
      (compressed_bytes, compressed_sha256_digest, uncompressed_sha256_diff_id)
    """
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        for src_path, arc_prefix in dir_mappings:
            if src_path.is_file():
                ti = tf.gettarinfo(str(src_path), arcname=arc_prefix)
                ti.uid = 0
                ti.gid = 0
                ti.uname = "root"
                ti.gname = "root"
                ti.mtime = 0
                with open(src_path, "rb") as f:
                    tf.addfile(ti, f)
            else:
                for root, dirs, files in os.walk(src_path):
                    dirs[:] = [
                        d
                        for d in sorted(dirs)
                        if d not in ("__pycache__", ".pytest_cache", ".git")
                    ]
                    for file in sorted(files):
                        if file.endswith((".pyc", ".pyo")):
                            continue
                        full_p = Path(root) / file
                        if not full_p.is_file() or full_p.is_symlink():
                            continue
                        rel_p = full_p.relative_to(src_path)
                        arcname = f"{arc_prefix.rstrip('/')}/{rel_p}"
                        ti = tf.gettarinfo(str(full_p), arcname=arcname)
                        ti.uid = 0
                        ti.gid = 0
                        ti.uname = "root"
                        ti.gname = "root"
                        ti.mtime = 0
                        with open(full_p, "rb") as f:
                            tf.addfile(ti, f)

    uncompressed_bytes = tar_buf.getvalue()
    diff_id = "sha256:" + hashlib.sha256(uncompressed_bytes).hexdigest()

    gz_buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gz_buf, mtime=0) as gz:
        gz.write(uncompressed_bytes)
    compressed_bytes = gz_buf.getvalue()
    layer_digest = "sha256:" + hashlib.sha256(compressed_bytes).hexdigest()
    return compressed_bytes, layer_digest, diff_id


def fetch_public_distroless_base(session: requests.Session) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetches the amd64 manifest and config of `gcr.io/distroless/python3-debian12:latest`."""
    LOCAL_BASE_CACHE.mkdir(parents=True, exist_ok=True)
    man_cache = LOCAL_BASE_CACHE / "manifest.json"
    cfg_cache = LOCAL_BASE_CACHE / "config.json"
    if man_cache.exists() and cfg_cache.exists():
        return json.loads(man_cache.read_text()), json.loads(cfg_cache.read_text())

    accept_headers = {
        "Accept": (
            "application/vnd.oci.image.index.v1+json, "
            "application/vnd.docker.distribution.manifest.list.v2+json, "
            "application/vnd.oci.image.manifest.v1+json, "
            "application/vnd.docker.distribution.manifest.v2+json"
        )
    }
    r_idx = session.get(
        f"https://gcr.io/v2/{DISTROLESS_REPO}/manifests/{DISTROLESS_TAG}",
        headers=accept_headers,
        timeout=30,
    )
    r_idx.raise_for_status()
    idx_data = r_idx.json()

    if "manifests" in idx_data:
        amd64_entry = next(
            m
            for m in idx_data["manifests"]
            if m.get("platform", {}).get("architecture") == "amd64"
            and m.get("platform", {}).get("os") == "linux"
        )
        r_man = session.get(
            f"https://gcr.io/v2/{DISTROLESS_REPO}/manifests/{amd64_entry['digest']}",
            headers=accept_headers,
            timeout=30,
        )
        r_man.raise_for_status()
        manifest = r_man.json()
    else:
        manifest = idx_data

    cfg_digest = manifest["config"]["digest"]
    r_cfg = session.get(
        f"https://gcr.io/v2/{DISTROLESS_REPO}/blobs/{cfg_digest}",
        timeout=30,
    )
    r_cfg.raise_for_status()
    config = r_cfg.json()

    man_cache.write_text(json.dumps(manifest))
    cfg_cache.write_text(json.dumps(config))
    return manifest, config


def get_distroless_layer_bytes(session: requests.Session, digest: str) -> bytes:
    """Downloads a base layer blob from `gcr.io/distroless/python3-debian12` with local disk cache."""
    LOCAL_BASE_CACHE.mkdir(parents=True, exist_ok=True)
    safe_name = digest.replace(":", "_")
    cache_file = LOCAL_BASE_CACHE / safe_name
    if cache_file.exists():
        return cache_file.read_bytes()
    r = session.get(f"https://gcr.io/v2/{DISTROLESS_REPO}/blobs/{digest}", timeout=60)
    r.raise_for_status()
    cache_file.write_bytes(r.content)
    return r.content


def ensure_blob_uploaded(
    session: requests.Session,
    headers: dict[str, str],
    registry_host: str,
    project_id: str,
    repo_name: str,
    target_image: str,
    blob_bytes: bytes | None,
    digest: str,
    mount_from_image: str | None = None,
) -> None:
    """Checks if a blob exists in target_image repo; mounts or uploads if missing."""
    base_url = f"https://{registry_host}/v2/{project_id}/{repo_name}/{target_image}"
    r_head = session.head(f"{base_url}/blobs/{digest}", headers=headers, timeout=30)
    if r_head.status_code == 200:
        return

    if mount_from_image and mount_from_image != target_image:
        mount_url = (
            f"{base_url}/blobs/uploads/?mount={digest}"
            f"&from={project_id}/{repo_name}/{mount_from_image}"
        )
        r_mount = session.post(mount_url, headers=headers, timeout=30)
        if r_mount.status_code == 201:
            return

    if blob_bytes is None:
        blob_bytes = get_distroless_layer_bytes(session, digest)

    r_init = session.post(f"{base_url}/blobs/uploads/", headers=headers, timeout=30)
    r_init.raise_for_status()
    upload_url = r_init.headers["Location"]
    if not upload_url.startswith("http"):
        upload_url = f"https://{registry_host}{upload_url}"
    sep = "&" if "?" in upload_url else "?"
    put_url = f"{upload_url}{sep}digest={digest}"
    put_headers = {**headers, "Content-Type": "application/octet-stream"}
    r_put = session.put(put_url, headers=put_headers, data=blob_bytes, timeout=300)
    r_put.raise_for_status()


def build_and_push(
    project_id: str,
    region: str,
    repo_name: str,
    target_image: str,
    target_tag: str,
    entrypoint_module: str,
    site_packages_dir: Path,
    app_mappings: list[tuple[Path, str]],
    extra_env: list[str] | None = None,
    mount_from_image: str | None = None,
    account: str | None = None,
) -> str:
    """Builds and pushes an OCI image to Artifact Registry."""
    registry_host = f"{region}-docker.pkg.dev"
    headers = get_auth_headers(account=account, content_type=None)
    session = requests.Session()

    base_manifest, base_cfg = fetch_public_distroless_base(session)
    distroless_layers = base_manifest["layers"]
    distroless_diff_ids = base_cfg["rootfs"]["diff_ids"]

    print(
        f"[1/4] Ensuring {len(distroless_layers)} distroless Python 3.11 base layers in "
        f"{registry_host}/{project_id}/{repo_name}/{target_image}..."
    )
    for layer in distroless_layers:
        ensure_blob_uploaded(
            session=session,
            headers=headers,
            registry_host=registry_host,
            project_id=project_id,
            repo_name=repo_name,
            target_image=target_image,
            blob_bytes=None,
            digest=layer["digest"],
            mount_from_image=mount_from_image,
        )

    print(f"[2/4] Packing site-packages layer from {site_packages_dir}...")
    sp_bytes, sp_digest, sp_diff_id = create_layer_blobs([(site_packages_dir, "app/site-packages")])
    print(f"      site-packages compressed size: {len(sp_bytes) / (1024*1024):.2f} MiB ({sp_digest[:20]}...)")
    ensure_blob_uploaded(
        session=session,
        headers=headers,
        registry_host=registry_host,
        project_id=project_id,
        repo_name=repo_name,
        target_image=target_image,
        blob_bytes=sp_bytes,
        digest=sp_digest,
    )

    print("[3/4] Packing application layer...")
    app_bytes, app_digest, app_diff_id = create_layer_blobs(app_mappings)
    print(f"      app layer compressed size: {len(app_bytes) / 1024:.1f} KiB ({app_digest[:20]}...)")
    ensure_blob_uploaded(
        session=session,
        headers=headers,
        registry_host=registry_host,
        project_id=project_id,
        repo_name=repo_name,
        target_image=target_image,
        blob_bytes=app_bytes,
        digest=app_digest,
    )

    env_list = [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
        "LANG=C.UTF-8",
        "PORT=8080",
        "PYTHONPATH=/app:/app/site-packages",
    ]
    if extra_env:
        env_list.extend(extra_env)

    new_cfg = dict(base_cfg)
    new_cfg["config"] = {
        **base_cfg.get("config", {}),
        "Env": env_list,
        "WorkingDir": "/app",
        "Entrypoint": [
            "/usr/bin/python3.11",
            "-m",
            "uvicorn",
            entrypoint_module,
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ],
    }
    new_cfg["rootfs"] = {
        "type": "layers",
        "diff_ids": [*distroless_diff_ids, sp_diff_id, app_diff_id],
    }

    cfg_bytes = json.dumps(new_cfg, separators=(",", ":")).encode("utf-8")
    cfg_digest = "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()
    ensure_blob_uploaded(
        session=session,
        headers=headers,
        registry_host=registry_host,
        project_id=project_id,
        repo_name=repo_name,
        target_image=target_image,
        blob_bytes=cfg_bytes,
        digest=cfg_digest,
    )

    new_manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "size": len(cfg_bytes),
            "digest": cfg_digest,
        },
        "layers": [
            *[
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "size": l["size"],
                    "digest": l["digest"],
                }
                for l in distroless_layers
            ],
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "size": len(sp_bytes),
                "digest": sp_digest,
            },
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "size": len(app_bytes),
                "digest": app_digest,
            },
        ],
    }

    manifest_bytes = json.dumps(new_manifest, separators=(",", ":")).encode("utf-8")
    put_man_url = (
        f"https://{registry_host}/v2/{project_id}/{repo_name}/"
        f"{target_image}/manifests/{target_tag}"
    )
    put_man_headers = {
        **headers,
        "Content-Type": "application/vnd.oci.image.manifest.v1+json",
    }
    print(f"[4/4] Publishing manifest {registry_host}/{project_id}/{repo_name}/{target_image}:{target_tag}...")
    r_put_man = session.put(put_man_url, headers=put_man_headers, data=manifest_bytes, timeout=30)
    r_put_man.raise_for_status()
    full_uri = f"{registry_host}/{project_id}/{repo_name}/{target_image}:{target_tag}"
    print(f"Successfully pushed {full_uri}")
    return full_uri


def main() -> None:
    load_dotenv_file()
    parser = argparse.ArgumentParser(description="Push Python 3.11 OCI image to Artifact Registry.")
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GCP_PROJECT_ID", ""),
        help="Target GCP Project ID",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_REGION", "europe-west1"),
        help="Artifact Registry region",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("ARTIFACT_REPO_NAME", "gecx-skincare-repo"),
        help="Artifact Registry repository name",
    )
    parser.add_argument("--image", required=True, help="Target image name (e.g., skincare-nano-banana-api)")
    parser.add_argument("--tag", default="latest", help="Target image tag")
    parser.add_argument("--entrypoint", required=True, help="Uvicorn entrypoint (e.g., skincare_api.app:app)")
    parser.add_argument("--site-packages", required=True, help="Path to Python 3.11 site-packages folder")
    parser.add_argument(
        "--map",
        action="append",
        required=True,
        help="Mapping local_dir:container_dir (e.g., skincare_api:app/skincare_api)",
    )
    parser.add_argument(
        "--mount-from",
        default=None,
        help="Optional sibling image in the same Artifact Registry repo to cross-mount base layers from",
    )
    parser.add_argument("--account", default=None, help="Optional gcloud account email")
    args = parser.parse_args()

    if not args.project_id or args.project_id == "YOUR_GCP_PROJECT_ID":
        raise SystemExit("Error: --project-id (or GCP_PROJECT_ID in .env) is required.")

    mappings: list[tuple[Path, str]] = []
    for m in args.map:
        src_s, dst_s = m.split(":", 1)
        mappings.append((Path(src_s).resolve(), dst_s))

    build_and_push(
        project_id=args.project_id,
        region=args.region,
        repo_name=args.repo,
        target_image=args.image,
        target_tag=args.tag,
        entrypoint_module=args.entrypoint,
        site_packages_dir=Path(args.site_packages).resolve(),
        app_mappings=mappings,
        mount_from_image=args.mount_from,
        account=args.account,
    )


if __name__ == "__main__":
    main()
