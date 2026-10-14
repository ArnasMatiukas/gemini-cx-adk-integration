#!/usr/bin/env python3
"""GCP authentication helper and daemonless OCI image builder for Artifact Registry."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tarfile
from typing import Any

import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
import requests

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DISTROLESS_REPO = "distroless/python3-debian12"
LOCAL_BASE_CACHE = Path("/tmp/distroless_py311_cache")


def load_dotenv(env_path: Path | None = None) -> None:
    env_path = env_path or (Path(__file__).resolve().parent.parent / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k and v and k not in os.environ:
                os.environ[k] = v


def _gcloud_sqlite_token(account: str | None = None) -> str | None:
    gcloud_dir = Path.home() / ".config" / "gcloud"
    db_path = gcloud_dir / "credentials.db"
    if not db_path.exists():
        return None
    account = account or os.environ.get("GCP_AUTH_ACCOUNT")
    if not account and (gcloud_dir / "active_config").exists():
        cfg = (gcloud_dir / "active_config").read_text().strip()
        cfg_file = gcloud_dir / "configurations" / f"config_{cfg}"
        if cfg_file.exists():
            for line in cfg_file.read_text().splitlines():
                if line.strip().startswith("account"):
                    account = line.split("=", 1)[1].strip()
                    break
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        if account:
            cur.execute("SELECT value FROM credentials WHERE account_id = ?", (account,))
        else:
            cur.execute("SELECT value FROM credentials LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        data = json.loads(row[0])
        creds = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=data["refresh_token"],
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            scopes=[CLOUD_PLATFORM_SCOPE],
        )
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except Exception:
        return None


def get_auth_headers(account: str | None = None, content_type: str | None = "application/json") -> dict[str, str]:
    token = _gcloud_sqlite_token(account) if (account or os.environ.get("GCP_AUTH_ACCOUNT")) else None
    if not token:
        try:
            creds, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
            creds.refresh(google.auth.transport.requests.Request())
            token = creds.token
        except Exception:
            token = _gcloud_sqlite_token(account)
    if not token:
        cmd = ["gcloud", "auth", "print-access-token"] + (["--account", account] if account else [])
        token = subprocess.check_output(cmd, text=True, timeout=15).strip()
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def get_project_number(project_id: str, account: str | None = None) -> str:
    r = requests.get(
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}",
        headers=get_auth_headers(account),
        timeout=20,
    )
    r.raise_for_status()
    return str(r.json()["projectNumber"])


def _pack_layer(mappings: list[tuple[Path, str]]) -> tuple[bytes, str, str]:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        for src, arc_prefix in mappings:
            if src.is_file():
                ti = tf.gettarinfo(str(src), arcname=arc_prefix)
                ti.uid = ti.gid = ti.mtime = 0
                with open(src, "rb") as f:
                    tf.addfile(ti, f)
            else:
                for root, dirs, files in os.walk(src):
                    dirs[:] = [d for d in sorted(dirs) if d not in ("__pycache__", ".git")]
                    for file in sorted(files):
                        if file.endswith((".pyc", ".pyo")):
                            continue
                        fp = Path(root) / file
                        if fp.is_file() and not fp.is_symlink():
                            ti = tf.gettarinfo(str(fp), arcname=f"{arc_prefix.rstrip('/')}/{fp.relative_to(src)}")
                            ti.uid = ti.gid = ti.mtime = 0
                            with open(fp, "rb") as f:
                                tf.addfile(ti, f)
    raw = tar_buf.getvalue()
    diff_id = "sha256:" + hashlib.sha256(raw).hexdigest()
    gz_buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gz_buf, mtime=0) as gz:
        gz.write(raw)
    comp = gz_buf.getvalue()
    return comp, "sha256:" + hashlib.sha256(comp).hexdigest(), diff_id


def _fetch_distroless_base(session: requests.Session) -> tuple[dict[str, Any], dict[str, Any]]:
    LOCAL_BASE_CACHE.mkdir(parents=True, exist_ok=True)
    m_file, c_file = LOCAL_BASE_CACHE / "manifest.json", LOCAL_BASE_CACHE / "config.json"
    if m_file.exists() and c_file.exists():
        return json.loads(m_file.read_text()), json.loads(c_file.read_text())
    accept = {"Accept": "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json"}
    idx = session.get(f"https://gcr.io/v2/{DISTROLESS_REPO}/manifests/latest", headers=accept, timeout=30).json()
    if "manifests" in idx:
        amd64 = next(m for m in idx["manifests"] if m.get("platform", {}).get("architecture") == "amd64")
        manifest = session.get(f"https://gcr.io/v2/{DISTROLESS_REPO}/manifests/{amd64['digest']}", headers=accept, timeout=30).json()
    else:
        manifest = idx
    config = session.get(f"https://gcr.io/v2/{DISTROLESS_REPO}/blobs/{manifest['config']['digest']}", timeout=30).json()
    m_file.write_text(json.dumps(manifest))
    c_file.write_text(json.dumps(config))
    return manifest, config


def _upload_blob(
    session: requests.Session,
    headers: dict[str, str],
    host: str,
    project_id: str,
    repo: str,
    image: str,
    digest: str,
    data: bytes | None = None,
    mount_from: str | None = None,
) -> None:
    base = f"https://{host}/v2/{project_id}/{repo}/{image}"
    if session.head(f"{base}/blobs/{digest}", headers=headers, timeout=20).status_code == 200:
        return
    if mount_from and mount_from != image:
        r_m = session.post(f"{base}/blobs/uploads/?mount={digest}&from={project_id}/{repo}/{mount_from}", headers=headers, timeout=20)
        if r_m.status_code == 201:
            return
    if data is None:
        cache_p = LOCAL_BASE_CACHE / digest.replace(":", "_")
        if not cache_p.exists():
            cache_p.write_bytes(session.get(f"https://gcr.io/v2/{DISTROLESS_REPO}/blobs/{digest}", timeout=60).content)
        data = cache_p.read_bytes()
    r_init = session.post(f"{base}/blobs/uploads/", headers=headers, timeout=30)
    r_init.raise_for_status()
    loc = r_init.headers["Location"]
    url = loc if loc.startswith("http") else f"https://{host}{loc}"
    session.put(f"{url}{'&' if '?' in url else '?'}digest={digest}", headers={**headers, "Content-Type": "application/octet-stream"}, data=data, timeout=300).raise_for_status()


def build_and_push_oci(
    project_id: str,
    region: str,
    repo_name: str,
    image_name: str,
    entrypoint: str,
    site_packages_dir: Path,
    app_mappings: list[tuple[Path, str]],
    mount_from: str | None = None,
    account: str | None = None,
) -> str:
    """Packs and pushes a Python 3.11 Distroless OCI image to Artifact Registry."""
    host = f"{region}-docker.pkg.dev"
    headers = get_auth_headers(account=account, content_type=None)
    session = requests.Session()
    base_man, base_cfg = _fetch_distroless_base(session)

    for layer in base_man["layers"]:
        _upload_blob(session, headers, host, project_id, repo_name, image_name, layer["digest"], mount_from=mount_from)

    sp_bytes, sp_dig, sp_diff = _pack_layer([(site_packages_dir, "app/site-packages")])
    _upload_blob(session, headers, host, project_id, repo_name, image_name, sp_dig, data=sp_bytes)

    app_bytes, app_dig, app_diff = _pack_layer(app_mappings)
    _upload_blob(session, headers, host, project_id, repo_name, image_name, app_dig, data=app_bytes)

    cfg = {
        **base_cfg,
        "config": {
            **base_cfg.get("config", {}),
            "Env": [
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
                "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/certs/ca-certificates.crt",
                "PORT=8080",
                "PYTHONPATH=/app:/app/site-packages",
            ],
            "WorkingDir": "/app",
            "Entrypoint": ["/usr/bin/python3.11", "-m", "uvicorn", entrypoint, "--host", "0.0.0.0", "--port", "8080"],
        },
        "rootfs": {"type": "layers", "diff_ids": [*base_cfg["rootfs"]["diff_ids"], sp_diff, app_diff]},
    }
    cfg_bytes = json.dumps(cfg, separators=(",", ":")).encode()
    cfg_dig = "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()
    _upload_blob(session, headers, host, project_id, repo_name, image_name, cfg_dig, data=cfg_bytes)

    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "size": len(cfg_bytes), "digest": cfg_dig},
        "layers": [
            *[{"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "size": l["size"], "digest": l["digest"]} for l in base_man["layers"]],
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "size": len(sp_bytes), "digest": sp_dig},
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "size": len(app_bytes), "digest": app_dig},
        ],
    }
    session.put(
        f"https://{host}/v2/{project_id}/{repo_name}/{image_name}/manifests/latest",
        headers={**headers, "Content-Type": "application/vnd.oci.image.manifest.v1+json"},
        data=json.dumps(manifest, separators=(",", ":")).encode(),
        timeout=30,
    ).raise_for_status()
    uri = f"{host}/{project_id}/{repo_name}/{image_name}:latest"
    print(f"  -> Pushed {uri}")
    return uri
