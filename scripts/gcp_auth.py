#!/usr/bin/env python3
"""Zero-config Google Cloud authentication and project metadata helper.

Supports obtaining OAuth2 Bearer tokens across all standard customer environments:
1. `GOOGLE_APPLICATION_CREDENTIALS` service account / workload identity file.
2. Standard Application Default Credentials (`google.auth.default()`).
3. Local `gcloud` SQLite credential store (`~/.config/gcloud/credentials.db`).
4. `gcloud auth print-access-token` CLI fallback.
"""

from __future__ import annotations

from pathlib import Path
import os
import sqlite3
import subprocess
from typing import Any

import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
import requests

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def load_dotenv_file(env_path: Path | None = None) -> None:
    """Loads key=value pairs from `.env` into `os.environ` if not already set."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and val and key not in os.environ:
            os.environ[key] = val


def _load_creds_from_gcloud_sqlite(account: str | None = None) -> google.oauth2.credentials.Credentials | None:
    """Reads OAuth2 user credentials directly from `~/.config/gcloud/credentials.db`."""
    gcloud_dir = Path.home() / ".config" / "gcloud"
    db_path = gcloud_dir / "credentials.db"
    if not db_path.exists():
        return None

    if not account:
        account = os.environ.get("GCP_AUTH_ACCOUNT")
    if not account:
        active_cfg_path = gcloud_dir / "active_config"
        if active_cfg_path.exists():
            cfg_name = active_cfg_path.read_text(encoding="utf-8").strip()
            cfg_file = gcloud_dir / "configurations" / f"config_{cfg_name}"
            if cfg_file.exists():
                for line in cfg_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("account"):
                        _, acc_val = line.split("=", 1)
                        account = acc_val.strip()
                        break

    try:
        import json

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
        if not data.get("refresh_token"):
            return None
        creds = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=data["refresh_token"],
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            scopes=[CLOUD_PLATFORM_SCOPE],
        )
        creds.refresh(google.auth.transport.requests.Request())
        return creds
    except Exception:
        return None


def get_access_token(account: str | None = None) -> str:
    """Returns a valid Google Cloud OAuth2 access token."""
    # 1. Explicit account or gcloud SQLite credential DB
    if account or os.environ.get("GCP_AUTH_ACCOUNT"):
        sqlite_creds = _load_creds_from_gcloud_sqlite(account)
        if sqlite_creds and sqlite_creds.token:
            return sqlite_creds.token

    # 2. Standard ADC / GOOGLE_APPLICATION_CREDENTIALS
    try:
        creds, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        creds.refresh(google.auth.transport.requests.Request())
        if creds.token:
            return creds.token
    except Exception:
        pass

    # 3. Fallback to gcloud SQLite credential store
    sqlite_creds = _load_creds_from_gcloud_sqlite(account)
    if sqlite_creds and sqlite_creds.token:
        return sqlite_creds.token

    # 4. Fallback to gcloud CLI
    cmd = ["gcloud", "auth", "print-access-token"]
    if account:
        cmd.extend(["--account", account])
    out = subprocess.check_output(cmd, text=True, timeout=15).strip()
    if not out:
        raise RuntimeError(
            "Unable to obtain Google Cloud access token. Run `gcloud auth application-default login` "
            "or `gcloud auth login` first."
        )
    return out


def get_auth_headers(account: str | None = None, content_type: str | None = "application/json") -> dict[str, str]:
    """Returns HTTP headers with `Authorization: Bearer <token>`."""
    token = get_access_token(account=account)
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def get_project_number(project_id: str, account: str | None = None) -> str:
    """Resolves the numeric Google Cloud project number for `project_id`."""
    headers = get_auth_headers(account=account)
    r = requests.get(
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}",
        headers=headers,
        timeout=20,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Failed to resolve project number for '{project_id}' (HTTP {r.status_code}): {r.text}"
        )
    data: dict[str, Any] = r.json()
    return str(data["projectNumber"])
