"""Configuration for the Google ADK Skincare Specialist Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SkincareAgentConfig:
    """Typed runtime configuration for the ADK Skincare Specialist Agent.

    All environment variables are injected automatically by `deploy_all.py` into
    the Vertex AI Agent Runtime (`reasoningEngines`) `deploymentSpec.env`, or
    loaded from `.env` during local execution.
    """

    project_id: str = (
        os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    )
    project_number: str = os.environ.get("GCP_PROJECT_NUMBER", "")
    location: str = os.environ.get("ADK_MODEL_LOCATION", "global")
    agent_engine_location: str = (
        os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
        or os.environ.get("AGENT_ENGINE_LOCATION")
        or os.environ.get("GCP_REGION", "europe-west1")
    )
    agent_engine_id: str = (
        os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID")
        or os.environ.get("AGENT_ENGINE_ID", "")
    )
    model_name: str = os.environ.get("ADK_MODEL_NAME", "gemini-3.8-flash")
    thinking_budget: int = int(os.environ.get("ADK_THINKING_BUDGET", "8192"))
    skincare_api_url: str = os.environ.get("SKINCARE_API_URL", "http://localhost:8080")


CONFIG = SkincareAgentConfig()
