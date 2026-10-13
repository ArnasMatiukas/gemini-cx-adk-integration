"""Core Google ADK Agent definition (`root_agent`) for the Lumiere Skincare Specialist."""

from __future__ import annotations

import os

from skincare_adk_agent.config import CONFIG

# Ensure Google GenAI SDK inside ADK uses Vertex AI in global location
# (even when Vertex AI Agent Runtime sets GOOGLE_CLOUD_LOCATION to the engine's regional endpoint)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
if CONFIG.project_id:
    os.environ["GOOGLE_CLOUD_PROJECT"] = CONFIG.project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = CONFIG.location

from google.adk.agents import Agent
from google.genai import types

from skincare_adk_agent.prompts import SKINCARE_SPECIALIST_INSTRUCTION
from skincare_adk_agent.tools.cloud_run_skincare_tool import (
    analyze_skincare_image_on_cloud_run,
)


def create_skincare_adk_agent() -> Agent:
    """Builds the Google ADK `root_agent` configured with `gemini-3.8-flash` and the Cloud Run tool."""
    return Agent(
        name="skincare_specialist_adk_agent",
        model=CONFIG.model_name,
        description=(
            "Clinical Skincare Specialist ADK Agent that delegates visual skin and product "
            "image diagnostics to the Cloud Run Nano Banana API (gemini-3.1-flash-image) "
            "and synthesizes personalized skincare regimens."
        ),
        instruction=SKINCARE_SPECIALIST_INSTRUCTION,
        tools=[analyze_skincare_image_on_cloud_run],
        output_key="skincare_specialist_response",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=CONFIG.thinking_budget),
        ),
    )


root_agent = create_skincare_adk_agent()
