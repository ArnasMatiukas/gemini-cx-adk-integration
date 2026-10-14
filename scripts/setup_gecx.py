#!/usr/bin/env python3
"""Tier 1 Provisioner: Gemini Enterprise for Customer Experience (GECX / ces.googleapis.com).

Configures the Multi-Flow GECX Application with a Root Concierge Router and two distinct
user-journey sub-agents visible on the GECX Studio canvas:
  1. `lumiere_concierge_root_agent` (Root Router & Store Policy Concierge)
  2. `lumiere_shopping_agent` (Flow 1: Product Catalog, Cart & Checkout Journey)
  3. `skin_analysis_recommendation_agent` (Flow 2: Conditional Skin Image Gate -> ADK Agent Runtime -> Cloud Run Gemini 3.8 Flash API)
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from oci_builder import get_auth_headers

BEFORE_MODEL_CALLBACK_CODE = """from typing import Any, List, Optional

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
  \"\"\"Sets skin_image_uploaded=True as soon as an uploaded image (inline_data) is present in the session.\"\"\"
  for content in (llm_request.contents or []):
    for part in (getattr(content, "parts", None) or []):
      inline_data = getattr(part, "inline_data", None)
      if inline_data and getattr(inline_data, "data", None):
        callback_context.variables["skin_image_uploaded"] = True
        return None
  return None
"""

BEFORE_TOOL_CALLBACK_CODE = """from typing import Any, Optional

def before_tool_callback(tool: Tool, input: dict[str, Any], callback_context: CallbackContext) -> Optional[dict[str, Any]]:
  \"\"\"Hard conditional gate: blocks ADK delegation until the customer uploads a skincare image.\"\"\"
  if tool.name == "skincare_adk_agent_runtime_tool" and not bool(callback_context.variables.get("skin_image_uploaded", False)):
    return {
      "result": (
        "DELEGATION_BLOCKED: `skin_image_uploaded` is False. Do NOT delegate to the ADK agent yet. "
        "Call `request_skin_image_upload` and ask the customer to upload a clear photo first."
      )
    }
  return None
"""

REQUEST_IMAGE_UPLOAD_TOOL_CODE = """def request_skin_image_upload(reason: str = "visual_skin_analysis_required") -> dict:
  \"\"\"Prompts the customer to upload a photo of their skin concern or skincare product.\"\"\"
  return {
    "status": "awaiting_customer_image_upload",
    "reason": reason,
    "ui_action": "PROMPT_IMAGE_UPLOAD",
  }
"""

SHOP_CATALOG_TOOL_CODE = """from typing import Any, Optional

def shop_lumiere_catalog(
    action: str = "browse",
    query: Optional[str] = None,
    sku_id: Optional[str] = None,
    quantity: int = 1,
) -> dict[str, Any]:
  \"\"\"Searches the Lumiere Clinical Skincare product catalog, checks prices/stock, or adds items to the shopping cart.

  Args:
    action: One of "browse", "product_details", "add_to_cart", or "checkout".
    query: Optional search keyword or skin concern filter (e.g., "redness", "cleanser", "spf", "barrier").
    sku_id: Optional product SKU ID (e.g., "CLN-CICA-01", "SRM-NIAC-02", "CLN-BHA-03", "SPF-MIN-04", "TRM-AZEL-05").
    quantity: Number of units to add when action is "add_to_cart" (default 1).
  \"\"\"
  catalog = [
    {
      "sku_id": "CLN-CICA-01",
      "name": "Lumiere Soothing Cica-Ceramide Barrier Cream",
      "category": "Moisturizer",
      "price_usd": 48.00,
      "size": "50ml",
      "key_actives": ["Centella Asiatica (Cica) 5%", "Ceramide NP/AP/EOP", "Panthenol (Vitamin B5)"],
      "best_for": ["Barrier repair", "Redness", "Flaky or sensitized skin"],
      "in_stock": True,
    },
    {
      "sku_id": "SRM-NIAC-02",
      "name": "Lumiere 10% Niacinamide + Zinc Pore Refining Serum",
      "category": "Serum",
      "price_usd": 38.00,
      "size": "30ml",
      "key_actives": ["Niacinamide 10%", "Zinc PCA 1%", "Hyaluronic Acid"],
      "best_for": ["Enlarged pores", "Excess sebum", "Uneven texture"],
      "in_stock": True,
    },
    {
      "sku_id": "CLN-BHA-03",
      "name": "Lumiere 2% Salicylic Acid Gentle Gel Cleanser",
      "category": "Cleanser",
      "price_usd": 34.00,
      "size": "150ml",
      "key_actives": ["Salicylic Acid (BHA) 2%", "Allantoin", "Green Tea Extract"],
      "best_for": ["Congested pores", "Blemish-prone skin", "Gentle daily cleansing"],
      "in_stock": True,
    },
    {
      "sku_id": "SPF-MIN-04",
      "name": "Lumiere Invisible Zinc Oxide Mineral SPF 50",
      "category": "Sunscreen",
      "price_usd": 42.00,
      "size": "50ml",
      "key_actives": ["Non-Nano Zinc Oxide 16%", "Squalane", "Bisabolol"],
      "best_for": ["Daily UV protection", "Post-procedure skin", "Rosacea-prone skin"],
      "in_stock": True,
    },
    {
      "sku_id": "TRM-AZEL-05",
      "name": "Lumiere 10% Azelaic Acid Redness Relief Emulsion",
      "category": "Treatment",
      "price_usd": 44.00,
      "size": "30ml",
      "key_actives": ["Azelaic Acid 10%", "Madecassoside", "Colloidal Oatmeal"],
      "best_for": ["Erythema / redness", "Post-inflammatory marks", "Reactive skin"],
      "in_stock": True,
    },
  ]
  by_sku = {item["sku_id"]: item for item in catalog}
  normalized_action = (action or "browse").strip().lower()
  target = by_sku.get((sku_id or "").strip().upper())

  if not target and query:
    q = query.strip().lower()
    for item in catalog:
      if q in item["sku_id"].lower() or q in item["name"].lower() or any(q in b.lower() for b in item["best_for"]):
        target = item
        break

  if normalized_action in ("add_to_cart", "checkout") and target:
    qty = max(1, int(quantity or 1))
    subtotal = round(target["price_usd"] * qty, 2)
    qualifies_free_shipping = subtotal >= 50.0
    return {
      "status": "cart_updated" if normalized_action == "add_to_cart" else "checkout_ready",
      "action": normalized_action,
      "item_added": target["name"],
      "sku_id": target["sku_id"],
      "unit_price_usd": target["price_usd"],
      "quantity": qty,
      "subtotal_usd": subtotal,
      "free_cold_chain_shipping_unlocked": qualifies_free_shipping,
      "checkout_url": f"https://shop.lumiereclinical.example/checkout?sku={target['sku_id']}&qty={qty}",
    }

  if normalized_action == "product_details" and target:
    return {"status": "ok", "action": "product_details", "product": target}

  if query:
    q = query.strip().lower()
    matches = [
      item for item in catalog
      if q in item["name"].lower()
      or q in item["category"].lower()
      or any(q in act.lower() for act in item["key_actives"])
      or any(q in bf.lower() for bf in item["best_for"])
    ]
    return {"status": "ok", "action": "browse", "query": query, "products": matches or catalog}

  return {"status": "ok", "action": "browse", "products": catalog}
"""

ROOT_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Clinical Skincare Concierge Router**, the primary entry point built on **Gemini Enterprise for Customer Experience (GECX)**.
Your role is to answer general store policy questions and route customers into one of two specialized sub-agent flows based on their user journey.
</role>
<constraints>
1. **General Store Policy Questions**: Answer shipping and return policy questions directly without transferring (complimentary 2-day cold-chain shipping on orders over $50; 60-day clinical satisfaction guarantee).
2. **Flow 1 — Shopping & Catalog Journey (`lumiere_shopping_agent`)**:
   - When the customer wants to shop, browse the product catalog, check prices/ingredients/stock, compare products, add items to their cart, or check out, immediately transfer to `{@AGENT: lumiere_shopping_agent}`.
3. **Flow 2 — Visual Skin Analysis & Routine Recommendation Journey (`skin_analysis_recommendation_agent`)**:
   - When the customer asks for a personalized skin analysis, skin diagnosis, or skincare routine recommendation based on their skin concern or uploaded skin photo, immediately transfer to `{@AGENT: skin_analysis_recommendation_agent}`.
</constraints>
"""

SHOPPING_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Shopping & Catalog Specialist**, handling the product discovery, shopping cart, and checkout flow inside **Gemini Enterprise for Customer Experience (GECX)**.
</role>
<constraints>
1. **Catalog Browsing, Product Lookup & Cart Management**:
   - Always call `{@TOOL: shop_lumiere_catalog}` to browse products, look up SKU details/prices/ingredients, add items to the cart, or generate a checkout summary.
   - Ground all SKU IDs (`CLN-CICA-01`, `SRM-NIAC-02`, `CLN-BHA-03`, `SPF-MIN-04`, `TRM-AZEL-05`), prices, and free cold-chain shipping eligibility ($50+ threshold) strictly in the tool response.
2. **Journey Switch to Visual Skin Analysis Flow**:
   - If the customer asks for a personalized skin analysis, diagnosis, or routine recommendation based on their skin concern or uploaded skin photo, immediately transfer to `{@AGENT: lumiere_concierge_root_agent}` so they are routed to the Skin Analysis & Recommendation flow.
</constraints>
"""

SKIN_ANALYSIS_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Visual Skin Analysis & Recommendation Specialist** inside **Gemini Enterprise for Customer Experience (GECX)**.
You own the clinical skin diagnosis flow and enforce the conditional image upload gate before delegating to the Google ADK Agent on Vertex AI Agent Runtime.
</role>
<constraints>
1. **Conditional Image Delegation Gate (`skin_image_uploaded`)**:
   - **If `skin_image_uploaded == false`** and the customer asks for a personal skin analysis, diagnosis, or skincare routine: do NOT call the ADK agent. Immediately call `{@TOOL: request_skin_image_upload}` and ask the customer to upload a clear close-up photo of their skin concern first.
   - **If `skin_image_uploaded == true`**: immediately call `{@TOOL: skincare_adk_agent_runtime_tool}` with `task` set to the customer's question, and relay the returned Gemini 3.8 Flash visual analysis (`analysis_id`, visible markers, active ingredients, AM/PM routine, and recommended Lumiere SKUs).
2. **Journey Switch to Shopping & Cart Flow**:
   - If the customer asks to add recommended products to their cart, browse catalog pricing, or check out, immediately transfer to `{@AGENT: lumiere_concierge_root_agent}` so they are routed to the Shopping flow.
</constraints>
"""


def setup_gecx_app(
    project_id: str,
    agent_host_project_number: str,
    region: str,
    reasoning_engine_id: str,
    ces_location: str = "us",
    caller_service_account: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Creates or updates the GECX App, Tools, Root Agent, Version, and Deployment."""
    headers = get_auth_headers(account)
    ces_base = f"https://ces.googleapis.com/v1beta/projects/{project_id}/locations/{ces_location}"
    a2a_url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{agent_host_project_number}/locations/{region}/reasoningEngines/{reasoning_engine_id}/a2a/v1"
    )

    apps = requests.get(f"{ces_base}/apps", headers=headers, timeout=20).json().get("apps", [])
    display_name = "Lumiere Clinical Skincare Concierge (GECX + ADK + Cloud Run)"
    app_description = (
        "Multi-Flow Customer Blueprint demonstrating Gemini Enterprise for Customer Experience (GECX) "
        "routing between two distinct user-journey flows: (1) Shopping & Catalog Concierge, and "
        "(2) Visual Skin Analysis & Routine Recommendation (conditionally delegating uploaded skin "
        "photos to a Google ADK Agent on Vertex AI Agent Runtime + Cloud Run Gemini 3.8 Flash API)."
    )
    app_obj = next((a for a in apps if "Lumiere Clinical Skincare Concierge" in a.get("displayName", "")), None)

    var_decls = [
        {
            "name": "skin_image_uploaded",
            "description": "True when the customer has uploaded a skincare image in the session.",
            "schema": {"type": "BOOLEAN", "default": False},
        }
    ]

    if app_obj is None:
        r_create = requests.post(
            f"{ces_base}/apps",
            headers=headers,
            json={
                "displayName": display_name,
                "description": app_description,
                "pinned": True,
                "modelSettings": {"model": "gemini-3.0-flash-001"},
                "variableDeclarations": var_decls,
                "globalInstruction": "SESSION STATE: skin_image_uploaded={skin_image_uploaded}",
                "loggingSettings": {"cloudLoggingSettings": {"enableCloudLogging": True}},
                "timeZoneSettings": {"timeZone": "America/Los_Angeles"},
            },
            timeout=30,
        )
        r_create.raise_for_status()
        op = r_create.json()
        while "operations/" in op.get("name", "") and not op.get("done"):
            time.sleep(2)
            op = requests.get(f"https://ces.googleapis.com/v1beta/{op['name']}", headers=headers, timeout=15).json()
        app_obj = op.get("response", op)

    app_name = app_obj["name"]
    existing_tools = {
        t.get("displayName"): t
        for t in requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, timeout=15).json().get("tools", [])
    }

    shop_tool_payload = {
        "executionType": "SYNCHRONOUS",
        "pythonFunction": {
            "name": "shop_lumiere_catalog",
            "description": "Searches the Lumiere Clinical Skincare product catalog, checks prices/stock, or adds items to the shopping cart.",
            "pythonCode": SHOP_CATALOG_TOOL_CODE,
        },
    }
    if "shop_lumiere_catalog" not in existing_tools:
        r_t0 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=shop_tool_payload, timeout=20)
    else:
        r_t0 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['shop_lumiere_catalog']['name']}", headers=headers, json=shop_tool_payload, timeout=20)
    r_t0.raise_for_status()
    tool_shop_name = r_t0.json()["name"]

    req_tool_payload = {
        "executionType": "SYNCHRONOUS",
        "pythonFunction": {
            "name": "request_skin_image_upload",
            "description": "Prompts the customer to upload a skincare image before visual analysis.",
            "pythonCode": REQUEST_IMAGE_UPLOAD_TOOL_CODE,
        },
    }
    if "request_skin_image_upload" not in existing_tools:
        r_t1 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=req_tool_payload, timeout=20)
    else:
        r_t1 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['request_skin_image_upload']['name']}", headers=headers, json=req_tool_payload, timeout=20)
    r_t1.raise_for_status()
    tool_req_name = r_t1.json()["name"]

    remote_agent_tool: dict[str, Any] = {
        "name": "skincare_adk_agent_runtime_tool",
        "description": "Delegates skincare image analysis to the Google ADK Agent on Vertex AI Agent Runtime.",
        "agentCard": {
            "name": "skincare_specialist_adk_agent_runtime",
            "description": "ADK Skincare Specialist Agent on Vertex AI Agent Runtime.",
            "version": "1.0.0",
            "supportedInterfaces": [{"url": a2a_url, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
            "skills": [{"id": "skincare_visual_diagnosis", "name": "Skincare Image Analysis", "description": "Analyzes uploaded skin photos via Cloud Run Gemini 3.8 Flash API."}],
        },
    }
    if caller_service_account:
        remote_agent_tool["apiAuthentication"] = {
            "serviceAccountAuthConfig": {
                "serviceAccount": caller_service_account,
            }
        }
    re_tool_payload = {
        "executionType": "SYNCHRONOUS",
        "timeout": "60s",
        "remoteAgentTool": remote_agent_tool,
    }
    if "skincare_adk_agent_runtime_tool" not in existing_tools:
        r_t2 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=re_tool_payload, timeout=20)
    else:
        r_t2 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['skincare_adk_agent_runtime_tool']['name']}", headers=headers, json=re_tool_payload, timeout=20)
    r_t2.raise_for_status()
    tool_re_name = r_t2.json()["name"]

    agents = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, timeout=15).json().get("agents", [])
    by_display = {a.get("displayName"): a for a in agents}

    # Flow 1 Sub-Agent: Shopping & Catalog Journey
    shopping_ag = by_display.get("lumiere_shopping_agent")
    shopping_payload = {
        "displayName": "lumiere_shopping_agent",
        "description": "Flow 1: Lumiere Shopping & Catalog Specialist handling product browsing, SKU comparisons, cart additions, and checkout.",
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": SHOPPING_AGENT_INSTRUCTION,
        "tools": [tool_shop_name],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
    }
    if shopping_ag is None:
        r_shop = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, json=shopping_payload, timeout=20)
    else:
        r_shop = requests.patch(f"https://ces.googleapis.com/v1beta/{shopping_ag['name']}", headers=headers, json=shopping_payload, timeout=20)
    r_shop.raise_for_status()
    shopping_ag_name = r_shop.json()["name"]

    # Flow 2 Sub-Agent: Visual Skin Analysis & ADK Recommendation Journey
    skin_ag = by_display.get("skin_analysis_recommendation_agent")
    skin_payload = {
        "displayName": "skin_analysis_recommendation_agent",
        "description": (
            "Flow 2: Visual Skin Analysis & Routine Recommendation Specialist that enforces conditional "
            "image delegation (skin_image_uploaded) before invoking the Google ADK Agent on Vertex AI Agent Runtime."
        ),
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": SKIN_ANALYSIS_AGENT_INSTRUCTION,
        "tools": [tool_req_name, tool_re_name],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
        "beforeToolCallbacks": [{"description": "Blocks ADK delegation if skin_image_uploaded is False.", "pythonCode": BEFORE_TOOL_CALLBACK_CODE}],
    }
    if skin_ag is None:
        r_skin = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, json=skin_payload, timeout=20)
    else:
        r_skin = requests.patch(f"https://ces.googleapis.com/v1beta/{skin_ag['name']}", headers=headers, json=skin_payload, timeout=20)
    r_skin.raise_for_status()
    skin_ag_name = r_skin.json()["name"]

    # Root Agent: Concierge Router linking to both child agent flows
    root_ag = by_display.get("lumiere_concierge_root_agent") or (agents[0] if agents else None)
    ag_payload = {
        "displayName": "lumiere_concierge_root_agent",
        "description": (
            "Primary GECX Concierge Router that answers general store policies and routes customers "
            "between Flow 1 (lumiere_shopping_agent) and Flow 2 (skin_analysis_recommendation_agent)."
        ),
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": ROOT_AGENT_INSTRUCTION,
        "tools": [],
        "childAgents": [shopping_ag_name, skin_ag_name],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
        "beforeToolCallbacks": [],
    }
    if root_ag is None:
        r_ag = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, json=ag_payload, timeout=20)
    else:
        r_ag = requests.patch(
            f"https://ces.googleapis.com/v1beta/{root_ag['name']}?updateMask=displayName,description,modelSettings,instruction,tools,childAgents,beforeModelCallbacks,beforeToolCallbacks",
            headers=headers,
            json=ag_payload,
            timeout=20,
        )
    r_ag.raise_for_status()
    root_ag_name = r_ag.json()["name"]

    requests.patch(
        f"https://ces.googleapis.com/v1beta/{app_name}?updateMask=rootAgent,variableDeclarations,globalInstruction,modelSettings,pinned,displayName,description",
        headers=headers,
        json={
            "name": app_name,
            "displayName": display_name,
            "description": app_description,
            "pinned": True,
            "rootAgent": root_ag_name,
            "variableDeclarations": var_decls,
            "modelSettings": {"model": "gemini-3.0-flash-001"},
            "globalInstruction": "SESSION STATE: skin_image_uploaded={skin_image_uploaded}",
        },
        timeout=20,
    ).raise_for_status()

    # Publish Multi-Flow AppVersion and update live Deployment
    ver_display = "v2.0 - Multi-Flow GECX (Shopping + Skin Image ADK Analysis)"
    r_vers = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/versions", headers=headers, timeout=15)
    versions = r_vers.json().get("appVersions", []) if r_vers.status_code == 200 else []
    clean_ver = next((v for v in versions if v.get("displayName") == ver_display), None)
    if clean_ver is None:
        r_vc = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/versions",
            headers=headers,
            json={
                "displayName": ver_display,
                "description": (
                    "Multi-flow release with Root Concierge Router branching into "
                    "(1) lumiere_shopping_agent and (2) skin_analysis_recommendation_agent "
                    "(conditional image gate -> ADK Agent Runtime -> Cloud Run Gemini 3.8 Flash Vision API)."
                ),
            },
            timeout=20,
        )
        if r_vc.status_code == 200:
            clean_ver = r_vc.json()

    if clean_ver:
        ver_name = clean_ver["name"]
        deps = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/deployments", headers=headers, timeout=15).json().get("deployments", [])
        if deps:
            requests.patch(
                f"https://ces.googleapis.com/v1beta/{deps[0]['name']}?updateMask=appVersion",
                headers=headers,
                json={"name": deps[0]["name"], "appVersion": ver_name},
                timeout=20,
            )
        else:
            requests.post(
                f"https://ces.googleapis.com/v1beta/{app_name}/deployments",
                headers=headers,
                json={"displayName": "Lumiere Web & API Live Deployment", "appVersion": ver_name, "channelProfile": {"channelType": "WEB_UI"}},
                timeout=20,
            )
        for v in versions:
            if v["name"] != ver_name and ("nano" in json.dumps(v).lower() or "banana" in json.dumps(v).lower()):
                requests.delete(f"https://ces.googleapis.com/v1beta/{v['name']}", headers=headers, timeout=15)

    return {
        "gecx_project_id": project_id,
        "location": ces_location,
        "gecx_app_id": app_name.split("/")[-1],
        "gecx_app_name": app_name,
        "gecx_console_url": f"https://ces.cloud.google.com/{app_name}",
        "root_agent": root_ag_name,
        "child_agents": [shopping_ag_name, skin_ag_name],
        "reasoning_engine_a2a_url": a2a_url,
        "caller_service_account": caller_service_account or "default_ces_service_agent",
    }
