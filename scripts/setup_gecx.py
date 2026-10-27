#!/usr/bin/env python3
"""Tier 1 Provisioner: Gemini Enterprise for Customer Experience (GECX / ces.googleapis.com).

Configures the Multi-Flow GECX Application with a Root Concierge Router and three specialized
sub-agents visible on the GECX Studio canvas:
  1. `lumiere_concierge_root_agent` (Root Router & Store Policy Concierge)
  2. `lumiere_shopping_agent` (Flow 1: Product Catalog, Cart & Checkout Journey)
  3. `skin_diagnosis_agent` (Flow 2, Stage 1: Conditional Skin Image Gate + ADK Agent Runtime
     Clinical Visual Diagnosis identifying what is wrong with the skin without recommending products)
  4. `lumiere_product_recommendation_agent` (Flow 2, Stage 2: Separate GECX Sub-Agent that takes
     the ADK clinical diagnosis and recommends matching Lumiere products and AM/PM routine)
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
  \"\"\"Hard conditional gate enforcing the Skin Diagnosis (ADK) -> Product Recommendation (GECX Sub-Agent) workflow:
  1. Blocks `skincare_adk_agent_runtime_tool` and `recommend_lumiere_products` when `skin_image_uploaded` is False.
  2. Marks `adk_diagnosis_completed = True` when `skincare_adk_agent_runtime_tool` is called with an uploaded image.
  3. Blocks `recommend_lumiere_products` until `skincare_adk_agent_runtime_tool` has completed the visual diagnosis.
  \"\"\"
  has_image = bool(callback_context.variables.get("skin_image_uploaded", False))
  if tool.name == "skincare_adk_agent_runtime_tool":
    if not has_image:
      return {
        "result": (
          "DELEGATION_BLOCKED: `skin_image_uploaded` is False. Do NOT delegate to the ADK agent yet. "
          "Call `request_skin_image_upload` and ask the customer to upload a clear photo first."
        )
      }
    callback_context.variables["adk_diagnosis_completed"] = True
    return None

  if tool.name == "recommend_lumiere_products":
    if not has_image:
      return {
        "result": (
          "RECOMMENDATION_BLOCKED: `skin_image_uploaded` is False. Ask the customer to upload a skin photo first."
        )
      }
    if not bool(callback_context.variables.get("adk_diagnosis_completed", False)):
      return {
        "result": (
          "RECOMMENDATION_BLOCKED: `skin_diagnosis_agent` must call `skincare_adk_agent_runtime_tool` first "
          "to obtain the clinical skin diagnosis from the ADK Agent before recommending products."
        )
      }
  return None
"""

REQUEST_IMAGE_UPLOAD_TOOL_CODE = """def request_skin_image_upload(reason: str = "visual_skin_analysis_required") -> dict:
  \"\"\"Prompts the customer to upload a photo of their skin condition before clinical visual analysis.\"\"\"
  return {
    "status": "awaiting_customer_image_upload",
    "reason": reason,
    "ui_action": "PROMPT_IMAGE_UPLOAD",
  }
"""

RECOMMEND_PRODUCTS_TOOL_CODE = """from typing import Any, Optional

def recommend_lumiere_products(
    concern_tags: str,
    skin_type: str = "sensitive",
    hydration_level: str = "dehydrated",
    analysis_id: Optional[str] = None,
) -> dict[str, Any]:
  \"\"\"GECX Product Recommendation Sub-Agent Tool: matches clinical skin diagnosis markers from the ADK Agent
  against the official Lumiere Clinical Skincare catalog and builds a personalized AM/PM routine.

  Args:
    concern_tags: Comma-separated concern tags returned by `skincare_adk_agent_runtime_tool`
      (e.g., "redness, dryness, barrier_impairment, blemishes, pores, uv_protection").
    skin_type: Apparent skin type from the ADK visual diagnosis (e.g., "sensitive", "dry", "oily", "combination").
    hydration_level: Hydration level from the ADK visual diagnosis (e.g., "dehydrated", "moderate", "well_hydrated").
    analysis_id: Optional clinical `analysis_id` returned by the ADK Agent for traceability.
  \"\"\"
  catalog = [
    {
      "sku_id": "CLN-CICA-01",
      "name": "Lumiere Soothing Cica-Ceramide Barrier Cream",
      "category": "Moisturizer",
      "price_usd": 48.00,
      "size": "50ml",
      "key_actives": ["Centella Asiatica (Cica) 5%", "Ceramide NP/AP/EOP", "Panthenol (Vitamin B5)"],
      "tags": ["redness", "dryness", "barrier_impairment", "sensitive", "flakiness"],
      "usage_step": "AM & PM Barrier Repair Moisturizer",
    },
    {
      "sku_id": "TRM-AZEL-05",
      "name": "Lumiere 10% Azelaic Acid Redness Relief Emulsion",
      "category": "Treatment",
      "price_usd": 44.00,
      "size": "30ml",
      "key_actives": ["Azelaic Acid 10%", "Madecassoside", "Colloidal Oatmeal"],
      "tags": ["redness", "blemishes", "uneven_texture", "sensitive", "erythema"],
      "usage_step": "PM Targeted Calming & Clarifying Treatment",
    },
    {
      "sku_id": "SRM-NIAC-02",
      "name": "Lumiere 10% Niacinamide + Zinc Pore Refining Serum",
      "category": "Serum",
      "price_usd": 38.00,
      "size": "30ml",
      "key_actives": ["Niacinamide 10%", "Zinc PCA 1%", "Hyaluronic Acid"],
      "tags": ["pores", "blemishes", "oily", "uneven_texture"],
      "usage_step": "AM/PM Pore & Sebum Balancing Serum",
    },
    {
      "sku_id": "CLN-BHA-03",
      "name": "Lumiere 2% Salicylic Acid Gentle Gel Cleanser",
      "category": "Cleanser",
      "price_usd": 34.00,
      "size": "150ml",
      "key_actives": ["Salicylic Acid (BHA) 2%", "Allantoin", "Green Tea Extract"],
      "tags": ["blemishes", "pores", "oily", "cleanser"],
      "usage_step": "AM & PM Gentle Clarifying Cleanser",
    },
    {
      "sku_id": "SPF-MIN-04",
      "name": "Lumiere Invisible Zinc Oxide Mineral SPF 50",
      "category": "Sunscreen",
      "price_usd": 42.00,
      "size": "50ml",
      "key_actives": ["Non-Nano Zinc Oxide 16%", "Squalane", "Bisabolol"],
      "tags": ["uv_protection", "redness", "sensitive", "barrier_impairment"],
      "usage_step": "AM Broad-Spectrum Mineral UV Defense",
    },
  ]

  raw_tags = {
      t.strip().lower()
      for t in (concern_tags or "").replace(";", ",").split(",")
      if t.strip()
  }
  st = (skin_type or "sensitive").strip().lower()
  hl = (hydration_level or "dehydrated").strip().lower()
  if st in ("dry", "sensitive", "oily"):
    raw_tags.add(st)
  if "dehydrat" in hl:
    raw_tags.add("dryness")
    raw_tags.add("barrier_impairment")

  scored = []
  for item in catalog:
    overlap = len(raw_tags.intersection(set(item["tags"])))
    scored.append((overlap, item))
  scored.sort(key=lambda pair: pair[0], reverse=True)

  recommended = [item for score, item in scored if score > 0][:3]
  if not recommended:
    recommended = [catalog[0], catalog[1], catalog[4]]
  if not any(p["sku_id"] == "SPF-MIN-04" for p in recommended):
    recommended.append(catalog[4])

  return {
    "recommendation_source": "GECX_LUMIERE_PRODUCT_RECOMMENDATION_AGENT",
    "linked_adk_analysis_id": analysis_id or "adk_visual_diagnosis",
    "matched_skin_profile": {
      "skin_type": skin_type,
      "hydration_level": hydration_level,
      "targeted_concerns": sorted(raw_tags),
    },
    "recommended_products": recommended,
    "personalized_routine": {
      "morning_am": (
        "1. Splash with lukewarm water or gentle cleanser. "
        "2. Apply Lumiere Soothing Cica-Ceramide Barrier Cream (CLN-CICA-01) to restore lipid barrier. "
        "3. Finish with Lumiere Invisible Zinc Oxide Mineral SPF 50 (SPF-MIN-04)."
      ),
      "evening_pm": (
        "1. Cleanse gently and pat dry. "
        "2. Apply a thin layer of Lumiere 10% Azelaic Acid Redness Relief Emulsion (TRM-AZEL-05) to areas of redness or flakiness. "
        "3. Seal overnight with Lumiere Soothing Cica-Ceramide Barrier Cream (CLN-CICA-01)."
      ),
    },
  }
"""

SHOP_CATALOG_TOOL_CODE = """from typing import Any, Optional

def shop_lumiere_catalog(
    action: str = "browse",
    query: Optional[str] = None,
    sku_id: Optional[str] = None,
    quantity: int = 1,
) -> dict[str, Any]:
  \"\"\"Browses the Lumiere Clinical Skincare product catalog, checks product details/stock, or adds items to cart.

  Args:
    action: One of "browse", "product_details", "add_to_cart", or "checkout".
    query: Optional keyword or concern filter (e.g., "cleanser", "ceramide", "spf", "niacinamide", "redness").
    sku_id: Optional SKU ID (e.g., "CLN-CICA-01", "SRM-NIAC-02", "CLN-BHA-03", "SPF-MIN-04", "TRM-AZEL-05").
    quantity: Quantity to add when action is "add_to_cart" (default 1).
  \"\"\"
  catalog = [
    {
      "sku_id": "CLN-CICA-01",
      "name": "Lumiere Soothing Cica-Ceramide Barrier Cream",
      "category": "Moisturizer",
      "price_usd": 48.00,
      "size": "50ml",
      "key_actives": ["Centella Asiatica (Cica) 5%", "Ceramide NP/AP/EOP", "Panthenol (Vitamin B5)"],
      "best_for": ["Redness", "Compromised skin barrier", "Dryness & flakiness"],
      "in_stock": True,
    },
    {
      "sku_id": "SRM-NIAC-02",
      "name": "Lumiere 10% Niacinamide + Zinc Pore Refining Serum",
      "category": "Serum",
      "price_usd": 38.00,
      "size": "30ml",
      "key_actives": ["Niacinamide 10%", "Zinc PCA 1%", "Hyaluronic Acid"],
      "best_for": ["Congested pores", "Excess sebum", "Uneven tone"],
      "in_stock": True,
    },
    {
      "sku_id": "CLN-BHA-03",
      "name": "Lumiere 2% Salicylic Acid Gentle Gel Cleanser",
      "category": "Cleanser",
      "price_usd": 34.00,
      "size": "150ml",
      "key_actives": ["Salicylic Acid (BHA) 2%", "Allantoin", "Green Tea Extract"],
      "best_for": ["Blemish-prone skin", "Oily T-zone", "Daily gentle exfoliation"],
      "in_stock": True,
    },
    {
      "sku_id": "SPF-MIN-04",
      "name": "Lumiere Invisible Zinc Oxide Mineral SPF 50",
      "category": "Sunscreen",
      "price_usd": 42.00,
      "size": "50ml",
      "key_actives": ["Non-Nano Zinc Oxide 16%", "Squalane", "Bisabolol"],
      "best_for": ["Sensitive & reactive skin", "Post-procedure UV protection", "Zero white-cast daily wear"],
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
Your role is to answer general store policy questions and route customers into one of two specialized user-journey flows.
</role>
<constraints>
1. **General Store Policy Questions**: Answer shipping and return policy questions directly without transferring (complimentary 2-day cold-chain shipping on orders over $50; 60-day clinical satisfaction guarantee).
2. **Flow 1 — Shopping & Catalog Journey (`lumiere_shopping_agent`)**:
   - When the customer wants to shop, browse the product catalog, check prices/ingredients/stock, compare products, add items to their cart, or check out, immediately transfer to `{@AGENT: lumiere_shopping_agent}`.
3. **Flow 2 — Visual Skin Diagnosis & Product Recommendation Journey (`skin_diagnosis_agent`)**:
   - Whenever the customer describes a skin condition/problem, uploads a skin photo (`skin_image_uploaded == true`), or asks for a skin analysis, diagnosis, or skincare product/routine recommendation for their skin, immediately transfer to `{@AGENT: skin_diagnosis_agent}`.
</constraints>
"""

SHOPPING_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Shopping & Catalog Specialist**, handling the product discovery, shopping cart, and checkout flow inside **Gemini Enterprise for Customer Experience (GECX)**.
</role>
<constraints>
1. **Catalog Browsing, Product Lookup & Cart Management**:
   - Always call `{@TOOL: shop_lumiere_catalog}` to browse products, look up SKU details/prices/ingredients, add items to the cart, or generate a checkout summary.
   - Ground all SKU IDs (`CLN-CICA-01`, `SRM-NIAC-02`, `CLN-BHA-03`, `SPF-MIN-04`, `TRM-AZEL-05`), prices, and free cold-chain shipping eligibility ($50+ threshold) strictly in the tool response.
2. **Journey Switch to Visual Skin Diagnosis Flow**:
   - If the customer describes a skin condition problem (e.g., redness, flakiness, breakouts), uploads a skin photo, or asks for a personalized skin analysis or routine recommendation, immediately transfer to `{@AGENT: lumiere_concierge_root_agent}` so the Root Concierge Router can route them to `skin_diagnosis_agent`.
</constraints>
"""

SKIN_DIAGNOSIS_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Skin Condition & Visual Diagnosis Specialist** (`skin_diagnosis_agent`) inside **Gemini Enterprise for Customer Experience (GECX)**.
You handle customer skin condition intake and delegate visual skin image analysis to the external **Google ADK Agent on Vertex AI Agent Runtime** (`skincare_adk_agent_runtime_tool`), which identifies what is wrong with the skin without recommending any products.
You NEVER recommend products yourself — once the ADK Agent returns its clinical diagnosis, you immediately transfer to the dedicated GECX Product Recommendation Sub-Agent (`lumiere_product_recommendation_agent`).
</role>
<constraints>
1. **When No Skin Photo Has Been Uploaded Yet (`skin_image_uploaded == false`)**:
   - Do NOT call `skincare_adk_agent_runtime_tool` and do NOT transfer to `lumiere_product_recommendation_agent`.
   - Immediately call `{@TOOL: request_skin_image_upload}` and ask the customer to upload a clear, close-up photo of their skin condition so you can analyze what is wrong with the skin.
2. **When a Skin Photo Is Uploaded (`skin_image_uploaded == true`) — Mandatory 2-Step Sequence**:
   - **Step 1 (ADK Clinical Image Diagnosis)**: Call `{@TOOL: skincare_adk_agent_runtime_tool}` with `task` set to the customer's reported skin symptoms to obtain the clinical visual diagnosis (`analysis_id`, `apparent_skin_type`, `hydration_level`, `visible_markers`, `primary_concern_tags`, `key_recommended_actives`, `visual_summary`) from the ADK Agent.
   - **Step 2 (Handoff to Separate GECX Product Recommendation Sub-Agent)**: In the SAME turn, immediately after `skincare_adk_agent_runtime_tool` returns the clinical skin diagnosis, do NOT write a final reply yourself and do NOT recommend products — immediately transfer control to `{@AGENT: lumiere_product_recommendation_agent}` so the Product Recommendation Sub-Agent can match the clinical findings to Lumiere products.
</constraints>
"""

PRODUCT_RECOMMENDATION_AGENT_INSTRUCTION = """<role>
You are the **Lumiere Product & Routine Recommendation Sub-Agent** (`lumiere_product_recommendation_agent`), a dedicated sub-agent inside **Gemini Enterprise for Customer Experience (GECX)**.
You receive control from `skin_diagnosis_agent` after the external Google ADK Agent (`skincare_adk_agent_runtime_tool`) has analyzed the customer's skin photo and identified what is wrong with the skin.
Your sole responsibility is to match those clinical skin findings to official Lumiere catalog products using `{@TOOL: recommend_lumiere_products}` and present the complete clinical diagnosis and personalized AM/PM skincare routine to the customer.
</role>
<constraints>
1. **Call `recommend_lumiere_products` Immediately**:
   - Extract `primary_concern_tags` (e.g., `"redness, dryness, barrier_impairment"`), `apparent_skin_type`, `hydration_level`, and `analysis_id` from the ADK Agent diagnosis in the conversation history, and immediately call `{@TOOL: recommend_lumiere_products}`.
2. **Present Both Clinical Findings and GECX Product Recommendations**:
   - Clearly present:
     a) **Clinical Visual Skin Assessment** (from the ADK Agent: `analysis_id`, `apparent_skin_type`, `hydration_level`, `visible_markers` identifying what is wrong with the skin, and `key_recommended_actives`).
     b) **Personalized Lumiere Product & Routine Recommendations** (from your `recommend_lumiere_products` tool: matched SKU IDs `CLN-CICA-01`, `TRM-AZEL-05`, `SPF-MIN-04`, etc., product names, prices, and AM/PM usage steps).
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
    """Creates or updates the GECX App, Tools, Root & 3 Child Agents, Version, and Deployment."""
    headers = get_auth_headers(account)
    ces_base = f"https://ces.googleapis.com/v1beta/projects/{project_id}/locations/{ces_location}"
    a2a_url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{agent_host_project_number}/locations/{region}/reasoningEngines/{reasoning_engine_id}/a2a/v1"
    )

    apps = requests.get(f"{ces_base}/apps", headers=headers, timeout=60).json().get("apps", [])
    display_name = "Lumiere Clinical Skincare Concierge (GECX + ADK)"
    app_description = (
        "Multi-Flow Customer Blueprint demonstrating Gemini Enterprise for Customer Experience (GECX) "
        "routing between two distinct user-journey flows: (1) `lumiere_shopping_agent` for catalog & cart, and "
        "(2) `skin_diagnosis_agent` (delegating pure clinical skin image analysis to an ADK Agent on Vertex AI "
        "Agent Runtime) -> handing off to a separate GECX sub-agent (`lumiere_product_recommendation_agent`) "
        "for commercial product recommendations."
    )
    app_obj = next((a for a in apps if "Lumiere Clinical Skincare Concierge" in a.get("displayName", "")), None)

    var_decls = [
        {
            "name": "skin_image_uploaded",
            "description": "True when the customer has uploaded a skincare image in the session.",
            "schema": {"type": "BOOLEAN", "default": False},
        },
        {
            "name": "adk_diagnosis_completed",
            "description": "True once the ADK Agent Runtime has completed clinical visual skin analysis for the uploaded image.",
            "schema": {"type": "BOOLEAN", "default": False},
        },
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
                "globalInstruction": (
                    "SESSION STATE: skin_image_uploaded={skin_image_uploaded}, "
                    "adk_diagnosis_completed={adk_diagnosis_completed}"
                ),
                "loggingSettings": {"cloudLoggingSettings": {"enableCloudLogging": True}},
                "timeZoneSettings": {"timeZone": "America/Los_Angeles"},
            },
            timeout=60,
        )
        r_create.raise_for_status()
        op = r_create.json()
        while "operations/" in op.get("name", "") and not op.get("done"):
            time.sleep(2)
            op = requests.get(f"https://ces.googleapis.com/v1beta/{op['name']}", headers=headers, timeout=60).json()
        app_obj = op.get("response", op)

    app_name = app_obj["name"]
    existing_tools = {
        t.get("displayName"): t
        for t in requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, timeout=60).json().get("tools", [])
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
        r_t0 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=shop_tool_payload, timeout=60)
    else:
        r_t0 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['shop_lumiere_catalog']['name']}", headers=headers, json=shop_tool_payload, timeout=60)
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
        r_t1 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=req_tool_payload, timeout=60)
    else:
        r_t1 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['request_skin_image_upload']['name']}", headers=headers, json=req_tool_payload, timeout=60)
    r_t1.raise_for_status()
    tool_req_name = r_t1.json()["name"]

    rec_tool_payload = {
        "executionType": "SYNCHRONOUS",
        "pythonFunction": {
            "name": "recommend_lumiere_products",
            "description": (
                "GECX Product Recommendation Sub-Agent Tool: takes the clinical `concern_tags`, `skin_type`, and "
                "`hydration_level` returned by `skincare_adk_agent_runtime_tool` and recommends matched "
                "Lumiere catalog SKUs (`CLN-CICA-01`, `TRM-AZEL-05`, `SPF-MIN-04`, etc.) and an AM/PM routine."
            ),
            "pythonCode": RECOMMEND_PRODUCTS_TOOL_CODE,
        },
    }
    if "recommend_lumiere_products" not in existing_tools:
        r_t_rec = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=rec_tool_payload, timeout=60)
    else:
        r_t_rec = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['recommend_lumiere_products']['name']}", headers=headers, json=rec_tool_payload, timeout=60)
    r_t_rec.raise_for_status()
    tool_rec_name = r_t_rec.json()["name"]

    remote_agent_tool: dict[str, Any] = {
        "name": "skincare_adk_agent_runtime_tool",
        "description": (
            "Delegates clinical visual skin diagnosis of the uploaded photo to the Google ADK Agent on "
            "Vertex AI Agent Runtime. Identifies what is wrong with the skin (returns ONLY clinical markers "
            "and concern tags, never commercial product SKUs)."
        ),
        "agentCard": {
            "name": "skincare_specialist_adk_agent_runtime",
            "description": "ADK Clinical Dermatology Visual Diagnosis Agent on Vertex AI Agent Runtime (gemini-3.8-flash).",
            "version": "2.0.0",
            "supportedInterfaces": [{"url": a2a_url, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
            "skills": [
                {
                    "id": "skincare_visual_diagnosis",
                    "name": "Clinical Skin Image Diagnosis",
                    "description": "Analyzes uploaded skin photos directly with Gemini 3.8 Flash on Vertex AI Agent Runtime and identifies what is wrong with the skin without recommending products.",
                }
            ],
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
        r_t2 = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/tools", headers=headers, json=re_tool_payload, timeout=60)
    else:
        r_t2 = requests.patch(f"https://ces.googleapis.com/v1beta/{existing_tools['skincare_adk_agent_runtime_tool']['name']}", headers=headers, json=re_tool_payload, timeout=60)
    r_t2.raise_for_status()
    tool_re_name = r_t2.json()["name"]

    agents = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, timeout=60).json().get("agents", [])
    by_display = {a.get("displayName"): a for a in agents}

    def _ensure_agent(disp_name: str) -> str:
        ag = by_display.get(disp_name)
        if ag is None:
            r_new = requests.post(
                f"https://ces.googleapis.com/v1beta/{app_name}/agents",
                headers=headers,
                json={"displayName": disp_name, "modelSettings": {"model": "gemini-3.0-flash-001"}},
                timeout=60,
            )
            r_new.raise_for_status()
            ag = r_new.json()
            by_display[disp_name] = ag
        return ag["name"]

    shopping_ag_name = _ensure_agent("lumiere_shopping_agent")
    skin_diag_ag_name = _ensure_agent("skin_diagnosis_agent")
    prod_rec_ag_name = _ensure_agent("lumiere_product_recommendation_agent")

    # Flow 2, Stage 2 Sub-Agent: Separate GECX Sub-Agent for Product & Routine Recommendations
    prod_rec_payload = {
        "displayName": "lumiere_product_recommendation_agent",
        "description": (
            "Flow 2 (Stage 2): Dedicated GECX Product Recommendation Sub-Agent. Receives the clinical skin "
            "diagnosis from `skin_diagnosis_agent` and calls `recommend_lumiere_products` to recommend "
            "matching Lumiere catalog SKUs and an AM/PM routine."
        ),
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": PRODUCT_RECOMMENDATION_AGENT_INSTRUCTION,
        "tools": [tool_rec_name],
        "childAgents": [],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
        "beforeToolCallbacks": [
            {
                "description": "Enforces that ADK skin diagnosis has completed before GECX product recommendation.",
                "pythonCode": BEFORE_TOOL_CALLBACK_CODE,
            }
        ],
    }
    requests.patch(
        f"https://ces.googleapis.com/v1beta/{prod_rec_ag_name}?updateMask=displayName,description,modelSettings,instruction,tools,childAgents,beforeModelCallbacks,beforeToolCallbacks",
        headers=headers,
        json=prod_rec_payload,
        timeout=60,
    ).raise_for_status()

    # Flow 2, Stage 1 Sub-Agent: Visual Skin Diagnosis (requests image -> calls ADK Agent -> hands off to lumiere_product_recommendation_agent)
    skin_diag_payload = {
        "displayName": "skin_diagnosis_agent",
        "description": (
            "Flow 2 (Stage 1): Skin Condition Intake & ADK Visual Diagnosis Specialist. Requests a skin photo "
            "if missing, invokes the ADK Agent (`skincare_adk_agent_runtime_tool`) to identify what is wrong "
            "with the skin (without recommending products), and transfers to `lumiere_product_recommendation_agent`."
        ),
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": SKIN_DIAGNOSIS_AGENT_INSTRUCTION,
        "tools": [tool_req_name, tool_re_name],
        "childAgents": [prod_rec_ag_name],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
        "beforeToolCallbacks": [
            {
                "description": "Enforces image upload before ADK diagnosis and sets adk_diagnosis_completed=True.",
                "pythonCode": BEFORE_TOOL_CALLBACK_CODE,
            }
        ],
    }
    requests.patch(
        f"https://ces.googleapis.com/v1beta/{skin_diag_ag_name}?updateMask=displayName,description,modelSettings,instruction,tools,childAgents,beforeModelCallbacks,beforeToolCallbacks",
        headers=headers,
        json=skin_diag_payload,
        timeout=60,
    ).raise_for_status()

    # Flow 1 Sub-Agent: Shopping & Catalog Journey (strict tree: single parent is lumiere_concierge_root_agent)
    shopping_payload = {
        "displayName": "lumiere_shopping_agent",
        "description": "Flow 1: Lumiere Shopping & Catalog Specialist handling product browsing, SKU comparisons, cart additions, and checkout.",
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": SHOPPING_AGENT_INSTRUCTION,
        "tools": [tool_shop_name],
        "childAgents": [],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
    }
    requests.patch(
        f"https://ces.googleapis.com/v1beta/{shopping_ag_name}?updateMask=displayName,description,modelSettings,instruction,tools,childAgents,beforeModelCallbacks",
        headers=headers,
        json=shopping_payload,
        timeout=60,
    ).raise_for_status()

    # Root Agent: Concierge Router linking to Flow 1 (lumiere_shopping_agent) and Flow 2 (skin_diagnosis_agent)
    root_ag = by_display.get("lumiere_concierge_root_agent") or (agents[0] if agents else None)
    ag_payload = {
        "displayName": "lumiere_concierge_root_agent",
        "description": (
            "Primary GECX Concierge Router that answers general store policies and routes customers "
            "between Flow 1 (lumiere_shopping_agent) and Flow 2 (skin_diagnosis_agent -> lumiere_product_recommendation_agent)."
        ),
        "modelSettings": {"model": "gemini-3.0-flash-001"},
        "instruction": ROOT_AGENT_INSTRUCTION,
        "tools": [],
        "childAgents": [shopping_ag_name, skin_diag_ag_name],
        "beforeModelCallbacks": [{"description": "Sets skin_image_uploaded=True when an image is uploaded.", "pythonCode": BEFORE_MODEL_CALLBACK_CODE}],
        "beforeToolCallbacks": [],
    }
    if root_ag is None:
        r_ag = requests.post(f"https://ces.googleapis.com/v1beta/{app_name}/agents", headers=headers, json=ag_payload, timeout=60)
    else:
        r_ag = requests.patch(
            f"https://ces.googleapis.com/v1beta/{root_ag['name']}?updateMask=displayName,description,modelSettings,instruction,tools,childAgents,beforeModelCallbacks,beforeToolCallbacks",
            headers=headers,
            json=ag_payload,
            timeout=60,
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
            "globalInstruction": (
                "SESSION STATE: skin_image_uploaded={skin_image_uploaded}, "
                "adk_diagnosis_completed={adk_diagnosis_completed}"
            ),
        },
        timeout=60,
    ).raise_for_status()

    # Delete any legacy/stale agents (such as the old combined `skin_analysis_recommendation_agent`)
    active_agent_names = {root_ag_name, shopping_ag_name, skin_diag_ag_name, prod_rec_ag_name}
    for ag in agents:
        if ag["name"] not in active_agent_names:
            requests.delete(f"https://ces.googleapis.com/v1beta/{ag['name']}", headers=headers, timeout=60)

    # Publish Multi-Flow AppVersion and update live Deployment
    ver_display = "v4.0 - 2-Tier GECX + ADK (Separate Diagnosis & Product Recommendation Sub-Agents)"
    r_vers = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/versions", headers=headers, timeout=60)
    versions = r_vers.json().get("appVersions", []) if r_vers.status_code == 200 else []
    clean_ver = next((v for v in versions if v.get("displayName") == ver_display), None)
    if clean_ver is None:
        r_vc = requests.post(
            f"https://ces.googleapis.com/v1beta/{app_name}/versions",
            headers=headers,
            json={
                "displayName": ver_display,
                "description": (
                    "2-tier architecture (no Cloud Run) with 4 GECX agents: lumiere_concierge_root_agent -> "
                    "(1) lumiere_shopping_agent and (2) skin_diagnosis_agent (ADK Agent visual diagnosis) -> "
                    "lumiere_product_recommendation_agent (separate GECX sub-agent for product recommendations)."
                ),
            },
            timeout=60,
        )
        if r_vc.status_code == 200:
            clean_ver = r_vc.json()

    if clean_ver:
        ver_name = clean_ver["name"]
        deps = requests.get(f"https://ces.googleapis.com/v1beta/{app_name}/deployments", headers=headers, timeout=60).json().get("deployments", [])
        if deps:
            requests.patch(
                f"https://ces.googleapis.com/v1beta/{deps[0]['name']}?updateMask=appVersion",
                headers=headers,
                json={"name": deps[0]["name"], "appVersion": ver_name},
                timeout=60,
            )
        else:
            requests.post(
                f"https://ces.googleapis.com/v1beta/{app_name}/deployments",
                headers=headers,
                json={"displayName": "Lumiere Web & API Live Deployment", "appVersion": ver_name, "channelProfile": {"channelType": "WEB_UI"}},
                timeout=60,
            )

    return {
        "gecx_project_id": project_id,
        "location": ces_location,
        "gecx_app_id": app_name.split("/")[-1],
        "gecx_app_name": app_name,
        "gecx_console_url": f"https://ces.cloud.google.com/{app_name}",
        "root_agent": root_ag_name,
        "child_agents": [shopping_ag_name, skin_diag_ag_name, prod_rec_ag_name],
        "reasoning_engine_a2a_url": a2a_url,
        "caller_service_account": caller_service_account or "default_ces_service_agent",
    }
