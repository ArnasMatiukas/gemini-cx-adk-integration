"""Unit tests for A2A input extraction (both RemoteA2aAgent inline parts and RemoteAgentTool metadata)."""

from __future__ import annotations

import json
import unittest

from skincare_adk_agent.a2a_server import (
    build_a2a_v10_message_object,
    extract_a2a_input,
)


class TestA2aExtraction(unittest.TestCase):
    """Tests extraction of text and skincare images across A2A v1.0 and v0.3 formats."""

    def test_extract_inline_raw_part_v10(self) -> None:
        payload = {
            "message": {
                "messageId": "m-1",
                "contextId": "ctx-inline-1",
                "role": "ROLE_USER",
                "parts": [
                    {"text": "Check my skin photo"},
                    {"raw": "iVBORw0KGgoAAAANSUhEUg==", "mediaType": "image/png"},
                ],
            }
        }
        text, b64, mime, ctx_id = extract_a2a_input(payload)
        self.assertEqual(text, "Check my skin photo")
        self.assertEqual(b64, "iVBORw0KGgoAAAANSUhEUg==")
        self.assertEqual(mime, "image/png")
        self.assertEqual(ctx_id, "ctx-inline-1")

    def test_extract_gecx_context_history_image(self) -> None:
        payload = {
            "message": {
                "messageId": "m-2",
                "contextId": "ctx-meta-2",
                "role": "ROLE_USER",
                "parts": [{"text": "Delegate skin analysis"}],
                "metadata": {
                    "gecx_a2a_agent_context": json.dumps(
                        {
                            "history": [
                                {
                                    "role": "USER",
                                    "chunks": [
                                        {"text": "Uploaded photo"},
                                        {
                                            "image": {
                                                "mimeType": "image/jpeg",
                                                "data": "/9j/4AAQSkZJRg==",
                                            }
                                        },
                                    ],
                                }
                            ]
                        }
                    )
                },
            }
        }
        text, b64, mime, ctx_id = extract_a2a_input(payload)
        self.assertEqual(text, "Delegate skin analysis")
        self.assertEqual(b64, "/9j/4AAQSkZJRg==")
        self.assertEqual(mime, "image/jpeg")
        self.assertEqual(ctx_id, "ctx-meta-2")

    def test_build_a2a_v10_message_object(self) -> None:
        resp = build_a2a_v10_message_object(
            response_text="Clinical summary",
            context_id="ctx-123",
            last_analysis={
                "analysis_id": "nb-skin-123",
                "model_used": "gemini-3.1-flash-image",
                "visual_assessment": {"apparent_skin_type": "sensitive"},
            },
        )
        self.assertEqual(resp["message"]["contextId"], "ctx-123")
        self.assertEqual(resp["message"]["role"], "ROLE_AGENT")
        self.assertEqual(resp["message"]["parts"][0]["text"], "Clinical summary")


if __name__ == "__main__":
    unittest.main()
