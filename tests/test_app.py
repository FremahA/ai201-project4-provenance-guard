"""
Provenance Guard — Test Suite
Run with: python -m pytest tests/ -v
Or:        python -m unittest discover tests
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a temp log file during tests so we don't pollute the real audit log
TEST_LOG = "test_audit_log.json"

import app as app_module
app_module.AUDIT_LOG_PATH = TEST_LOG

from app import (
    app,
    compute_stylometric_score,
    compute_punctuation_score,
    compute_confidence,
    get_attribution,
    generate_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_test_log():
    if os.path.exists(TEST_LOG):
        os.remove(TEST_LOG)


AI_TEXT = (
    "Artificial intelligence represents a transformative paradigm shift in modern society. "
    "It is important to note that while the benefits of AI are numerous, it is equally "
    "essential to consider the ethical implications. Furthermore, stakeholders across "
    "various sectors must collaborate to ensure responsible deployment of these systems. "
    "In conclusion, a balanced approach will be critical to navigating the complexities "
    "of this transformative era and ensuring equitable outcomes for all members of society."
)

HUMAN_TEXT = (
    "ok so i finally tried that new ramen place downtown and honestly? underwhelming. "
    "the broth was fine but they put WAY too much sodium in it and i was thirsty for "
    "like three hours after. my friend got the spicy version and said it was better. "
    "probably won't go back unless someone drags me there lol"
)


# ---------------------------------------------------------------------------
# Unit tests: signal functions
# ---------------------------------------------------------------------------
class TestStylometricScore(unittest.TestCase):

    def test_returns_float_in_range(self):
        score = compute_stylometric_score(HUMAN_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ai_text_scores_higher_than_human(self):
        ai_score    = compute_stylometric_score(AI_TEXT)
        human_score = compute_stylometric_score(HUMAN_TEXT)
        # AI text should generally score higher (more uniform structure)
        self.assertGreater(ai_score, human_score)

    def test_very_short_text_returns_midrange(self):
        score = compute_stylometric_score("Hello world.")
        # Short texts fall back to 0.5 for one or both sub-metrics
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_string(self):
        score = compute_stylometric_score("")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestPunctuationScore(unittest.TestCase):

    def test_returns_float_in_range(self):
        score = compute_punctuation_score(HUMAN_TEXT)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_string_returns_midrange(self):
        score = compute_punctuation_score("")
        self.assertEqual(score, 0.5)

    def test_heavy_punctuation_scores_lower(self):
        # Lots of punctuation → more human-like → lower AI score
        heavy = "Wait — really?! No way! I can't believe it... seriously?!"
        sparse = "The system demonstrates significant capacity for advanced reasoning."
        self.assertLess(
            compute_punctuation_score(heavy),
            compute_punctuation_score(sparse)
        )


class TestConfidenceScoring(unittest.TestCase):

    def test_high_llm_produces_high_confidence(self):
        score = compute_confidence(0.9, 0.8, 0.8)
        self.assertGreater(score, 0.75)

    def test_low_llm_produces_low_confidence(self):
        score = compute_confidence(0.1, 0.2, 0.1)
        self.assertLess(score, 0.40)

    def test_weighting_llm_dominates(self):
        # LLM=1.0, others=0.0 → 0.60
        self.assertAlmostEqual(compute_confidence(1.0, 0.0, 0.0), 0.60, places=2)
        # LLM=0.0, stylo=1.0, punct=0.0 → 0.25
        self.assertAlmostEqual(compute_confidence(0.0, 1.0, 0.0), 0.25, places=2)
        # LLM=0.0, stylo=0.0, punct=1.0 → 0.15
        self.assertAlmostEqual(compute_confidence(0.0, 0.0, 1.0), 0.15, places=2)

    def test_result_clamped_to_valid_range(self):
        score = compute_confidence(0.5, 0.5, 0.5)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestAttribution(unittest.TestCase):

    def test_high_confidence_is_likely_ai(self):
        self.assertEqual(get_attribution(0.80), "likely_ai")
        self.assertEqual(get_attribution(0.75), "likely_ai")
        self.assertEqual(get_attribution(1.00), "likely_ai")

    def test_mid_confidence_is_uncertain(self):
        self.assertEqual(get_attribution(0.60), "uncertain")
        self.assertEqual(get_attribution(0.40), "uncertain")
        self.assertEqual(get_attribution(0.74), "uncertain")

    def test_low_confidence_is_likely_human(self):
        self.assertEqual(get_attribution(0.39), "likely_human")
        self.assertEqual(get_attribution(0.20), "likely_human")
        self.assertEqual(get_attribution(0.00), "likely_human")


class TestLabelGeneration(unittest.TestCase):

    def test_high_confidence_ai_label(self):
        label = generate_label(0.90)
        self.assertIn("likely generated by an AI tool", label)
        self.assertIn("appeal", label.lower())

    def test_uncertain_label(self):
        label = generate_label(0.55)
        self.assertIn("wasn't able to make a confident determination", label)
        self.assertIn("appeal", label.lower())

    def test_human_label(self):
        label = generate_label(0.20)
        self.assertIn("written by a human", label)

    def test_all_labels_are_plain_language(self):
        for score in [0.10, 0.55, 0.90]:
            label = generate_label(score)
            for jargon in ["classifier", "logit", "threshold", "score", "heuristic"]:
                self.assertNotIn(jargon, label.lower(),
                    msg=f"Label for score {score} contains jargon: '{jargon}'")

    def test_labels_differ_across_thresholds(self):
        ai_label    = generate_label(0.90)
        unc_label   = generate_label(0.55)
        human_label = generate_label(0.20)
        self.assertNotEqual(ai_label, unc_label)
        self.assertNotEqual(unc_label, human_label)
        self.assertNotEqual(ai_label, human_label)


# ---------------------------------------------------------------------------
# Integration tests: API endpoints
# ---------------------------------------------------------------------------
class TestSubmitEndpoint(unittest.TestCase):

    def setUp(self):
        clean_test_log()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        clean_test_log()

    @patch("app.classify_with_llm", return_value=0.85)
    def test_submit_returns_all_required_fields(self, mock_llm):
        res = self.client.post("/submit",
            data=json.dumps({"text": AI_TEXT, "creator_id": "test-user"}),
            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        for field in ["content_id", "attribution", "confidence",
                      "llm_score", "stylo_score", "punct_score", "label", "status"]:
            self.assertIn(field, data, msg=f"Missing field: {field}")

    @patch("app.classify_with_llm", return_value=0.85)
    def test_submit_writes_to_audit_log(self, mock_llm):
        self.client.post("/submit",
            data=json.dumps({"text": AI_TEXT, "creator_id": "test-user"}),
            content_type="application/json")
        entries = app_module.load_log()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["type"], "submission")
        self.assertIn("timestamp", entries[0])
        self.assertIn("confidence", entries[0])

    def test_submit_missing_text_returns_400(self):
        res = self.client.post("/submit",
            data=json.dumps({"creator_id": "test-user"}),
            content_type="application/json")
        self.assertEqual(res.status_code, 400)

    def test_submit_missing_creator_id_returns_400(self):
        res = self.client.post("/submit",
            data=json.dumps({"text": "Some text here."}),
            content_type="application/json")
        self.assertEqual(res.status_code, 400)

    @patch("app.classify_with_llm", return_value=0.1)
    def test_human_text_returns_likely_human_label(self, mock_llm):
        res = self.client.post("/submit",
            data=json.dumps({"text": HUMAN_TEXT, "creator_id": "test-user"}),
            content_type="application/json")
        data = json.loads(res.data)
        self.assertIn("written by a human", data["label"])


class TestAppealEndpoint(unittest.TestCase):

    def setUp(self):
        clean_test_log()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        clean_test_log()

    @patch("app.classify_with_llm", return_value=0.85)
    def test_appeal_updates_status_to_under_review(self, mock_llm):
        # Submit first
        res = self.client.post("/submit",
            data=json.dumps({"text": AI_TEXT, "creator_id": "test-user"}),
            content_type="application/json")
        content_id = json.loads(res.data)["content_id"]

        # Appeal
        res = self.client.post("/appeal",
            data=json.dumps({
                "content_id": content_id,
                "creator_reasoning": "I wrote this myself."
            }),
            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "under_review")

    @patch("app.classify_with_llm", return_value=0.85)
    def test_appeal_appears_in_audit_log(self, mock_llm):
        res = self.client.post("/submit",
            data=json.dumps({"text": AI_TEXT, "creator_id": "test-user"}),
            content_type="application/json")
        content_id = json.loads(res.data)["content_id"]

        self.client.post("/appeal",
            data=json.dumps({
                "content_id": content_id,
                "creator_reasoning": "I wrote this myself."
            }),
            content_type="application/json")

        entries = app_module.load_log()
        appeal_entries = [e for e in entries if e.get("type") == "appeal"]
        self.assertEqual(len(appeal_entries), 1)
        self.assertEqual(appeal_entries[0]["creator_reasoning"], "I wrote this myself.")

    def test_appeal_unknown_content_id_returns_404(self):
        res = self.client.post("/appeal",
            data=json.dumps({
                "content_id": "nonexistent-id",
                "creator_reasoning": "I wrote this."
            }),
            content_type="application/json")
        self.assertEqual(res.status_code, 404)

    def test_appeal_missing_reasoning_returns_400(self):
        res = self.client.post("/appeal",
            data=json.dumps({"content_id": "some-id"}),
            content_type="application/json")
        self.assertEqual(res.status_code, 400)


class TestLogEndpoint(unittest.TestCase):

    def setUp(self):
        clean_test_log()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        clean_test_log()

    def test_log_returns_entries_key(self):
        res = self.client.get("/log")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("entries", data)
        self.assertIsInstance(data["entries"], list)

    @patch("app.classify_with_llm", return_value=0.5)
    def test_log_contains_submitted_entry(self, mock_llm):
        self.client.post("/submit",
            data=json.dumps({"text": HUMAN_TEXT, "creator_id": "test-user"}),
            content_type="application/json")
        res = self.client.get("/log")
        data = json.loads(res.data)
        self.assertEqual(len(data["entries"]), 1)


class TestDashboardEndpoint(unittest.TestCase):

    def setUp(self):
        clean_test_log()
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        clean_test_log()

    def test_dashboard_returns_200(self):
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Analytics Dashboard", res.data)

    def test_index_returns_200(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Provenance Guard", res.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
