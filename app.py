import os
import json
import uuid
import math
import re
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
# 10 submissions per minute / 100 per day per IP.
# Reasoning: a human writer submitting their own work might submit a handful
# of pieces in a session, but not dozens per minute. 10/min is generous for
# legitimate use while blocking automated flooding. 100/day prevents a script
# from exhausting the Groq quota overnight.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Audit log helpers
# ---------------------------------------------------------------------------
AUDIT_LOG_PATH = "audit_log.json"


def load_log():
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH, "r") as f:
        return json.load(f)


def save_log(entries):
    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def append_log(entry):
    entries = load_log()
    entries.append(entry)
    save_log(entries)


# ---------------------------------------------------------------------------
# Signal 1: LLM Classification (Groq)
# ---------------------------------------------------------------------------
def classify_with_llm(text):
    """
    Sends text to Groq and asks for an AI-likelihood score.
    Returns a float 0.0–1.0 where 1.0 = very likely AI-generated.
    """
    prompt = f"""You are an expert at distinguishing AI-generated text from human-written text.

Analyze the following text and return ONLY a JSON object with this exact structure:
{{"ai_score": <float between 0.0 and 1.0>, "reasoning": "<one sentence>"}}

Where ai_score means:
- 1.0 = almost certainly AI-generated
- 0.0 = almost certainly written by a human
- 0.5 = genuinely uncertain

Do not include any other text outside the JSON.

Text to analyze:
\"\"\"
{text}
\"\"\"
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON even if there's extra whitespace or markdown fencing
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = float(data.get("ai_score", 0.5))
            return max(0.0, min(1.0, score))  # clamp to [0, 1]
        else:
            return 0.5  # fallback if parse fails

    except Exception:
        return 0.5  # fallback on any API error


# ---------------------------------------------------------------------------
# Signal 2: Stylometric Heuristics
# ---------------------------------------------------------------------------
def compute_stylometric_score(text):
    """
    Computes two structural metrics and combines them into a 0.0–1.0 score.
    Higher = more likely AI-generated.

    Metrics:
    1. Sentence length variance — AI text tends to have uniform sentence lengths.
       Low variance → higher AI score.
    2. Type-token ratio (TTR) — unique words / total words.
       AI text reuses vocabulary more. Low TTR → higher AI score.
    """
    # Split into sentences (simple split on . ! ?)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # --- Metric 1: Sentence length variance ---
    if len(sentences) < 2:
        variance_score = 0.5  # not enough data
    else:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)

        # Human writing: std_dev typically 4–12+
        # AI writing: std_dev typically 1–5
        # Normalize: low std_dev → high AI score
        # Clamp std_dev to [0, 15], then invert
        clamped = max(0.0, min(15.0, std_dev))
        variance_score = 1.0 - (clamped / 15.0)

    # --- Metric 2: Type-token ratio ---
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        ttr_score = 0.5  # not enough data
    else:
        ttr = len(set(words)) / len(words)
        # Human writing TTR: typically 0.55–0.85
        # AI writing TTR: typically 0.40–0.65
        # Low TTR → high AI score
        # Normalize: TTR of 0.4 → score 1.0, TTR of 0.85 → score 0.0
        ttr_score = max(0.0, min(1.0, (0.85 - ttr) / 0.45))

    # Combine: equal weight between the two sub-metrics
    stylo_score = (0.5 * variance_score) + (0.5 * ttr_score)
    return round(stylo_score, 4)


# ---------------------------------------------------------------------------
# Confidence scoring + label generation
# ---------------------------------------------------------------------------
def compute_confidence(llm_score, stylo_score):
    """
    Combines signals using 70/30 weighting.
    LLM score weighted higher — captures semantic meaning better.
    """
    return round((0.7 * llm_score) + (0.3 * stylo_score), 4)


def get_attribution(confidence):
    if confidence >= 0.75:
        return "likely_ai"
    elif confidence >= 0.40:
        return "uncertain"
    else:
        return "likely_human"


def generate_label(confidence):
    """
    Returns plain-language transparency label based on confidence score.
    Three variants:
      - likely_ai    (≥ 0.75)
      - uncertain    (0.40–0.74)
      - likely_human (< 0.40)
    """
    if confidence >= 0.75:
        return (
            "This content was likely generated by an AI tool. Our system analyzed "
            "the writing style and structure and found strong indicators of AI authorship. "
            "If you created this yourself, you can submit an appeal."
        )
    elif confidence >= 0.40:
        return (
            "Our system wasn't able to make a confident determination about this content. "
            "It shows some characteristics of AI-generated text, but also signs of human "
            "authorship. No action has been taken. If you have concerns, you can submit an appeal."
        )
    else:
        return (
            "This content appears to have been written by a human. Our system found "
            "strong indicators of human authorship in the writing style and structure."
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(force=True)

    text = data.get("text", "").strip()
    creator_id = data.get("creator_id", "").strip()

    if not text:
        return jsonify({"error": "text field is required"}), 400
    if not creator_id:
        return jsonify({"error": "creator_id field is required"}), 400

    content_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Run both signals
    llm_score = classify_with_llm(text)
    stylo_score = compute_stylometric_score(text)

    # Combine into confidence score
    confidence = compute_confidence(llm_score, stylo_score)
    attribution = get_attribution(confidence)
    label = generate_label(confidence)

    # Write to audit log
    log_entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": round(llm_score, 4),
        "stylo_score": round(stylo_score, 4),
        "status": "classified",
        "type": "submission",
    }
    append_log(log_entry)

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": round(llm_score, 4),
        "stylo_score": round(stylo_score, 4),
        "label": label,
        "status": "classified",
    })


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(force=True)

    content_id = data.get("content_id", "").strip()
    creator_reasoning = data.get("creator_reasoning", "").strip()

    if not content_id:
        return jsonify({"error": "content_id is required"}), 400
    if not creator_reasoning:
        return jsonify({"error": "creator_reasoning is required"}), 400

    entries = load_log()

    # Find and update the original submission entry
    original_found = False
    for entry in entries:
        if entry.get("content_id") == content_id and entry.get("type") == "submission":
            entry["status"] = "under_review"
            original_found = True
            break

    if not original_found:
        return jsonify({"error": "content_id not found"}), 404

    # Append a separate appeal record
    appeal_entry = {
        "content_id": content_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "appeal",
        "creator_reasoning": creator_reasoning,
        "status": "under_review",
    }
    entries.append(appeal_entry)
    save_log(entries)

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "message": "Your appeal has been received and will be reviewed by a human moderator.",
    })


@app.route("/log", methods=["GET"])
def get_log():
    entries = load_log()
    # Return most recent 50 entries, newest first
    return jsonify({"entries": list(reversed(entries[-50:]))})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
