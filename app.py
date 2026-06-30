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
# Signal 3: Punctuation & Complexity Score
# ---------------------------------------------------------------------------
def compute_punctuation_score(text):
    """
    Measures two additional structural properties:
    1. Punctuation density — punctuation marks per word.
       AI text uses punctuation conservatively and uniformly.
       Low density → higher AI score.
    2. Average sentence complexity — conjunctions + commas per sentence.
       AI text tends toward consistent multi-clause sentences.
       Very high or very uniform complexity → higher AI score.

    Returns a float 0.0–1.0 where 1.0 = more likely AI.
    """
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.5

    # --- Metric 1: Punctuation density ---
    punct_marks = re.findall(r'[!?,;:\-\—\…\(\)\"\']', text)
    density = len(punct_marks) / len(words)
    # Human writing: density typically 0.08–0.25+
    # AI writing: density typically 0.03–0.10
    # Low density → high AI score
    # Normalize: density 0.03 → score 1.0, density 0.20+ → score 0.0
    density_score = max(0.0, min(1.0, (0.20 - density) / 0.17))

    # --- Metric 2: Sentence complexity variance ---
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        complexity_score = 0.5
    else:
        # Complexity = commas + conjunctions per sentence
        conjunctions = {'and', 'but', 'or', 'so', 'yet', 'because', 'although',
                        'while', 'however', 'therefore', 'furthermore', 'moreover'}
        def sentence_complexity(s):
            commas = s.count(',')
            words_in_s = set(re.findall(r'\b\w+\b', s.lower()))
            conj_count = len(words_in_s & conjunctions)
            return commas + conj_count

        complexities = [sentence_complexity(s) for s in sentences]
        mean_c = sum(complexities) / len(complexities)
        variance_c = sum((c - mean_c) ** 2 for c in complexities) / len(complexities)
        std_c = math.sqrt(variance_c)
        # Low variance in complexity = AI-like (consistent sentence structure)
        # High variance = more human
        clamped_c = max(0.0, min(5.0, std_c))
        complexity_score = 1.0 - (clamped_c / 5.0)

    return round((0.5 * density_score) + (0.5 * complexity_score), 4)


# ---------------------------------------------------------------------------
# Confidence scoring + label generation
# ---------------------------------------------------------------------------
def compute_confidence(llm_score, stylo_score, punct_score):
    """
    Combines 3 signals using 60/25/15 weighting.
    LLM dominates (semantic meaning), stylometrics second (structure),
    punctuation/complexity third (stylistic fingerprint).
    """
    return round((0.60 * llm_score) + (0.25 * stylo_score) + (0.15 * punct_score), 4)


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
@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Provenance Guard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #f5f5f5; color: #222; margin: 0; padding: 2rem; max-width: 700px; }
    h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
    p.sub { color: #666; margin-top: 0; margin-bottom: 2rem; font-size: 0.9rem; }
    label { display: block; font-size: 0.85rem; font-weight: 600;
            margin-bottom: 0.4rem; margin-top: 1rem; }
    input, textarea { width: 100%; box-sizing: border-box; padding: 0.6rem 0.75rem;
                      border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem;
                      font-family: inherit; }
    textarea { height: 120px; resize: vertical; }
    button { margin-top: 1rem; background: #333; color: white; border: none;
             padding: 0.65rem 1.5rem; border-radius: 6px; font-size: 0.95rem;
             cursor: pointer; }
    button:hover { background: #555; }
    pre { background: white; border-radius: 8px; padding: 1rem;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); white-space: pre-wrap;
          word-break: break-word; font-size: 0.85rem; }
    hr { border: none; border-top: 1px solid #ddd; margin: 2.5rem 0; }
    a { color: #333; font-size: 0.85rem; }
    .nav { margin-bottom: 1.5rem; }
  </style>
</head>
<body>
  <h1>Provenance Guard</h1>
  <p class="sub">AI content attribution API &nbsp;·&nbsp;
    <a href="/dashboard">Analytics Dashboard</a> &nbsp;·&nbsp;
    <a href="/log">Audit Log (JSON)</a></p>

  <h2 style="font-size:1.1rem;">Submit Content</h2>
  <form id="submitForm">
    <label>Creator ID</label>
    <input type="text" id="creatorId" placeholder="e.g. user-123" value="test-user" />
    <label>Text to Analyze</label>
    <textarea id="textInput" placeholder="Paste a poem, story excerpt, or blog post here..."></textarea>
    <button type="submit">Analyze</button>
  </form>
  <pre id="submitResult" style="display:none;"></pre>

  <hr>

  <h2 style="font-size:1.1rem;">Submit an Appeal</h2>
  <form id="appealForm">
    <label>Content ID (from a previous submission)</label>
    <input type="text" id="contentId" placeholder="e.g. 4dbd781c-..." />
    <label>Your Reasoning</label>
    <textarea id="reasoning" placeholder="Explain why you believe this was misclassified..."></textarea>
    <button type="submit">Submit Appeal</button>
  </form>
  <pre id="appealResult" style="display:none;"></pre>

  <script>
    document.getElementById('submitForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const result = document.getElementById('submitResult');
      result.style.display = 'block';
      result.textContent = 'Analyzing...';
      try {
        const res = await fetch('/submit', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            text: document.getElementById('textInput').value,
            creator_id: document.getElementById('creatorId').value
          })
        });
        const data = await res.json();
        result.textContent = JSON.stringify(data, null, 2);
        if (data.content_id) {
          document.getElementById('contentId').value = data.content_id;
        }
      } catch (err) {
        result.textContent = 'Error: ' + err.message;
      }
    });

    document.getElementById('appealForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const result = document.getElementById('appealResult');
      result.style.display = 'block';
      result.textContent = 'Submitting appeal...';
      try {
        const res = await fetch('/appeal', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            content_id: document.getElementById('contentId').value,
            creator_reasoning: document.getElementById('reasoning').value
          })
        });
        const data = await res.json();
        result.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        result.textContent = 'Error: ' + err.message;
      }
    });
  </script>
</body>
</html>"""


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

    # Run all 3 signals
    llm_score = classify_with_llm(text)
    stylo_score = compute_stylometric_score(text)
    punct_score = compute_punctuation_score(text)

    # Combine into confidence score (60/25/15 weighting)
    confidence = compute_confidence(llm_score, stylo_score, punct_score)
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
        "punct_score": round(punct_score, 4),
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
        "punct_score": round(punct_score, 4),
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


@app.route("/dashboard", methods=["GET"])
def dashboard():
    entries = load_log()

    submissions = [e for e in entries if e.get("type") == "submission"]
    appeals = [e for e in entries if e.get("type") == "appeal"]

    total = len(submissions)
    appeal_count = len(appeals)
    appeal_rate = round((appeal_count / total * 100), 1) if total > 0 else 0

    counts = {"likely_ai": 0, "uncertain": 0, "likely_human": 0}
    confidence_sums = {"likely_ai": 0.0, "uncertain": 0.0, "likely_human": 0.0}

    for e in submissions:
        attr = e.get("attribution", "uncertain")
        if attr in counts:
            counts[attr] += 1
            confidence_sums[attr] += e.get("confidence", 0.0)

    def avg_conf(attr):
        if counts[attr] == 0:
            return "N/A"
        return round(confidence_sums[attr] / counts[attr], 3)

    def pct(n):
        return round(n / total * 100, 1) if total > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Provenance Guard — Analytics Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f5f5; color: #222; margin: 0; padding: 2rem; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    p.sub {{ color: #666; margin-top: 0; margin-bottom: 2rem; font-size: 0.9rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
             gap: 1rem; margin-bottom: 2rem; }}
    .card {{ background: white; border-radius: 8px; padding: 1.25rem 1.5rem;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .card .label {{ font-size: 0.8rem; text-transform: uppercase; color: #888;
                    letter-spacing: 0.05em; margin-bottom: 0.4rem; }}
    .card .value {{ font-size: 2rem; font-weight: 700; }}
    .card .value.ai {{ color: #c0392b; }}
    .card .value.uncertain {{ color: #e67e22; }}
    .card .value.human {{ color: #27ae60; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    th {{ background: #333; color: white; text-align: left;
          padding: 0.75rem 1rem; font-size: 0.85rem; }}
    td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
    tr:last-child td {{ border-bottom: none; }}
    .refresh {{ float: right; font-size: 0.8rem; color: #999; margin-top: 0.25rem; }}
  </style>
</head>
<body>
  <h1>Provenance Guard — Analytics Dashboard</h1>
  <p class="sub">Live view from audit log &nbsp;·&nbsp;
     <span class="refresh">Auto-refresh: reload page for latest data</span></p>

  <div class="grid">
    <div class="card">
      <div class="label">Total Submissions</div>
      <div class="value">{total}</div>
    </div>
    <div class="card">
      <div class="label">Flagged as AI</div>
      <div class="value ai">{counts['likely_ai']} <small style="font-size:1rem;color:#999">({pct(counts['likely_ai'])}%)</small></div>
    </div>
    <div class="card">
      <div class="label">Uncertain</div>
      <div class="value uncertain">{counts['uncertain']} <small style="font-size:1rem;color:#999">({pct(counts['uncertain'])}%)</small></div>
    </div>
    <div class="card">
      <div class="label">Likely Human</div>
      <div class="value human">{counts['likely_human']} <small style="font-size:1rem;color:#999">({pct(counts['likely_human'])}%)</small></div>
    </div>
    <div class="card">
      <div class="label">Appeal Rate</div>
      <div class="value">{appeal_rate}%</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Attribution Category</th>
        <th>Count</th>
        <th>% of Submissions</th>
        <th>Avg Confidence Score</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Likely AI</td>
        <td>{counts['likely_ai']}</td>
        <td>{pct(counts['likely_ai'])}%</td>
        <td>{avg_conf('likely_ai')}</td>
      </tr>
      <tr>
        <td>Uncertain</td>
        <td>{counts['uncertain']}</td>
        <td>{pct(counts['uncertain'])}%</td>
        <td>{avg_conf('uncertain')}</td>
      </tr>
      <tr>
        <td>Likely Human</td>
        <td>{counts['likely_human']}</td>
        <td>{pct(counts['likely_human'])}%</td>
        <td>{avg_conf('likely_human')}</td>
      </tr>
      <tr style="font-weight:600; background:#fafafa;">
        <td>Total / Appeals</td>
        <td>{total}</td>
        <td>—</td>
        <td>{appeal_count} appeal(s) filed ({appeal_rate}%)</td>
      </tr>
    </tbody>
  </table>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
