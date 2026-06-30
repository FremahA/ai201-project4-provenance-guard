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
    :root {
      --teal:    #0e7c7b;
      --teal-dk: #095c5b;
      --indigo:  #3730a3;
      --amber:   #92400e;
      --rose:    #9f1239;
      --green:   #166534;
      --bg:      #f0fdf9;
      --surface: #ffffff;
      --border:  #99f6e4;
      --text:    #1c1917;
      --muted:   #44403c;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0; padding: 0;
      min-height: 100vh;
    }
    header {
      background: linear-gradient(135deg, #0e7c7b 0%, #3730a3 100%);
      color: white;
      padding: 1.5rem 2rem;
    }
    header h1 { margin: 0; font-size: 1.6rem; letter-spacing: -0.01em; }
    header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.85; }
    nav { margin-top: 0.75rem; display: flex; gap: 1rem; flex-wrap: wrap; }
    nav a {
      color: white; text-decoration: none;
      background: rgba(255,255,255,0.18);
      padding: 0.3rem 0.85rem; border-radius: 20px;
      font-size: 0.85rem; font-weight: 500;
      border: 1px solid rgba(255,255,255,0.3);
      transition: background 0.15s;
    }
    nav a:hover, nav a:focus {
      background: rgba(255,255,255,0.32);
      outline: 2px solid white; outline-offset: 2px;
    }
    main { max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; }
    section {
      background: var(--surface);
      border-radius: 12px;
      padding: 1.75rem;
      margin-bottom: 1.75rem;
      box-shadow: 0 2px 8px rgba(14,124,123,0.08);
      border: 1px solid var(--border);
    }
    h2 {
      margin: 0 0 1.25rem;
      font-size: 1.1rem;
      color: var(--indigo);
      display: flex; align-items: center; gap: 0.5rem;
    }
    h2 .icon { font-size: 1.2rem; }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 0.4rem;
      margin-top: 1rem;
    }
    label:first-of-type { margin-top: 0; }
    input, textarea {
      width: 100%;
      padding: 0.6rem 0.8rem;
      border: 2px solid #d1fae5;
      border-radius: 8px;
      font-size: 0.95rem;
      font-family: inherit;
      color: var(--text);
      background: #f0fdf4;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    input:focus, textarea:focus {
      outline: none;
      border-color: var(--teal);
      box-shadow: 0 0 0 3px rgba(14,124,123,0.2);
      background: white;
    }
    textarea { height: 130px; resize: vertical; }
    button[type="submit"] {
      margin-top: 1.1rem;
      background: linear-gradient(135deg, var(--teal) 0%, var(--indigo) 100%);
      color: white;
      border: none;
      padding: 0.7rem 1.75rem;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s, box-shadow 0.15s;
      letter-spacing: 0.01em;
    }
    button[type="submit"]:hover { opacity: 0.9; box-shadow: 0 4px 12px rgba(14,124,123,0.3); }
    button[type="submit"]:focus {
      outline: 3px solid var(--teal); outline-offset: 3px;
    }
    .result-box {
      display: none;
      margin-top: 1.25rem;
      border-radius: 10px;
      overflow: hidden;
      border: 2px solid #d1fae5;
    }
    .result-banner {
      padding: 0.65rem 1rem;
      font-weight: 700;
      font-size: 0.95rem;
      letter-spacing: 0.01em;
    }
    .result-banner.likely_ai    { background: #fee2e2; color: var(--rose);  border-bottom: 2px solid #fca5a5; }
    .result-banner.uncertain     { background: #fef3c7; color: var(--amber); border-bottom: 2px solid #fcd34d; }
    .result-banner.likely_human  { background: #dcfce7; color: var(--green); border-bottom: 2px solid #86efac; }
    .result-banner.neutral       { background: #e0e7ff; color: var(--indigo); border-bottom: 2px solid #a5b4fc; }
    .result-label {
      padding: 0.75rem 1rem;
      font-size: 0.9rem;
      background: #fafafa;
      border-bottom: 1px solid #eee;
      font-style: italic;
      color: var(--muted);
    }
    pre.result-json {
      margin: 0;
      padding: 1rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.82rem;
      background: #f8fafc;
      color: #334155;
      max-height: 300px;
      overflow-y: auto;
    }
    .sr-only {
      position: absolute; width: 1px; height: 1px;
      padding: 0; margin: -1px; overflow: hidden;
      clip: rect(0,0,0,0); white-space: nowrap; border: 0;
    }
  </style>
</head>
<body>
  <header>
    <h1>🛡️ Provenance Guard</h1>
    <p>AI content attribution — helping platforms build trust with creators</p>
    <nav aria-label="Site navigation">
      <a href="/dashboard">📊 Analytics Dashboard</a>
      <a href="/log">📋 Audit Log</a>
    </nav>
  </header>

  <main>
    <section aria-labelledby="submit-heading">
      <h2 id="submit-heading"><span class="icon" aria-hidden="true">✍️</span> Submit Content for Analysis</h2>

      <form id="submitForm" novalidate>
        <label for="creatorId">Creator ID</label>
        <input type="text" id="creatorId" name="creatorId"
               placeholder="e.g. user-123" value="test-user"
               aria-required="true" autocomplete="off" />

        <label for="textInput">Text to Analyze</label>
        <textarea id="textInput" name="textInput"
                  placeholder="Paste a poem, short story excerpt, or blog post here..."
                  aria-required="true"
                  aria-describedby="textHint"></textarea>
        <p id="textHint" style="font-size:0.8rem;color:#6b7280;margin:0.3rem 0 0;">
          Longer texts (100+ words) produce more reliable results.
        </p>

        <button type="submit">Analyze Content</button>
      </form>

      <div class="result-box" id="submitResultBox" role="region" aria-live="polite" aria-label="Analysis result">
        <div class="result-banner" id="submitBanner"></div>
        <div class="result-label" id="submitLabel"></div>
        <pre class="result-json" id="submitJSON"></pre>
      </div>
    </section>

    <section aria-labelledby="appeal-heading">
      <h2 id="appeal-heading"><span class="icon" aria-hidden="true">🙋</span> Submit an Appeal</h2>
      <p style="font-size:0.9rem;color:var(--muted);margin:0 0 1rem;">
        If you believe your content was misclassified, you can contest the decision.
        The content ID is filled automatically after a submission above.
      </p>

      <form id="appealForm" novalidate>
        <label for="contentId">Content ID</label>
        <input type="text" id="contentId" name="contentId"
               placeholder="e.g. 4dbd781c-6850-455d-9a29-..."
               aria-required="true" autocomplete="off" />

        <label for="reasoning">Your Reasoning</label>
        <textarea id="reasoning" name="reasoning"
                  placeholder="Explain why you believe this content was misclassified..."
                  aria-required="true"></textarea>

        <button type="submit">Submit Appeal</button>
      </form>

      <div class="result-box" id="appealResultBox" role="region" aria-live="polite" aria-label="Appeal result">
        <div class="result-banner neutral" id="appealBanner"></div>
        <pre class="result-json" id="appealJSON"></pre>
      </div>
    </section>
  </main>

  <script>
    const LABELS = {
      likely_ai:    '🔴 Likely AI-Generated',
      uncertain:    '🟡 Uncertain — Could Be Either',
      likely_human: '🟢 Likely Human-Written',
    };

    document.getElementById('submitForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const box    = document.getElementById('submitResultBox');
      const banner = document.getElementById('submitBanner');
      const lbl    = document.getElementById('submitLabel');
      const json   = document.getElementById('submitJSON');

      box.style.display = 'block';
      banner.className = 'result-banner neutral';
      banner.textContent = '⏳ Analyzing…';
      lbl.textContent = '';
      json.textContent = '';

      try {
        const res  = await fetch('/submit', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            text:       document.getElementById('textInput').value,
            creator_id: document.getElementById('creatorId').value,
          })
        });
        const data = await res.json();

        const attr = data.attribution || 'neutral';
        banner.className = 'result-banner ' + attr;
        banner.textContent = LABELS[attr] || attr;
        lbl.textContent = data.label || '';
        json.textContent = JSON.stringify(data, null, 2);

        if (data.content_id) {
          document.getElementById('contentId').value = data.content_id;
        }
      } catch (err) {
        banner.className = 'result-banner neutral';
        banner.textContent = '⚠️ Error';
        json.textContent = err.message;
      }
    });

    document.getElementById('appealForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const box    = document.getElementById('appealResultBox');
      const banner = document.getElementById('appealBanner');
      const json   = document.getElementById('appealJSON');

      box.style.display = 'block';
      banner.textContent = '⏳ Submitting appeal…';
      json.textContent = '';

      try {
        const res  = await fetch('/appeal', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            content_id:        document.getElementById('contentId').value,
            creator_reasoning: document.getElementById('reasoning').value,
          })
        });
        const data = await res.json();
        banner.textContent = data.status === 'under_review'
          ? '✅ Appeal received — status: under review'
          : '⚠️ ' + (data.error || 'Unexpected response');
        json.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        banner.textContent = '⚠️ Error';
        json.textContent = err.message;
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
    :root {{
      --teal:   #0e7c7b; --teal-dk: #095c5b;
      --indigo: #3730a3;
      --rose:   #9f1239; --rose-bg: #fee2e2;
      --amber:  #92400e; --amber-bg: #fef3c7;
      --green:  #166534; --green-bg: #dcfce7;
      --blue:   #1e40af; --blue-bg:  #dbeafe;
      --purple: #5b21b6; --purple-bg:#ede9fe;
      --bg:     #f0fdf9; --surface: #ffffff;
      --border: #99f6e4; --text: #1c1917; --muted: #44403c;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg); color: var(--text);
      margin: 0; padding: 0; min-height: 100vh;
    }}
    header {{
      background: linear-gradient(135deg, #0e7c7b 0%, #3730a3 100%);
      color: white; padding: 1.5rem 2rem;
    }}
    header h1 {{ margin: 0; font-size: 1.6rem; }}
    header p  {{ margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.85; }}
    nav {{ margin-top: 0.75rem; display: flex; gap: 1rem; flex-wrap: wrap; }}
    nav a {{
      color: white; text-decoration: none;
      background: rgba(255,255,255,0.18);
      padding: 0.3rem 0.85rem; border-radius: 20px;
      font-size: 0.85rem; font-weight: 500;
      border: 1px solid rgba(255,255,255,0.3);
    }}
    nav a:hover, nav a:focus {{
      background: rgba(255,255,255,0.32);
      outline: 2px solid white; outline-offset: 2px;
    }}
    main {{ max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }}
    h2 {{ font-size: 1rem; color: var(--indigo); margin: 0 0 1.25rem;
          text-transform: uppercase; letter-spacing: 0.06em; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem; margin-bottom: 2rem;
    }}
    .card {{
      background: var(--surface); border-radius: 12px;
      padding: 1.25rem 1.5rem;
      box-shadow: 0 2px 8px rgba(14,124,123,0.08);
      border: 1px solid var(--border);
    }}
    .card .clabel {{
      font-size: 0.75rem; text-transform: uppercase;
      letter-spacing: 0.06em; font-weight: 600;
      margin-bottom: 0.5rem;
    }}
    .card .cvalue {{ font-size: 2.2rem; font-weight: 800; line-height: 1; }}
    .card .cpct   {{ font-size: 0.85rem; font-weight: 500; margin-top: 0.2rem; }}
    .card.total   {{ border-top: 4px solid var(--teal);   }}
    .card.ai      {{ border-top: 4px solid #f87171;
                     background: var(--rose-bg); }}
    .card.ai      .clabel {{ color: var(--rose); }}
    .card.ai      .cvalue {{ color: var(--rose); }}
    .card.uncertain {{ border-top: 4px solid #fbbf24;
                       background: var(--amber-bg); }}
    .card.uncertain .clabel {{ color: var(--amber); }}
    .card.uncertain .cvalue {{ color: var(--amber); }}
    .card.human   {{ border-top: 4px solid #4ade80;
                     background: var(--green-bg); }}
    .card.human   .clabel {{ color: var(--green); }}
    .card.human   .cvalue {{ color: var(--green); }}
    .card.appeals {{ border-top: 4px solid #818cf8;
                     background: var(--purple-bg); }}
    .card.appeals .clabel {{ color: var(--purple); }}
    .card.appeals .cvalue {{ color: var(--purple); }}
    table {{
      width: 100%; border-collapse: collapse;
      background: var(--surface); border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(14,124,123,0.08);
      border: 1px solid var(--border);
    }}
    thead tr {{ background: linear-gradient(135deg, #0e7c7b, #3730a3); }}
    th {{
      color: white; text-align: left;
      padding: 0.85rem 1.1rem; font-size: 0.82rem;
      text-transform: uppercase; letter-spacing: 0.05em;
    }}
    td {{ padding: 0.85rem 1.1rem; border-bottom: 1px solid #e0fdf4; font-size: 0.9rem; }}
    tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover {{ background: #f0fdf9; }}
    .badge {{
      display: inline-block; padding: 0.2rem 0.65rem;
      border-radius: 20px; font-size: 0.78rem; font-weight: 700;
    }}
    .badge.ai       {{ background: var(--rose-bg);   color: var(--rose);   }}
    .badge.uncertain{{ background: var(--amber-bg);  color: var(--amber);  }}
    .badge.human    {{ background: var(--green-bg);  color: var(--green);  }}
    .badge.appeals  {{ background: var(--purple-bg); color: var(--purple); }}
    .section-wrap {{
      background: var(--surface); border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(14,124,123,0.08);
      border: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <header>
    <h1>📊 Analytics Dashboard</h1>
    <p>Live detection patterns — Provenance Guard</p>
    <nav aria-label="Site navigation">
      <a href="/">✍️ Submit Content</a>
      <a href="/log">📋 Audit Log</a>
    </nav>
  </header>

  <main>
    <section aria-label="Summary statistics">
      <div class="grid">
        <div class="card total">
          <div class="clabel">Total Submissions</div>
          <div class="cvalue">{total}</div>
          <div class="cpct" style="color:var(--muted);">all time</div>
        </div>
        <div class="card ai" aria-label="Flagged as AI: {counts['likely_ai']} submissions, {pct(counts['likely_ai'])} percent">
          <div class="clabel">🔴 Flagged as AI</div>
          <div class="cvalue">{counts['likely_ai']}</div>
          <div class="cpct">{pct(counts['likely_ai'])}% of total</div>
        </div>
        <div class="card uncertain" aria-label="Uncertain: {counts['uncertain']} submissions, {pct(counts['uncertain'])} percent">
          <div class="clabel">🟡 Uncertain</div>
          <div class="cvalue">{counts['uncertain']}</div>
          <div class="cpct">{pct(counts['uncertain'])}% of total</div>
        </div>
        <div class="card human" aria-label="Likely human: {counts['likely_human']} submissions, {pct(counts['likely_human'])} percent">
          <div class="clabel">🟢 Likely Human</div>
          <div class="cvalue">{counts['likely_human']}</div>
          <div class="cpct">{pct(counts['likely_human'])}% of total</div>
        </div>
        <div class="card appeals" aria-label="Appeal rate: {appeal_rate} percent">
          <div class="clabel">🙋 Appeal Rate</div>
          <div class="cvalue">{appeal_rate}%</div>
          <div class="cpct">{appeal_count} appeal(s) filed</div>
        </div>
      </div>
    </section>

    <section aria-label="Detailed breakdown" style="margin-top:1.5rem;">
      <div class="section-wrap">
        <h2>Detection Breakdown</h2>
        <table>
          <thead>
            <tr>
              <th scope="col">Attribution</th>
              <th scope="col">Count</th>
              <th scope="col">% of Submissions</th>
              <th scope="col">Avg Confidence Score</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><span class="badge ai">Likely AI</span></td>
              <td>{counts['likely_ai']}</td>
              <td>{pct(counts['likely_ai'])}%</td>
              <td>{avg_conf('likely_ai')}</td>
            </tr>
            <tr>
              <td><span class="badge uncertain">Uncertain</span></td>
              <td>{counts['uncertain']}</td>
              <td>{pct(counts['uncertain'])}%</td>
              <td>{avg_conf('uncertain')}</td>
            </tr>
            <tr>
              <td><span class="badge human">Likely Human</span></td>
              <td>{counts['likely_human']}</td>
              <td>{pct(counts['likely_human'])}%</td>
              <td>{avg_conf('likely_human')}</td>
            </tr>
            <tr>
              <td><span class="badge appeals">Appeals</span></td>
              <td>{appeal_count}</td>
              <td>{appeal_rate}%</td>
              <td>—</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
