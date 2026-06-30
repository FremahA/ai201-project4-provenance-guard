# Provenance Guard

A backend API for AI content attribution on creative writing platforms. Provenance Guard analyzes submitted text using a multi-signal detection pipeline, returns a confidence score and plain-language transparency label, and provides an appeals workflow for contested classifications.

## Quickstart

```bash
# 1. Clone and set up virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here

# 4. Run the server
python app.py
```

Visit `http://127.0.0.1:5000` for the submission form, or `http://127.0.0.1:5000/dashboard` for the analytics dashboard.

---

## Architecture Overview

A submitted piece of text travels through the following components:

**Submission flow:** `POST /submit` receives the text and creator ID → Signal 1 (Groq LLM) analyzes semantic and stylistic coherence → Signal 2 (stylometric heuristics) computes sentence length variance and vocabulary diversity → Signal 3 (punctuation/complexity) measures punctuation density and sentence complexity variance → the three scores are combined into a single confidence score using weighted averaging → the confidence score maps to one of three transparency label variants → a structured entry is written to the audit log → the full result is returned to the caller as JSON.

**Appeal flow:** `POST /appeal` receives a content ID and creator reasoning → the original audit log entry is located and its status updated to `under_review` → a separate appeal record is appended to the log → a confirmation is returned to the caller.

```
POST /submit
  │
  ├─► [Signal 1] Groq LLM (llama-3.3-70b-versatile)
  │     └─ Returns llm_score (0.0–1.0)
  │
  ├─► [Signal 2] Stylometric Heuristics
  │     └─ Sentence length variance + type-token ratio
  │        Returns stylo_score (0.0–1.0)
  │
  ├─► [Signal 3] Punctuation & Complexity
  │     └─ Punctuation density + sentence complexity variance
  │        Returns punct_score (0.0–1.0)
  │
  ├─► Confidence = (0.60 × llm_score) + (0.25 × stylo_score) + (0.15 × punct_score)
  │
  ├─► Transparency Label Generator
  │
  ├─► Audit Log (audit_log.json)
  │
  └─► JSON Response

POST /appeal
  │
  ├─► Locate original entry by content_id
  ├─► Update status → "under_review"
  ├─► Append appeal record to audit log
  └─► JSON Confirmation
```

---

## Detection Signals

### Signal 1: LLM Classification (Groq — llama-3.3-70b-versatile)

**What it measures:** Semantic and stylistic coherence holistically. The model evaluates whether text reads as human-authored based on naturalness of expression, idiomatic variation, tonal consistency, and whether the writing "sounds" like a specific person wrote it in a specific moment. It returns a structured JSON score between 0.0 and 1.0.

**Why chosen:** LLMs capture meaning-level patterns that statistical signals miss — overly hedged phrasing, suspiciously balanced argumentation, unnaturally smooth transitions, and the absence of personal voice. It is the strongest and most reliable signal for medium-to-long texts.

**What it misses:** Short texts (under 50 words) give the model too little signal. Very formal human writing — academic papers, legal briefs — can score as AI-like. The model has no memory of what AI output looks like for niche genres such as dialect poetry or experimental prose. A sophisticated user could craft text specifically designed to fool the LLM.

---

### Signal 2: Stylometric Heuristics (Pure Python)

**What it measures:** Two statistical properties of the text's structure:

- **Sentence length variance:** Standard deviation of word counts per sentence. AI text tends to produce sentences of similar length (low variance); human writing has more rhythmic irregularity. Low standard deviation pushes the score higher.
- **Type-token ratio (TTR):** Unique words divided by total words. AI text reuses vocabulary more frequently (lower TTR); human writing is lexically more diverse. A TTR below ~0.6 nudges the score upward.

**Why chosen:** Completely independent from the LLM — no API call, no semantic interpretation. Captures structural uniformity that LLMs sometimes overlook in polished AI output. Computed in pure Python with no external libraries.

**What it misses:** Short texts (fewer than 3 sentences) produce unreliable variance estimates. Poetry and minimalist prose intentionally have low sentence-length variance and may be incorrectly flagged. Non-native English speakers who write in a consistent, formal register may score higher than expected.

---

### Signal 3: Punctuation & Complexity (Pure Python — Ensemble)

**What it measures:** Two additional structural properties:

- **Punctuation density:** Punctuation marks per word. Human writing uses more varied punctuation (em-dashes, ellipses, exclamation marks mid-sentence). AI text is punctuated conservatively and uniformly. Low density pushes the score higher.
- **Sentence complexity variance:** Standard deviation of a per-sentence complexity score (commas + conjunctions per sentence). AI text tends toward consistently multi-clause sentences; human writing varies more. Low variance pushes the score higher.

**Why chosen:** Independent from both TTR/variance and the LLM. Punctuation patterns are a genuine stylistic fingerprint that neither of the other signals captures, making the three-signal combination more informative than any two alone.

**What it misses:** Non-native English speakers may use punctuation sparingly by habit. Minimalist prose styles (Hemingway-like writing) use few punctuation marks by design and may score as AI-like on this signal.

---

## Confidence Scoring

All three signals produce a score between 0.0 and 1.0, where higher means more likely AI-generated. They are combined using a weighted average:

```
confidence = (0.60 × llm_score) + (0.25 × stylo_score) + (0.15 × punct_score)
```

The LLM score carries 60% of the weight because it captures semantic meaning — the hardest dimension to fake and the most reliable for texts of reasonable length. Stylometric heuristics carry 25% as structural corroboration. Punctuation/complexity carries 15% as a tiebreaker for borderline cases.

**Thresholds:**

| Confidence Score | Attribution   | Label Variant        |
|------------------|---------------|----------------------|
| 0.75 – 1.00      | likely_ai     | High-confidence AI   |
| 0.40 – 0.74      | uncertain     | Uncertain            |
| 0.00 – 0.39      | likely_human  | High-confidence Human |

The threshold for `likely_human` is intentionally tight (≤ 0.39). A false positive — labeling a human writer's work as AI — is more harmful than a false negative on a creative writing platform. The uncertain band is wide (0.40–0.74) to reflect this asymmetry. When the system is not confident, it says so rather than forcing a binary call.

**Validation — Example Submissions:**

*Submission 1 — Human-written text (casual, personal narrative):*
```json
{
  "text": "The sun dipped below the horizon, painting the sky in hues of amber and rose. I sat on the porch, coffee in hand, watching the neighborhood slowly go quiet.",
  "attribution": "likely_human",
  "confidence": 0.2925,
  "llm_score": 0.2,
  "stylo_score": 0.5082,
  "punct_score": 0.35
}
```

*Submission 2 — AI-generated text (short, formal):*
```json
{
  "text": "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment.",
  "attribution": "uncertain",
  "confidence": 0.6556,
  "llm_score": 0.8,
  "stylo_score": 0.3188,
  "punct_score": 0.42
}
```

*Submission 3 — AI-generated text (long, multi-paragraph, highly uniform):*
```json
{
  "text": "The rapid advancement of artificial intelligence technologies has fundamentally transformed the landscape of modern industries...[full passage]",
  "attribution": "likely_ai",
  "confidence": 0.7757,
  "llm_score": 0.9,
  "stylo_score": 0.4636,
  "punct_score": 0.7986
}
```

The confidence scores span the full range: 0.29 (likely human) → 0.66 (uncertain) → 0.78 (likely AI). All three transparency label variants are reachable. The short AI text landed in `uncertain` because the stylometric and punctuation signals are less decisive on 3-sentence samples — a known limitation documented below. The longer AI passage gave all three signals enough text to work with, pushing the combined score above the 0.75 threshold. This is honest behavior: the system reflects genuine uncertainty on short texts rather than forcing a high-confidence label.

---

## Transparency Label

Three label variants, written exactly as they appear in the API response:

**High-confidence AI** (`confidence ≥ 0.75`):
> "This content was likely generated by an AI tool. Our system analyzed the writing style and structure and found strong indicators of AI authorship. If you created this yourself, you can submit an appeal."

**Uncertain** (`0.40 ≤ confidence < 0.75`):
> "Our system wasn't able to make a confident determination about this content. It shows some characteristics of AI-generated text, but also signs of human authorship. No action has been taken. If you have concerns, you can submit an appeal."

**High-confidence Human** (`confidence < 0.40`):
> "This content appears to have been written by a human. Our system found strong indicators of human authorship in the writing style and structure."

Labels use plain language throughout — no terms like "classifier output," "logit score," or "confidence threshold." A non-technical reader understands what each label means and what action, if any, they should take. The AI label explicitly offers an appeal path because false positives are the most harmful outcome.

---

## Appeals Workflow

Any creator can contest a classification by submitting a `POST /appeal` request with their `content_id` and a plain-text explanation of why they believe the classification is wrong.

```bash
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "4dbd781c-6850-455d-9a29-4e530877ac6b",
    "creator_reasoning": "I wrote this myself from personal experience. My writing style tends to be descriptive and literary."
  }'
```

Response:
```json
{
  "content_id": "4dbd781c-6850-455d-9a29-4e530877ac6b",
  "status": "under_review",
  "message": "Your appeal has been received and will be reviewed by a human moderator."
}
```

The system updates the original submission entry's status to `under_review` and appends a separate appeal record to the audit log containing the creator's reasoning and a timestamp. Automated re-classification is not performed — appeals require human review.

---

## Rate Limiting

The `POST /submit` endpoint is rate limited to **10 requests per minute** and **100 requests per day** per IP address.

**Reasoning:** A human writer submitting their own creative work might post a handful of pieces in a session — 10 per minute is generous for legitimate use. An automated script flooding the system to probe the classifier or exhaust the Groq API quota would hit the limit quickly. 100 per day prevents overnight abuse while comfortably accommodating any realistic single-user session. The per-minute limit is the primary protection against burst attacks; the per-day limit protects the Groq API budget.

When the limit is exceeded, the server returns HTTP `429 Too Many Requests`:

```
200
200
200
200
200
200
200
200
200
200
429
429
```

---

## Audit Log

Every attribution decision and appeal is written to `audit_log.json` as a structured JSON entry. The log is accessible via `GET /log`.

**Submission entry:**
```json
{
  "content_id": "4dbd781c-6850-455d-9a29-4e530877ac6b",
  "creator_id": "test-user-1",
  "timestamp": "2026-06-30T20:46:44.002831+00:00",
  "attribution": "likely_human",
  "confidence": 0.2925,
  "llm_score": 0.2,
  "stylo_score": 0.5082,
  "punct_score": 0.35,
  "status": "under_review",
  "type": "submission"
}
```

**Appeal entry:**
```json
{
  "content_id": "4dbd781c-6850-455d-9a29-4e530877ac6b",
  "timestamp": "2026-06-30T20:49:51.883753+00:00",
  "type": "appeal",
  "creator_reasoning": "I wrote this myself from personal experience watching the sunset in my neighborhood. My writing style tends to be descriptive and literary.",
  "status": "under_review"
}
```

Each submission entry includes: timestamp, content ID, creator ID, attribution result, combined confidence score, all three individual signal scores, and status. Appeal entries include the original content ID, the creator's reasoning, and timestamp — making it straightforward for a human reviewer to view the original decision alongside the appeal.

---

## Analytics Dashboard

Visit `http://127.0.0.1:5000/dashboard` for a live view of detection patterns, including total submissions, breakdown by attribution category, appeal rate, and average confidence score per category. The dashboard reads directly from the audit log on each page load.

---

## Stretch Features Implemented

### Ensemble Detection (3 Signals)
The detection pipeline uses three independent signals rather than two. The third signal — punctuation density and sentence complexity variance — is computed in pure Python and captures stylistic patterns that neither the LLM nor the vocabulary/rhythm heuristics detect. Weighting: 60% LLM / 25% stylometric / 15% punctuation-complexity.

### Analytics Dashboard
A browser-accessible dashboard at `GET /dashboard` shows live detection patterns (counts and percentages by attribution category), appeal rate, and average confidence score per category.

---

## Known Limitations

**Formal human writing is the most likely false positive.** Academic essays, legal briefs, and research reports written by humans share many surface features with AI-generated text: low sentence-length variance (structured argumentation tends toward similar-length sentences), lower vocabulary diversity than casual writing (domain-specific terms repeat), and conservative punctuation. A human scholar writing in their second language is especially at risk. The stylometric and punctuation signals cannot distinguish between "formal human register" and "AI uniformity" — they measure the same surface properties. The LLM signal mitigates this somewhat, but a formal human text that also reads as stylistically polished may still land in the `uncertain` band. The wide uncertain range (0.40–0.74) is specifically designed to avoid false positives in this case, but it cannot eliminate them.

---

## Spec Reflection

**One way the spec helped:** Defining the three transparency label variants in `planning.md` before writing any code made implementing `generate_label()` trivial — the function is a direct translation of the written spec. Having the exact text decided upfront also prevented scope creep during implementation.

**One way implementation diverged:** The original spec weighted signals at 70/30 (LLM/stylometric). When the third signal was added as a stretch feature, the weighting was updated to 60/25/15. More significantly, the AI text test case scored `uncertain` (0.66) rather than `likely_ai` as expected during planning. This revealed that the stylometric and punctuation signals are weak on short texts — a limitation documented in the edge cases section of planning.md but underestimated in practice. Rather than adjusting the thresholds to force a `likely_ai` result, the system was left as-is: the `uncertain` label is the honest answer for a short AI-generated paragraph, and the planning documents note this as a known limitation.

---

## AI Usage

**Instance 1: Flask app skeleton and Groq signal function**

Directed the AI to generate a Flask app skeleton with a `POST /submit` route stub and a `classify_with_llm()` function, providing the Detection Signals section of `planning.md` and the architecture diagram as context. The AI produced working code but the prompt parsing was fragile — it used a simple string search for `{` that would fail on responses with preamble text. Revised this to use `re.search(r'\{.*\}', raw, re.DOTALL)` so the JSON is extracted reliably regardless of what text precedes it. Also added explicit clamping (`max(0.0, min(1.0, score))`) because the AI-generated version trusted the model's output range without bounds checking.

**Instance 2: Stylometric signal and confidence scoring logic**

Directed the AI to generate `compute_stylometric_score()` and `compute_confidence()`, providing the uncertainty representation section of `planning.md` with specific thresholds. The AI implemented a sentence splitter that split on periods only, which broke on abbreviations (e.g., "Dr. Smith" split into two fragments). Revised the splitter to use `re.split(r'[.!?]+', text)` to handle all sentence-ending punctuation. Also revised the TTR normalization range — the AI used 0.3–0.9 as the normalization bounds, but testing showed human writing TTR rarely drops below 0.55 on short texts, so the range was tightened to 0.40–0.85 to produce more meaningful score variation.
