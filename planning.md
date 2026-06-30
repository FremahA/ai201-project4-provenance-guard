# Provenance Guard — Planning Document

## Architecture

### Submission Flow

```
POST /submit
  │
  ├─► [Signal 1] Groq LLM Classification
  │     └─ Sends text to llama-3.3-70b-versatile
  │        Returns: llm_score (0.0–1.0, higher = more likely AI)
  │
  ├─► [Signal 2] Stylometric Heuristics
  │     └─ Computes sentence length variance + type-token ratio
  │        Returns: stylo_score (0.0–1.0, higher = more likely AI)
  │
  ├─► [Confidence Scoring]
  │     └─ combined = (0.7 × llm_score) + (0.3 × stylo_score)
  │        Maps to: likely_ai | uncertain | likely_human
  │
  ├─► [Transparency Label Generator]
  │     └─ Selects label variant based on combined score
  │
  ├─► [Audit Log]
  │     └─ Writes structured JSON entry to audit_log.json
  │
  └─► JSON Response
        {content_id, attribution, confidence, llm_score,
         stylo_score, label, status}
```

### Appeal Flow

```
POST /appeal  {content_id, creator_reasoning}
  │
  ├─► Lookup original entry in audit_log.json by content_id
  │
  ├─► Update entry status: "classified" → "under_review"
  │
  ├─► Append appeal record to audit_log.json
  │     └─ Includes: content_id, creator_reasoning, timestamp
  │
  └─► JSON Response
        {content_id, status: "under_review", message}
```

---

## Detection Signals

### Signal 1: LLM Classification (Groq — llama-3.3-70b-versatile)

**What it measures:** Semantic and stylistic coherence holistically. The model evaluates whether the text reads as human-authored based on naturalness of expression, idiomatic variation, tonal inconsistencies, and whether it "sounds" like a person wrote it in a specific context.

**Output:** A score from 0.0 to 1.0 (1.0 = high confidence AI, 0.0 = high confidence human). Extracted from the model's structured JSON response.

**Why it's useful:** Captures meaning-level patterns that statistical signals miss — such as overly hedged phrasing, suspiciously balanced argumentation, or unnaturally smooth transitions.

**What it misses:**
- Short texts (< 50 words) give the model too little signal to work with
- Very formal human writing (academic papers, legal documents) can score as AI-like
- The model has no memory of what "typical" AI output looks like for niche genres (e.g., dialect poetry)
- Prompt injection: a user could craft text specifically designed to fool the LLM

### Signal 2: Stylometric Heuristics (Pure Python)

**What it measures:** Statistical properties of the text's structure. Specifically:

1. **Sentence Length Variance** — computes the standard deviation of sentence word-counts. AI text tends to produce sentences of similar length (low variance); human writing has more rhythmic irregularity (high variance). A variance below a threshold increases the AI score.

2. **Type-Token Ratio (TTR)** — unique words divided by total words. AI text reuses vocabulary at a higher rate (lower TTR); human writing tends to be more lexically diverse. A TTR below ~0.6 nudges the score upward.

**Output:** A score from 0.0 to 1.0, computed as a weighted combination of the two normalized sub-metrics.

**Why it's useful:** Purely objective — computed from the text itself with no API calls. Captures structural uniformity that LLMs sometimes miss, especially in polished AI output.

**What it misses:**
- Short texts (< 3 sentences) produce unreliable variance estimates
- Poetry and experimental writing naturally have low variance and may be incorrectly flagged
- A writer who happens to write in a consistent, formal style (common for non-native English speakers) may score as AI-like
- Deliberately varied AI output (with post-processing to introduce length variation) would evade this signal

---

## Signal Combination & Confidence Scoring

Both signals produce a score between 0.0 and 1.0, where higher = more likely AI.

**Weighting:**
```
confidence = (0.7 × llm_score) + (0.3 × stylo_score)
```

The LLM score is weighted higher because it captures semantic meaning, which is harder to fake. Stylometrics acts as a structural corroboration — it can push borderline cases in one direction but rarely overrides a strong LLM result.

**Thresholds:**

| Score Range   | Attribution    | Label Variant       |
|---------------|----------------|---------------------|
| 0.75 – 1.00   | likely_ai      | High-confidence AI  |
| 0.40 – 0.74   | uncertain      | Uncertain           |
| 0.00 – 0.39   | likely_human   | High-confidence Human |

**What 0.6 means:** The system has a moderate lean toward AI but is genuinely uncertain. Both signals may be split, or the LLM returned a mid-range score. This content should receive the uncertain label — the system does not force a binary call here.

**Asymmetric design note:** The threshold for `likely_human` is intentionally tight (≤ 0.39). A false positive (labeling a human's work as AI) is more harmful than a false negative on a writing platform. The uncertain band is wide (0.40–0.74) to reflect this — when in doubt, we do not accuse.

---

## Transparency Label

Three label variants, written exactly as they will appear in the API response:

**High-confidence AI** (`confidence ≥ 0.75`):
> "This content was likely generated by an AI tool. Our system analyzed the writing style and structure and found strong indicators of AI authorship. If you created this yourself, you can submit an appeal."

**Uncertain** (`0.40 ≤ confidence < 0.75`):
> "Our system wasn't able to make a confident determination about this content. It shows some characteristics of AI-generated text, but also signs of human authorship. No action has been taken. If you have concerns, you can submit an appeal."

**High-confidence Human** (`confidence < 0.40`):
> "This content appears to have been written by a human. Our system found strong indicators of human authorship in the writing style and structure."

**Design rationale:** Labels avoid all technical jargon. No scores are shown to end users — only plain-language verdicts. The AI label explicitly offers an appeal path, because false positives are the most harmful outcome.

---

## Appeals Workflow

**Who can appeal:** Any creator who submitted content (identified by `creator_id`).

**What they provide:**
- `content_id` — the ID returned when they submitted
- `creator_reasoning` — a free-text explanation (e.g., "I wrote this myself; I am a non-native English speaker")

**What the system does:**
1. Looks up the original audit log entry for that `content_id`
2. Updates the entry's `status` field from `"classified"` to `"under_review"`
3. Appends a separate appeal record to the audit log containing: `content_id`, `creator_reasoning`, `timestamp`, `type: "appeal"`
4. Returns a confirmation JSON response

**What a human reviewer sees (appeal queue):**
- The original classification entry (attribution, confidence, both signal scores)
- The creator's reasoning
- Timestamp of the original decision and the appeal

**Automated re-classification:** Not implemented. Appeals require human review.

---

## Anticipated Edge Cases

**1. Formal human writing scored as AI**
An academic essay, legal brief, or formal report written by a human may have low sentence length variance and lower vocabulary diversity — both traits that push the stylometric score upward. The LLM may also find it unusually "polished." A human scholar writing in their second language is especially at risk. *Mitigation:* The uncertain band is wide; this case should land in uncertain rather than likely_ai unless both signals are very strong.

**2. Short texts (< 50 words)**
A haiku, a two-sentence bio, or a short social post gives both signals very little to work with. Sentence length variance is statistically meaningless with 1–2 sentences; the LLM cannot identify stylistic patterns. *Mitigation:* For texts under 50 words, the system still returns a result but the label will note limited confidence. Scores from short texts are less reliable.

**3. Lightly edited AI output**
A user who generates text with an AI tool and then edits it significantly may produce something that scores mid-range on both signals. This is genuinely ambiguous — edited AI output is a known hard case for detection. *Mitigation:* The system acknowledges uncertainty honestly; this case should land in the uncertain band.

**4. Repetitive or stylistically constrained human writing**
A poem that intentionally repeats phrases and uses simple vocabulary (common in spoken word or minimalist poetry) may trigger the heuristic signal. *Mitigation:* The LLM signal should recognize intentional constraint as a stylistic choice; the 70/30 weighting means the LLM can override a high stylometric score.

---

## API Endpoints

| Method | Endpoint   | Input                              | Output                                                  |
|--------|------------|------------------------------------|---------------------------------------------------------|
| POST   | /submit    | `{text, creator_id}`               | `{content_id, attribution, confidence, llm_score, stylo_score, label, status}` |
| POST   | /appeal    | `{content_id, creator_reasoning}`  | `{content_id, status, message}`                         |
| GET    | /log       | —                                  | `{entries: [...]}`                                      |

---

## Stretch Features

### Stretch 1: Ensemble Detection (3rd Signal — Punctuation & Complexity Score)

**What it measures:** Two additional structural properties:
1. **Punctuation density** — punctuation marks per word. Human writing tends to use more varied punctuation (em-dashes, ellipses, exclamation marks mid-sentence). AI text is punctuated more uniformly and conservatively.
2. **Average sentence complexity** — average number of clauses per sentence (approximated by counting conjunctions and commas per sentence). AI text tends toward longer, multi-clause sentences with consistent complexity; human writing varies more.

**Output:** A `punct_score` from 0.0–1.0 (higher = more likely AI).

**Updated weighting with 3 signals:**
```
confidence = (0.60 × llm_score) + (0.25 × stylo_score) + (0.15 × punct_score)
```
LLM still dominates. Stylometric score remains the second strongest signal. Punctuation/complexity acts as a tiebreaker for borderline cases. Total weights sum to 1.0.

**Why this signal adds value:** It's independent from both TTR/variance (which measure vocabulary and rhythm) and the LLM (which measures semantic coherence). Punctuation patterns are a genuine stylistic fingerprint that neither of the other two signals captures.

**What it misses:** Non-native English speakers may use punctuation more sparingly. Minimalist prose styles (Hemingway-like) use few punctuation marks by design and may score as AI-like.

---

### Stretch 2: Analytics Dashboard

**What it shows:** A browser-accessible HTML page served by Flask at `GET /dashboard` that reads the live audit log and displays:
1. **Detection patterns** — breakdown of total submissions by attribution category (likely_ai / uncertain / likely_human), shown as counts and percentages
2. **Appeal rate** — number of appeals as a percentage of total submissions
3. **Average confidence by category** — mean confidence score for each attribution bucket (shows whether the system is making decisive calls or hovering near thresholds)

**Implementation:** Single-file HTML page returned by a Flask route. Reads `audit_log.json` on each request and computes stats server-side before rendering. No JavaScript framework needed — plain HTML table and inline stats.

**Why these metrics:** Detection patterns reveal whether the system is biased toward one label. Appeal rate is a proxy for false positive rate (creators only appeal when they believe they were wrongly flagged). Average confidence by category shows whether scores are well-calibrated or clustered near thresholds.

---

## AI Tool Plan

### Milestone 3: Submission Endpoint + Signal 1

**Spec sections to provide:** Detection Signals (Signal 1 description + output format) + Architecture diagram (submission flow)

**What to ask AI to generate:**
- Flask app skeleton with `POST /submit` route stub returning hardcoded JSON
- `classify_with_llm(text)` function that calls Groq and returns a `llm_score` between 0.0–1.0

**How to verify:**
- Call the function directly with 2 inputs (clearly AI text, clearly human text) and print the raw score
- Confirm the score is a float between 0 and 1 before wiring into the endpoint

### Milestone 4: Signal 2 + Confidence Scoring

**Spec sections to provide:** Detection Signals (Signal 2 description) + Confidence Scoring section (weighting formula and thresholds) + Architecture diagram

**What to ask AI to generate:**
- `compute_stylometric_score(text)` computing sentence length variance and TTR, returning a 0–1 score
- `compute_confidence(llm_score, stylo_score)` applying the 70/30 weighted formula

**How to verify:**
- Run all 4 test inputs (clearly AI, clearly human, two borderline cases) through both signals separately
- Confirm clearly-AI text scores noticeably higher than clearly-human text
- Check that borderline cases land in the 0.40–0.74 range

### Milestone 5: Production Layer

**Spec sections to provide:** Transparency Label section (all 3 variants + thresholds) + Appeals Workflow section + Architecture diagram (both flows)

**What to ask AI to generate:**
- `generate_label(confidence)` returning the correct label text for each score range
- `POST /appeal` endpoint that updates status and appends to audit log

**How to verify:**
- Submit inputs designed to hit all 3 score ranges and confirm each label variant appears
- Submit an appeal, then call `GET /log` and confirm the entry shows `"status": "under_review"` and `creator_reasoning` is populated
