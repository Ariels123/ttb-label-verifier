# TTB Label Verifier

A standalone tool that helps a TTB compliance agent check an alcohol-beverage **label image**
against its **application data** — brand, class/type, alcohol content, net contents,
bottler/producer, country of origin, and the mandatory **Government Warning** — and returns a
big, obvious **PASS / FAIL** per field in well under five seconds. It also **reads the label and
shows what it found, with a per-field confidence level**, and has a **batch mode** for hundreds
of labels at once. All OCR runs **locally** — no cloud calls — so it works behind a locked-down
federal firewall.

**Live demo:** http://178.156.185.31:8080 · **Run locally:** [below](#running).

> **Reviewer quick start (no setup):** open the live demo → click a built-in example such as
> **"✗ wrong ABV"** to see a FAIL, or **"✓ correct"** to see an all-PASS → then drag in one of the
> real-world label photos (the `wm*.jpg` files in [`sample_images/`](sample_images)) to try a genuine photo.

---

## What it does

There are two things going on, on one screen:

1. **Verification (the core task).** The agent enters the application fields (what the COLA
   application claims), uploads the label, and the tool confirms the label matches — with a
   clear per-field PASS / FAIL and a strict check on the Government Warning.
2. **Reading the label (extraction).** The tool also **reads each field off the label**
   ("Detected on label") and shows it next to the application value, each with a **read-confidence**
   pill. So even with no application data entered, you see what the label says and how confident the
   OCR was.

Keeping "Expected (application)" and "Detected (label)" as two separate columns is deliberate:
they're two independent sources, so comparing them is real verification — never the circular trap
of checking the label against itself.

## How a result reads

Each field row shows a **status**, the **expected** (application) value, the **detected** (label)
value + confidence, and a note. **Only FAIL is a real problem** — CHECK BY EYE means the OCR
couldn't read that field (verify it manually), and DETECTED / NOT ENTERED are informational, not
failures:

| Status | Meaning |
|---|---|
| ✅ **PASS** | the entered application value matches the label |
| ❌ **FAIL** | a genuine mismatch (e.g. label says 40%, application says 45%) |
| 👁 **CHECK BY EYE** | the OCR couldn't read that field — verify it manually (not asserted as a failure) |
| 👁 **DETECTED** | read from the label; no application value was entered to verify against |
| – **NOT ENTERED** | left blank and nothing readable on the label |

The **Government Warning** is strict (Jenny's rule): exact 27 CFR 16.21 wording **and** an
ALL-CAPS `GOVERNMENT WARNING:` heading. A title-case heading, altered wording, or a flipped
"should **not** drink" all fail. A non-US warning (e.g. the UK "drink responsibly" statement) is
reported as **"Non-US warning only"** — present, but it does not satisfy the US requirement.

## Confidence & honesty

Per-field **read confidence** comes from Tesseract's word-level confidence (TSV) on the server, and
Tesseract.js word confidence in the browser. Two design choices follow from it:

- A field the OCR **couldn't read** is flagged **CHECK BY EYE**, not a confident FAIL — because a
  blank read is ambiguous (unreadable vs. truly absent), and a compliance tool shouldn't assert a
  failure it isn't sure of (maps to Jenny's "if we can't read it, we ask for a better image").
- **Extraction is gated on confidence and plausibility** — on a poor read it stays *silent* rather
  than show OCR garbage, and it ignores implausible values (a "100%" that's grape content, a "70 L"
  bottle). Showing a confidently-wrong brand is worse than showing nothing.

## Architecture

```
image ─► ocr.py (local: PIL → Tesseract + OpenCV) ─► text + per-word confidence
                                                        │
                                                        ▼
                          verifier.py (match + extract + confidence) ─► per-field result
                                                        ▲
                                       application fields (brand, ABV, …)
```

- **`code/verifier.py`** — framework-free matching + extraction engine. Forgiving on identity fields
  (case/punctuation/OCR-slip tolerant), numeric on ABV (handles proof, no 3-digit truncation),
  volume-normalized on net contents (750 mL = 75 cL = 0.75 L), strict on the Government Warning.
  Pure logic, unit-tested.
- **`code/ocr.py`** — local OCR. Opens each upload through PIL (decompression-bomb guard + EXIF
  orientation + downscale), then **unions complementary Tesseract passes** (PSM 3 body + PSM 11
  title) so stylized titles aren't dropped, capturing per-word confidence via TSV. A poor or
  low-confidence read escalates to an OpenCV preprocessing tier; the whole chain is time-bounded
  per call so one bad image can't blow the SLA. An optional PaddleOCR deep tier is wired but **off
  by default** (it OOMs a small box) — the bigger-box accuracy upgrade.
- **`code/app.py`** — Flask app + the single-file UI. Endpoints: `/verify` (image), `/batch`, and
  `/verify_text` (browser-extracted text). A global semaphore caps total concurrent Tesseract so
  simultaneous batches can't saturate the box; uploads and the JSON path are size-capped.

## Repository layout

```
code/                    the application code
  app.py                 Flask server + the single-page UI (HTML/CSS/JS inline)
  verifier.py            matching + extraction engine (framework-free, unit-tested)
  ocr.py                 local OCR ladder (PIL → Tesseract + OpenCV)
  gen_and_test.py        dev helper: regenerate the demo labels + run the OCR/verify self-test
  make_samples.py        dev helper: regenerate the three base demo labels
tests/                   unit tests for the compliance-critical rules
sample_images/           9 synthetic demo labels (the "Try an example" set) + 19 real-world test photos
requirements.txt         Python dependencies
Dockerfile               container image (Tesseract + OpenCV baked in)
```

## Tests

```bash
python3 -m pytest tests/       # unit tests — the compliance-critical matching rules
python3 code/gen_and_test.py   # renders the demo labels + runs them end-to-end through OCR + verify
```

The unit tests lock in the rules that must not regress: the strict Government Warning (exact wording
+ ALL-CAPS heading, line-wrap tolerant), numeric ABV (proof handling, no 3-digit truncation),
volume-normalized net contents, and the confidence/plausibility gating that keeps extraction silent
on a poor read. `gen_and_test.py` is an end-to-end check: it asserts the expected PASS/FAIL verdict
for each synthetic label.

## Browser OCR — the default, and the most powerful engine (optional, advisory)

OCR is the heavy part, so by default it runs **in the user's own browser, on their own hardware**
— which keeps the shared server light and scales for free. Only the extracted **text** is sent to
the server; the image never leaves the user's machine. The in-browser engine is tiered, strongest
first:

The user **chooses the engine** from a labelled selector (each option carries a one-line
description), strongest first:

1. **PaddleOCR (PP-OCRv5)** — the deep model via **ONNX Runtime Web**, GPU-accelerated with
   **WebGPU** (automatic WASM fallback). The most accurate engine in the whole system — markedly
   better than Tesseract on real label photos — running entirely client-side; models (~a few MB)
   download once from a CDN and are browser-cached. **⚠ Foreign-software notice:** PaddleOCR is
   developed by **Baidu (China)**. Selecting it runs Baidu's OCR models in the user's browser (the
   image still never leaves the device, but the *software* originates from China), so the UI shows
   an explicit warning to select it — important in a government context. If the deep model can't
   read an image it falls back to the server.
2. **Tesseract** (Tesseract.js) — fully **open-source** (Apache 2.0); lighter, less accurate on
   photos, and loads **no Baidu code**. The safe choice when policy disallows foreign-developed AI.
3. **Server** — uploads the image and runs the light server-side Tesseract; no in-browser
   engine at all.

Browser OCR is **advisory**: client-supplied text is untrusted, so the server remains the source of
truth for any authoritative decision (and the `/verify_text` inputs are hard-capped + sanitized
against abuse). The server's own OCR is deliberately kept lightweight (Tesseract, no heavy models)
because the heavy lifting now happens on the client. *(Default engine is configurable; for a strict
federal posture, default to Tesseract or Server so foreign-developed software is opt-in.)*

## Batch mode

Upload many labels at once. By default it runs a **Government Warning sweep** — confirms the one
universally-required element on every label, no per-label data needed (ideal for a 200–300 importer
dump). Optionally add a shared application template for same-product batches. Results table has a
**"show only failures"** filter, **expandable rows** (click to see the label image + full field
detail), a sticky header, and **CSV export**.

## Audit logging

Every request is logged so you can see **who** accessed the tool, **when**, and **what result**
they got. Because the tool has no login, "who" is the client **IP address + user-agent** (not a
person's identity — true per-user identity would require adding authentication). Each verification
also records the entered application values and the values detected on the label (full-audit mode).

- **Where:** JSON lines in a rotating file — `TTB_AUDIT_LOG` (default: `ttb-audit.log` under
  `TMPDIR`, which in the deployed setup is the mounted volume, so it survives container restarts;
  rotated at 10 MB × 5 backups). A concise one-line summary (no sensitive field values) also goes
  to the container log.
- **Read it (deployed):** `docker logs ttb-verifier` for the quick who/when/what trail, or read the
  JSON file on the host volume for the full per-field detail.
- **No web endpoint exposes the log** — it is readable only on the server. Health-check and favicon
  requests are excluded as noise.

## Running

```bash
brew install tesseract            # macOS  (Linux: apt-get install tesseract-ocr)
pip install -r requirements.txt
python code/app.py                # http://localhost:5050
python3 -m pytest tests/          # run the unit tests
```

Docker (Tesseract + OpenCV baked in; no outbound calls at runtime):

```bash
docker build -t ttb-verifier .
docker run -p 8080:7860 -m 1200m -e TMPDIR=/data -v /some/tmp:/data ttb-verifier
```

The deployed instance runs this image with a memory cap and a volume-backed temp dir.

## How decisions map to the brief

| Stakeholder signal | Decision |
|---|---|
| Marcus: *firewall blocks outbound ML endpoints* | **All processing local** — Tesseract on the box, no cloud API. |
| Sarah: *~5 seconds or nobody uses it* | Speed-first OCR + large-image downscale + per-call time budget → ~1–5 s (measured 0/19 real photos over 5 s). |
| Sarah: *my 73-year-old mother could use it* | One screen, giant PASS/FAIL, large type, high contrast, the verdict auto-scrolls into view, status as text + icon (not color alone). |
| Dave: *"STONE'S THROW" = "Stone's Throw"* | Forgiving identity matching. |
| Jenny: *warning must be exact, ALL-CAPS* | Strict Government Warning check (line-wrap tolerant; negation-flip and title-case fail). |
| Sarah/Janet: *200–300-label importer dumps* | Batch mode + warning sweep + filter + CSV. |
| Jenny: *weird angles / glare* | EXIF auto-rotate + an OpenCV preprocessing tier, within budget; genuinely unreadable photos return a "couldn't read" state. |

## Assumptions / scope

- Standalone proof-of-concept; **no COLA integration** (out of scope per Marcus). Application
  fields are entered alongside the image rather than looked up.
- No persistence of label images or PII — uploads are processed in a temp file and deleted; in the
  browser-OCR / extraction views the image never leaves the user's machine.
- The statutory warning is the standard 27 CFR 16.21 health-warning statement.

## Known limitations / trade-offs

- **Physical distortion still reads partially.** The browser now runs the **PP-OCRv5 deep model**
  by default (the most accurate engine in the system), so ordinary ornate/low-contrast labels read
  well. What stays genuinely hard is *physical* distortion — text wrapped around a **curved bottle**,
  shot at a steep **angle**, or **occluded** (a finger over the label) — which no flat single-pass
  OCR reads without dewarping. On those the tool stays silent on extraction and flags "verify by
  eye" rather than guessing. Next steps: perspective/cylinder dewarping, or height-based brand
  detection.
- **"Bold" heading isn't verified** — font weight isn't reliably recoverable from OCR text, so only
  wording + caps are checked.
- **Forgiving identity matching** can over-match very short, similar strings (intentional for
  "STONE'S THROW"–style tolerance).
- **Browser OCR is advisory and CDN-loaded.** The deep engine (ONNX Runtime Web + PP-OCRv5 weights)
  and the Tesseract.js fallback are fetched from public CDNs; for a strictly air-gapped deployment
  they should be self-hosted / vendored. The server remains authoritative for any decision.
- **Batch** applies one optional template to all labels (or a warning-only sweep); per-label
  application data (a CSV of different products) is a documented next step.
