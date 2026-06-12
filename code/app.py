"""TTB Label Verifier — the web layer (Flask).

WHAT THIS FILE IS
-----------------
The front door. A deliberately THIN layer that wires HTTP requests to the two engines —
ocr.py (image -> text) and verifier.py (text + application fields -> per-field verdict) —
and serves the entire UI. There is no database and no framework beyond Flask: state lives
in the request, results are computed and returned, temp files are deleted. Keeping the app
thin is intentional, so the compliance logic stays in the unit-tested verifier.

THE THREE THINGS IT SERVES
--------------------------
  1. The UI — one self-contained HTML page (the `PAGE` string near the bottom: inline
     CSS + vanilla JS, no build step, no CDN framework). Two screens: Single Label and Batch.
  2. A JSON API:
       POST /verify       multipart image + fields -> full per-field result (server OCR)
       POST /verify_text  JSON of browser-extracted text + fields -> result (ADVISORY; the
                          image never leaves the user's machine — see the handler's docstring)
       POST /batch        many images + one optional field template -> list of results
       GET  /health       liveness probe for the deploy
       GET  /sample_images/<f> serves the built-in demo labels for the "Try an example" links
  3. Static-ish bits: a favicon drawn inline as SVG.

DESIGN BARS (from the brief's stakeholders)
-------------------------------------------
  * SIMPLE & HIGH-CONTRAST — Sarah's "my 73-year-old mother could use it": big PASS/FAIL,
    large type, one screen, status conveyed as text + icon, not color alone.
  * FAST — local OCR only; end-to-end timing is measured and shown.
  * SELF-CONTAINED — no cloud calls at runtime (TTB's outbound firewall).

SECURITY POSTURE (this file owns the untrusted-input boundary)
--------------------------------------------------------------
Everything crossing the wire is treated as hostile: request size is capped
(MAX_CONTENT_LENGTH); the /verify_text JSON path is independently length-capped; batch
size-checks every file BEFORE writing it to disk and enforces a total-bytes budget; temp
file suffixes come from an extension allow-list, never from the untrusted filename; and a
global semaphore (`_OCR_SEM`) bounds total concurrent Tesseract across ALL requests so
OCR — which is CPU-bound — can't oversubscribe the cores and freeze the box.

Run:  python app.py        ->  http://localhost:5050
"""
from __future__ import annotations

import ipaddress
import json
import logging
import logging.handlers
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from flask import Flask, Response, g, jsonify, request, send_from_directory

import ocr
import verifier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # cap request size; batch also enforces per-file + total budgets
FIELD_KEYS = ("brand_name", "class_type", "alcohol_content", "net_contents", "producer", "origin")
MAX_FILE_BYTES = 15 * 1024 * 1024     # per image
MAX_BATCH_BYTES = 300 * 1024 * 1024   # total across a batch
MAX_TEXT_LEN = 20_000                 # a real label's text is < 2 KB; cap the client-OCR JSON path

# Bound TOTAL concurrent Tesseract across ALL requests (not just within one batch). OCR is
# CPU-bound, so without this two simultaneous batches would oversubscribe the cores and freeze
# the box. Every extract_text() call goes through this semaphore.
_OCR_SEM = threading.BoundedSemaphore(max(1, os.cpu_count() or 2))

_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic"}


# ============================================================================================
# AUDIT / ACCESS LOGGING
# --------------------------------------------------------------------------------------------
# Records, for every request: WHO (client IP + user-agent — the tool has no login, so this is
# the best available identity, NOT a person), WHEN (UTC timestamp), and WHAT RESULT they got.
# This is FULL-AUDIT mode: on the verification endpoints the record also includes the entered
# application values and the values detected on the label (i.e. potentially sensitive content),
# by deliberate choice. Two sinks:
#   * a rotating JSON-lines FILE (the system of record) — full detail, persists across restarts
#     because it defaults under TMPDIR, which in production is the mounted volume. Rotated at
#     10 MB x 5 so it can't fill the disk. Override the path with TTB_AUDIT_LOG.
#   * the CONTAINER LOG (stderr -> `docker logs`) — a CONCISE one-line summary only (no sensitive
#     field values), for a quick who/when/what trail.
# There is intentionally NO web endpoint exposing the log; read it on the box.
AUDIT_LOG_PATH = os.environ.get("TTB_AUDIT_LOG") or os.path.join(tempfile.gettempdir(), "ttb-audit.log")

_audit_file = logging.getLogger("ttb.audit.file")      # full JSON -> rotating file
_audit_file.setLevel(logging.INFO)
_audit_file.propagate = False
try:
    _afh = logging.handlers.RotatingFileHandler(AUDIT_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5)
    _afh.setFormatter(logging.Formatter("%(message)s"))
    _audit_file.addHandler(_afh)
except Exception:                                      # read-only fs etc. -> file sink simply disabled
    pass

_audit_console = logging.getLogger("ttb.audit.console")  # concise summary -> container log
_audit_console.setLevel(logging.INFO)
_audit_console.propagate = False
_ach = logging.StreamHandler()
_ach.setFormatter(logging.Formatter("AUDIT %(message)s"))
_audit_console.addHandler(_ach)

_NO_AUDIT_PATHS = {"/health", "/favicon.ico"}            # infra noise — don't log
_CTRL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _sanitize(s, n: int = 200) -> str:
    """Make an untrusted value safe to put in a log line: strip control characters (newlines,
    NULs, carriage returns) so a crafted filename/header can't FORGE or split audit entries,
    and length-cap it. Audit-log integrity depends on this."""
    return _CTRL_CHARS.sub("?", str(s if s is not None else ""))[:n]


def _client_ip(req) -> str:
    """The PEER-CONFIRMED client IP (req.remote_addr) — the authoritative 'who' for the audit log.

    We deliberately do NOT trust the X-Forwarded-For header here: this app is exposed directly
    (no reverse proxy in front of the published port), so XFF is fully attacker-controlled and
    honoring it would let anyone SPOOF their identity in the audit trail. The raw XFF, when
    present, is recorded SEPARATELY as a non-authoritative "claimed" value (see _audit_after) so
    operators can still see spoof attempts. If a trusted proxy is ever placed in front, switch to
    werkzeug ProxyFix with a known proxy set rather than reading the header directly.
    """
    ip = _sanitize(req.remote_addr or "?", 64)
    try:
        ipaddress.ip_address(ip)                         # remote_addr should always parse; be defensive
    except ValueError:
        ip = "?"
    return ip


def _verdict(result: dict) -> str:
    """One-word outcome for the audit line."""
    rs = result.get("results", [])
    if result.get("unreadable"):
        return "unreadable"
    if any(r["status"] == "fail" for r in rs):
        return "fail"
    if any(r["status"] == "lowconf" for r in rs):
        return "needs_review"
    return "pass" if result.get("provided") else "extracted"


def _result_detail(result: dict) -> dict:
    """The 'what result' half of a record. FULL-AUDIT: includes per-field expected + detected."""
    counts = {}
    for r in result.get("results", []):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "engine": result.get("engine", "server"),
        "filename": _sanitize(result.get("filename"), 200),   # upload filename is attacker-controlled
        "verdict": _verdict(result),
        "passed": result.get("passed"),
        "provided": result.get("provided"),
        "counts": counts,
        "result_ms": result.get("total_ms"),
        "fields": [{"field": r["field"], "expected": r.get("expected"),
                    "detected": r.get("detected"), "status": r["status"]}
                   for r in result.get("results", [])],
    }


def _write_audit(rec: dict) -> None:
    """Emit one audit record: full JSON to the file, a concise summary to the container log.
    Best-effort — logging must NEVER break the actual request."""
    try:
        _audit_file.info(json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass
    try:
        summary = f"{rec.get('ip','?')} {rec.get('method','')} {rec.get('path','')} -> {rec.get('status','')}"
        if rec.get("verdict"):
            summary += f" [{rec['verdict']} {rec.get('filename') or '-'} {rec.get('result_ms','?')}ms]"
        _audit_console.info(summary)
    except Exception:
        pass


@app.before_request
def _audit_before():
    g._audit_t0 = time.time()


@app.after_request
def _audit_after(resp):
    """Single audit-logging point. Logs every request (except health/favicon), enriched with the
    full per-field result on verification endpoints (the handlers stash detail on `g`)."""
    try:
        if request.path in _NO_AUDIT_PATHS:
            return resp
        env = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "ip": _client_ip(request),                       # peer-confirmed, sanitized + validated
            "ua": _sanitize(request.headers.get("User-Agent"), 300),
            "method": _sanitize(request.method, 16),
            "path": _sanitize(request.path, 300),
            "status": resp.status_code,
            "ms": int((time.time() - getattr(g, "_audit_t0", time.time())) * 1000),
        }
        # If a (spoofable) X-Forwarded-For is present, record it as an explicitly NON-authoritative
        # claimed value — never as `ip` — so a forged header is visible but can't rewrite the "who".
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            env["ip_claimed_xff"] = _sanitize(xff, 100)
        batch = getattr(g, "_audit_batch", None)
        detail = getattr(g, "_audit_detail", None)
        if batch is not None:                          # one small line per label (atomic, greppable)
            _write_audit({**env, "batch": True, "n": len(batch)})
            for item in batch:
                _write_audit({**env, **_result_detail(item)})
        elif detail is not None:
            _write_audit({**env, **detail})
        else:                                          # page load / example fetch / error
            _write_audit(env)
    except Exception:
        pass                                           # never let logging break a response
    return resp


def _safe_suffix(filename: str) -> str:
    """Never derive a temp-file suffix from an untrusted filename; allow-list image extensions."""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in _ALLOWED_EXT else ".img"


def _read(path: str) -> dict:
    """read_label() (text + per-word confidence) under the global concurrency cap."""
    with _OCR_SEM:
        return ocr.read_label(path)


def _verify_upload(fileobj, fields: dict) -> dict:
    """OCR one uploaded image and verify it against the application `fields`.

    The single-label server path: write the upload to a temp file (suffix from the
    allow-list, never the raw filename), read it under the concurrency cap, run the
    verifier, and stamp the result with end-to-end timing, the raw OCR text, and the
    filename. The temp file is always deleted in `finally`, even on error.

    Args:
        fileobj: a Werkzeug FileStorage (has .save) — or any object with .read(), so the
                 same helper is usable from a script/test, not just a request.
        fields:  the application values to verify against, keyed by FIELD_KEYS.
    Returns:
        the verifier result dict, plus total_ms / ocr_text / filename.
    """
    filename = getattr(fileobj, 'filename', 'image.png')
    suffix = _safe_suffix(filename)

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # close the mkstemp fd immediately (it was leaking on the save path)
    try:
        if hasattr(fileobj, 'save'):
            fileobj.save(path)
        else:
            with open(path, 'wb') as f:
                f.write(fileobj.read())

        t0 = time.time()
        data = _read(path)
        result = verifier.verify(fields, data["text"], data["words"], data["mean_conf"])
        result["total_ms"] = int((time.time() - t0) * 1000)
        result["ocr_text"] = data["text"].strip()
        result["filename"] = filename
        return result
    finally:
        if os.path.exists(path):
            os.unlink(path)


@app.get("/")
def index() -> Response:
    """Serve the single-page UI (the inline `PAGE`)."""
    return Response(PAGE, mimetype="text/html")


@app.get("/health")
def health():
    """Liveness probe — the deploy/smoke-test hits this to confirm the app is up."""
    return jsonify({"ok": True})


# The sample images (the synthetic demo labels the "Try an example" links use, alongside the
# real-world test photos) live in sample_images/ at the REPO ROOT — one level up from this code/
# folder (dirname of __file__, then its parent).
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_images")


@app.get("/sample_images/<path:name>")
def example(name):
    """Serve a built-in demo label by filename (send_from_directory guards against traversal)."""
    return send_from_directory(SAMPLES_DIR, name)


@app.get("/favicon.ico")
def favicon():
    """Return the magnifier (🔍) favicon, drawn inline as SVG so there's no static asset."""
    return Response(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
        "<text y='13' font-size='13'>\U0001F50D</text></svg>",
        mimetype="image/svg+xml")


@app.post("/verify")
def verify_one():
    """Single-label server verification: multipart `image` + application fields -> result JSON."""
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "Please choose a label image."}), 400
    fields = {k: (request.form.get(k) or "").strip() for k in FIELD_KEYS}
    result = _verify_upload(f, fields)
    g._audit_detail = _result_detail(result)   # audit log captures the full per-field result
    return jsonify(result)


@app.post("/verify_text")
def verify_text():
    """Verify against OCR text the CLIENT produced in the browser (Tesseract.js). Only the
    extracted TEXT reaches us — the image never leaves the user's machine — and the
    authoritative verifier.py runs HERE. ADVISORY ONLY: client-supplied text is untrusted
    (an applicant could fabricate it), so this is an applicant self-pre-flight, never a
    reviewer-grade PASS; a final decision re-OCRs the uploaded image server-side. Inputs are
    hard-capped so this cheap endpoint can't be used to DoS the fuzzy-matching path."""
    if (request.content_length or 0) > 256 * 1024:
        return jsonify({"error": "Request too large."}), 413
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request."}), 400
    text = str(data.get("ocr_text") or "")[:MAX_TEXT_LEN]
    if not text.strip():
        return jsonify({"error": "No text was read from the image in your browser."}), 400
    fields = {k: str(data.get(k) or "")[:200].strip() for k in FIELD_KEYS}
    try:
        client_ms = max(0, int(data.get("client_ms") or 0))
    except (TypeError, ValueError):
        client_ms = 0
    # Optional per-word confidence from the browser OCR engine (Tesseract.js). Untrusted input,
    # so sanitize hard: each item must be a [word, conf] pair with a FINITE numeric conf (reject
    # bool — a Python bool is an int subclass — and NaN/inf), and clamp every conf to 0..100 so a
    # crafted payload can't produce a NaN/out-of-range confidence pill downstream.
    raw_words = data.get("words")
    words = None
    if isinstance(raw_words, list):
        words = [(str(w[0])[:60], max(0.0, min(100.0, float(w[1])))) for w in raw_words[:5000]
                 if isinstance(w, (list, tuple)) and len(w) == 2
                 and isinstance(w[1], (int, float)) and not isinstance(w[1], bool)
                 and math.isfinite(w[1])]
    try:
        mean_conf = data.get("mean_conf")
        mean_conf = max(0.0, min(100.0, float(mean_conf))) if mean_conf is not None else None
    except (TypeError, ValueError):
        mean_conf = None
    result = verifier.verify(fields, text, words, mean_conf)
    result["ocr_text"] = text.strip()
    result["total_ms"] = client_ms
    result["filename"] = str(data.get("filename") or "browser-ocr")[:200]
    result["engine"] = "browser"
    result["advisory"] = True  # client-supplied text — applicant pre-flight, not a reviewer-grade decision
    g._audit_detail = _result_detail(result)   # audit log captures the full per-field result
    return jsonify(result)


@app.post("/batch")
def verify_batch():
    """Batch verification: many `images` + one optional shared field template -> list of results.

    Two passes. PASS 1 (this thread) validates and lands files on disk: each file is
    size-checked BEFORE it is written (oversize files are recorded as skipped, never stored),
    and a running total enforces a whole-batch byte budget so a flood of uploads can't fill
    the disk. PASS 2 OCRs + verifies them on a small thread pool (capped at the core count,
    because Tesseract is CPU-bound and oversubscription froze the box in testing). Every temp
    file is removed as its image finishes, and again in a `finally` sweep.
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Please choose one or more label images."}), 400
    if len(files) > 400:
        return jsonify({"error": f"Too many files ({len(files)}). Submit at most 400 per batch."}), 413
    fields = {k: (request.form.get(k) or "").strip() for k in FIELD_KEYS}

    # Cap parallel OCR at the core count. Tesseract is CPU-bound, so oversubscribing
    # (the old 2x = up to 16 workers) saturates a small box and freezes the server
    # for the whole batch — which is exactly what happened.
    max_workers = max(1, min(os.cpu_count() or 2, 4))
    
    # We need to save files to disk first because Flask file objects 
    # aren't thread-safe for reading in parallel from the same stream.
    # Save to disk first (Flask file objects aren't thread-safe to read in parallel), but
    # SIZE-CHECK each file BEFORE writing — oversize files are skipped, never landed on disk —
    # and enforce a total-bytes budget so a batch can't exhaust the box's disk.
    temp_files = []  # (path_or_None, filename, skip_reason_or_None)
    total = 0
    for f in files:
        try:
            f.stream.seek(0, os.SEEK_END); size = f.stream.tell(); f.stream.seek(0)
        except Exception:
            size = 0
        if size > MAX_FILE_BYTES:
            temp_files.append((None, f.filename, "File too large (>15 MB) — skipped"))
            continue
        total += size
        if total > MAX_BATCH_BYTES:
            for p, _, _ in temp_files:
                if p and os.path.exists(p):
                    os.unlink(p)
            return jsonify({"error": "Batch exceeds 300 MB total. Submit fewer or smaller images."}), 413
        fd, path = tempfile.mkstemp(suffix=_safe_suffix(f.filename))
        with os.fdopen(fd, 'wb') as tmp:
            f.save(tmp)
        temp_files.append((path, f.filename, None))

    items = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def process_file(file_info):
                path, filename, skip = file_info
                if skip:
                    return {"filename": filename, "passed": False, "provided": 0, "total_ms": 0, "fails": [skip]}
                try:
                    t0 = time.time()
                    data = _read(path)
                    r = verifier.verify(fields, data["text"], data["words"], data["mean_conf"])
                    fails = [x["field"] for x in r["results"] if x["status"] == "fail"]
                    return {"filename": filename, "passed": r["passed"], "provided": r.get("provided", 0),
                            "total_ms": int((time.time() - t0) * 1000), "fails": fails,
                            "unreadable": r.get("unreadable", False), "mean_conf": r.get("mean_conf"),
                            "results": r["results"], "ocr_text": data["text"].strip()[:3000]}
                finally:
                    if os.path.exists(path):
                        os.unlink(path)

            items = list(executor.map(process_file, temp_files))
    finally:
        for path, _, _ in temp_files:
            if path and os.path.exists(path):
                os.unlink(path)

    g._audit_batch = items   # audit log writes one full per-field record per label in the batch
    return jsonify({"items": items})


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTB Label Verifier</title>
<style>
/* Styling — high-contrast, large type, generous spacing (the "73-year-old could use it"
   bar). Palette is defined once as CSS variables in :root and reused everywhere; PASS=green,
   FAIL=red, caution=amber, but status is ALWAYS also shown as text + icon, never color alone
   (accessibility / color-blind safety). No external stylesheet — all inline, no build step. */
:root{--ink:#1a2230;--mut:#5b6573;--line:#d7dde6;--bg:#f4f6f9;--card:#fff;
 --blue:#1a4f8a;--green:#1f8f4e;--greenbg:#e7f6ed;--red:#c0392b;--redbg:#fcebe9;--amber:#b7791f;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 20px 60px}
header h1{font-size:36px;margin:0 0 4px}
header p{color:var(--mut);margin:0 0 18px;font-size:19px}
.tabs{display:flex;gap:12px;margin-bottom:24px}
.tab{background:var(--card);border:2px solid var(--line);border-radius:12px;padding:12px 24px;font-size:18px;font-weight:700;cursor:pointer;color:var(--mut)}
.tab.on{border-color:var(--blue);color:var(--blue);background:#eef4fb}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:28px;box-shadow:0 2px 8px rgba(20,40,80,.08)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:32px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
h2{font-size:16px;text-transform:uppercase;letter-spacing:1px;color:var(--mut);margin:0 0 16px;font-weight:800}
label{display:block;font-weight:700;font-size:16px;margin:16px 0 6px}
input[type=text]{width:100%;font-size:19px;padding:12px 14px;border:3px solid var(--line);border-radius:10px}
input[type=text]:focus{outline:none;border-color:var(--blue)}
.drop{border:3px dashed var(--line);border-radius:16px;background:#fafbfd;min-height:280px;display:flex;flex-direction:column;cursor:pointer;padding:18px;color:var(--mut);position:relative}
.drop:hover,.drop.over{border-color:var(--blue);background:#eef4fb}
.dropmain{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:8px}
.drop img{max-width:100%;max-height:230px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,.1);user-select:none;-webkit-user-drag:none;-webkit-user-select:none}
.drop .big{font-size:60px;line-height:1}
.dropcta{margin-top:12px;background:var(--blue);color:#fff;font-weight:800;font-size:18px;padding:13px;border-radius:12px;text-align:center}
.drop:hover .dropcta,.drop.over .dropcta{filter:brightness(1.12)}
.imgtools{display:none;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
.imgtools.on{display:flex}
.imgtool{font-size:14px;font-weight:700;padding:7px 13px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;color:var(--ink)}
.imgtool:hover{background:var(--bg)}
.imgtool.active{background:var(--blue);color:#fff;border-color:var(--blue)}
.crophint{font-size:13px;color:var(--mut)}
.rotwrap{display:flex;align-items:center;gap:7px;font-size:14px;font-weight:700;color:var(--ink)}
.rotwrap input[type=range]{width:120px;cursor:pointer}
.rotwrap #rotVal{font-variant-numeric:tabular-nums;color:var(--mut);font-weight:400;min-width:34px}
.cropsel{position:absolute;border:2px dashed var(--blue);background:rgba(26,79,138,.18);pointer-events:none;display:none;z-index:5}
.go{margin-top:24px;width:100%;background:var(--blue);color:#fff;border:none;border-radius:14px;padding:20px;font-size:22px;font-weight:800;cursor:pointer;transition:transform 0.1s}
.go:active{transform:scale(0.98)}
.go:hover{filter:brightness(1.1)} .go:disabled{opacity:.5;cursor:default}
.result{margin-top:32px}
.banner{display:flex;align-items:center;gap:18px;border-radius:14px;padding:22px 28px;font-size:28px;font-weight:900}
.banner.pass{background:var(--greenbg);color:var(--green)} .banner.fail{background:var(--redbg);color:var(--red)}
.banner .dot{font-size:42px}
.banner .bmsg{display:flex;flex-direction:column;gap:2px;min-width:0}
.banner .bigword{font-size:30px;font-weight:900;letter-spacing:1px;line-height:1}
.banner .subline{font-size:16px;font-weight:600;opacity:.92}
.timing{margin-left:auto;font-size:17px;font-weight:700;color:var(--mut)}
table{width:100%;border-collapse:separate;border-spacing:0;margin-top:20px;font-size:18px}
th{text-align:left;color:var(--mut);font-size:14px;text-transform:uppercase;letter-spacing:1px;padding:12px 15px;border-bottom:3px solid var(--line)}
td{padding:16px 15px;border-bottom:1px solid var(--line);vertical-align:top}
.s{font-weight:900;font-size:15px;white-space:nowrap}.s.pass{color:var(--green)}.s.fail{color:var(--red)}.s.skip{color:var(--mut)}.s.lowconf{color:var(--amber)}.s.detected{color:var(--blue)}
.conf{display:inline-block;padding:3px 10px;border-radius:20px;font-weight:800;font-size:13px;white-space:nowrap}
.conf.hi{background:var(--greenbg);color:var(--green)}
.conf.md{background:#fff7e6;color:var(--amber)}
.conf.lo{background:var(--redbg);color:var(--red)}
.conf.na{background:#eef1f5;color:var(--mut)}
.badge{display:inline-block;padding:4px 10px;border-radius:6px;font-weight:800;font-size:14px;text-transform:uppercase}
.badge.pass{background:var(--green);color:#fff} .badge.fail{background:var(--red);color:#fff}
.fieldname{font-weight:800;color:var(--ink)}
.note{color:var(--mut);font-size:16px;line-height:1.4}
details{margin-top:24px;background:#f8fafc;padding:16px;border-radius:12px;border:1px solid var(--line)}
summary{cursor:pointer;color:var(--blue);font-weight:800;font-size:18px}
pre{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px;white-space:pre-wrap;font-size:15px;color:#2c3a4d;max-height:300px;overflow:auto;margin-top:12px}
.err{background:var(--redbg);color:var(--red);border-radius:12px;padding:16px 20px;margin-top:20px;font-weight:800;border:2px solid var(--red)}
.hint{color:var(--mut);font-size:16px;margin-top:10px}
input::placeholder{font-style:italic;color:#aab2bd;opacity:1}
.banner.warn{background:#fff7e6;color:#b7791f}
.badge.warn{background:#b7791f;color:#fff}
.examples{margin-top:14px;font-size:15px;color:var(--mut)}
.examples a{display:inline-block;margin:4px 12px 0 0;color:var(--blue);font-weight:700;text-decoration:none}
.examples a:hover{text-decoration:underline}
.ocrtoggle{display:flex;align-items:center;gap:10px;margin-top:18px;font-weight:700;font-size:16px;cursor:pointer;color:var(--ink)}
.ocrtoggle input{width:22px;height:22px;cursor:pointer;flex:none}
.ocrtoggle em{font-style:normal;color:var(--mut);font-weight:400}
.privacy{margin-top:10px;color:var(--green);font-size:16px;font-weight:700}
.fallback{margin-top:18px;background:#fff7e6;color:#8a5a00;border:2px solid #e6b800;border-radius:12px;padding:14px 18px;font-size:16px;font-weight:600;line-height:1.45}
.enginepick{margin-top:16px}
.englabel{font-weight:700;font-size:15px;color:var(--ink);margin-bottom:8px}
.engopt{display:grid;grid-template-columns:22px 1fr;gap:4px 10px;align-items:start;padding:9px 0;cursor:pointer;border-top:1px solid var(--line)}
.engopt input{width:20px;height:20px;cursor:pointer;margin-top:2px;grid-row:span 2}
.engname{font-weight:700;font-size:15px;color:var(--ink)}
.engdesc{font-size:14px;color:var(--mut);line-height:1.4}
.engflag{display:inline-block;margin-left:6px;font-size:12px;font-weight:700;color:#a3201a;background:#fdecea;border:1px solid #f0b3ad;border-radius:6px;padding:1px 7px;vertical-align:1px;position:relative;cursor:help;text-decoration:underline dotted}
.engok{display:inline-block;margin-left:6px;font-size:12px;font-weight:700;color:var(--green);background:var(--greenbg);border-radius:6px;padding:1px 7px;vertical-align:1px}
.engtip{display:none;position:absolute;top:160%;left:0;z-index:60;width:330px;max-width:78vw;background:#2b1410;color:#fff;border:1px solid #e0897f;border-radius:10px;padding:11px 13px;font-size:13px;font-weight:400;line-height:1.5;text-decoration:none;box-shadow:0 8px 26px rgba(0,0,0,.28)}
.engtip:before{content:"";position:absolute;bottom:100%;left:16px;border:7px solid transparent;border-bottom-color:#2b1410}
.engflag:hover .engtip,.engflag:focus .engtip,.engflag:focus-within .engtip{display:block}
.prog{margin-top:14px;height:10px;background:#e7edf5;border-radius:6px;overflow:hidden}
.progbar{height:100%;width:0;background:var(--blue);transition:width .25s}
@keyframes indet{0%{transform:translateX(-110%)}100%{transform:translateX(260%)}}
.working{font-size:19px;color:var(--blue);font-weight:800;padding:18px 2px}
.workbar{height:8px;background:#e7edf5;border-radius:5px;overflow:hidden;margin-top:12px;max-width:340px}
.workbar>div{height:100%;width:40%;background:var(--blue);animation:indet 1.1s ease-in-out infinite}
.filter{display:inline-flex;align-items:center;gap:10px;margin:18px 0 4px;font-weight:700;font-size:16px;cursor:pointer}
.filter input{width:20px;height:20px}
#bwrap.failonly .brow.pass,#bwrap.failonly .brow.warn{display:none}
#bwrap{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:12px;margin-top:6px}
#bwrap table{margin-top:0}
#bwrap thead th{position:sticky;top:0;background:var(--card);box-shadow:inset 0 -2px 0 var(--line)}
.bhelp{color:var(--mut);font-size:15px;line-height:1.5;margin:2px 0 18px}
.bhelp b{color:var(--ink)}
.brow{cursor:pointer}.brow:hover{background:#f4f8fd}
.expcell{width:22px;text-align:center;color:var(--mut)}.exp{font-size:12px}
.bdetail{display:none}.bdetail.open{display:table-row}
.bdetail>td{background:#f6f9fc;padding:0}
.bdetailwrap{display:flex;gap:20px;padding:16px;align-items:flex-start;flex-wrap:wrap}
.bthumb{max-width:240px;max-height:240px;border-radius:10px;border:1px solid var(--line);flex:none;box-shadow:0 2px 8px rgba(20,40,80,.1)}
.bdetailbody{flex:1;min-width:280px}
.bdetailbody table{margin-top:0}
#bwrap.failonly .bdetail.pass,#bwrap.failonly .bdetail.warn{display:none}
.btnrow{display:flex;gap:12px;align-items:stretch;margin-top:24px}
.btnrow .go{flex:1;margin-top:0}
.clearbtn{background:#fff;color:var(--mut);border:2px solid var(--line);border-radius:14px;padding:0 24px;font-size:17px;font-weight:800;cursor:pointer}
.clearbtn:hover{border-color:var(--blue);color:var(--blue)}
:focus-visible{outline:3px solid #1a4f8a;outline-offset:2px}
.drop:focus-visible{border-color:var(--blue);background:#eef4fb}
</style></head>
<body><div class="wrap">
<noscript><div class="err" style="margin-bottom:20px">This tool needs JavaScript to verify labels. Please enable JavaScript in your browser settings and reload the page.</div></noscript>
<div id="capWarn" class="err" style="display:none;margin-bottom:20px"></div>
<header><h1>&#128269; TTB Label Verifier</h1>
<p>Enter the application details, upload a photo of the label, and we'll confirm the label matches — including the mandatory Government Warning. New here? Click an example below to see it in action.</p></header>
<div class="tabs"><button class="tab on" id="tSingle" onclick="show('single')">Single Label</button>
<button class="tab" id="tBatch" onclick="show('batch')">Batch Processing</button></div>

<div class="card" id="single">
 <div class="grid">
  <div>
   <h2>Application Details</h2>
   <label for="brand_name">Brand Name</label><input type="text" id="brand_name" placeholder="e.g. Old Tom Distillery">
   <label for="class_type">Class / Type</label><input type="text" id="class_type" placeholder="e.g. Kentucky Straight Bourbon Whiskey">
   <label for="alcohol_content">Alcohol Content (% ABV or Proof)</label><input type="text" id="alcohol_content" placeholder="e.g. 45">
   <label for="net_contents">Net Contents</label><input type="text" id="net_contents" placeholder="e.g. 750 mL">
   <label for="producer">Bottler / Producer</label><input type="text" id="producer" placeholder="e.g. Old Tom Distillery, Bardstown KY">
   <label for="origin">Country of Origin</label><input type="text" id="origin" placeholder="e.g. USA (imports only)">
  </div>
  <div>
   <h2>Label Image</h2>
   <div class="drop" id="drop" role="button" tabindex="0" aria-label="Upload a label image"
        onclick="dropClick()"
        onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();dropClick();}">
     <div class="dropmain"><div class="big">&#128247;</div><div class="hint">Drag a clear photo of the label here, or</div></div>
     <div class="dropcta">&#128073; Click here to start</div>
   </div>
   <input type="file" id="file" accept="image/*" style="display:none">
   <div class="imgtools" id="imgtools" title="Adjust the image before checking — helpful for angled or curved-bottle photos">
     <button type="button" class="imgtool" onclick="rotateImg(-90)" title="Rotate 90° left (sideways photo)">&#8634; 90&deg;</button>
     <button type="button" class="imgtool" onclick="rotateImg(90)" title="Rotate 90° right (sideways photo)">90&deg; &#8635;</button>
     <label class="rotwrap" title="Fine rotation — straighten a tilted label so the text is level">Straighten <input type="range" id="rotSlider" min="-45" max="45" value="0" step="1" oninput="rotLive(this.value)" onchange="rotBake(this.value)"><span id="rotVal">0&deg;</span></label>
     <button type="button" class="imgtool" id="cropBtn" onclick="toggleCrop()">&#9986;&#65039; Crop</button>
     <button type="button" class="imgtool" id="cropApply" onclick="applyCrop()" style="display:none">Apply crop</button>
     <span class="crophint" id="cropHint"></span>
   </div>
   <div class="examples">Try an example (fills the form &amp; runs it):
     <a href="#" onclick="loadExample('sample_correct.png');return false">&#10003; correct</a>
     <a href="#" onclick="loadExample('sample_bad_abv.png');return false">&#10007; wrong ABV</a>
     <a href="#" onclick="loadExample('sample_bad_warning.png');return false">&#10007; bad warning</a>
     <a href="#" onclick="loadExample('sample_wine.png');return false">&#127863; wine</a>
     <a href="#" onclick="loadExample('sample_sideways.png');return false">&#8635; sideways</a>
   </div>
   <details class="advanced" open><summary>Privacy &amp; advanced options</summary>
   <div class="enginepick" role="radiogroup" aria-label="OCR engine">
     <div class="englabel">&#9889; OCR engine <em style="font-weight:400;color:var(--mut)">— runs on your device unless noted; the image never leaves your computer for the browser engines</em></div>
     <label class="engopt"><input type="radio" name="engine" value="paddle" checked onchange="engineChanged('')">
       <span class="engname">PaddleOCR (PP-OCRv5)<span class="engflag" tabindex="0" role="note" aria-label="Foreign software warning"><span aria-hidden="true">&#9888; Baidu &middot; China</span><span class="engtip">&#9888;&#65039; <b>PaddleOCR is developed by Baidu, a company based in China.</b> Selecting it runs Baidu&rsquo;s OCR models inside your browser. Your label image stays on your device, but the OCR software itself originates from China &mdash; review your organization&rsquo;s policy on foreign-developed software before using this option in a government environment.</span></span></span>
       <span class="engdesc">Most accurate. Deep-learning OCR running on your device&rsquo;s GPU; the image stays on your computer. Falls back to the server if it can&rsquo;t read an image.</span></label>
     <label class="engopt"><input type="radio" name="engine" value="tesseract" onchange="engineChanged('')">
       <span class="engname">Tesseract<span class="engok">open-source</span></span>
       <span class="engdesc">Fully open-source (Apache 2.0). Lighter but less accurate on photos; runs on your device, image stays local.</span></label>
     <label class="engopt"><input type="radio" name="engine" value="server" onchange="engineChanged('')">
       <span class="engname">Server</span>
       <span class="engdesc">Uploads the image and runs OCR on the TTB server &mdash; no in-browser engine. Use if your policy disallows in-browser AI models.</span></label>
   </div>
   </details>
  </div>
 </div>
 <div class="btnrow">
   <button class="go" id="goBtn" onclick="verifyOne()" disabled title="Add a label image first">Verify Label Now</button>
   <button class="clearbtn" id="clearBtn" onclick="clearSingle()" title="Clear the form, image, and result">Clear</button>
 </div>
 <div id="prog" class="prog" style="display:none"><div id="progbar" class="progbar"></div></div>
 <div class="result" id="result" role="status" aria-live="polite"></div>
</div>

<div class="card" id="batch" style="display:none">
 <h2>Batch — check many labels at once</h2>
 <p class="bhelp">Upload a set of labels and we'll confirm the mandatory <b>Government Warning</b> on every one — no details needed (ideal for a 200–300 importer dump). <b>Optional:</b> if they're all the <b>same product</b>, add its application details to also check brand, ABV, net contents, etc.</p>
 <div class="drop" id="bdrop" style="min-height:200px" role="button" tabindex="0" aria-label="Add label images"
      onclick="document.getElementById('bfile').click()"
      onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();document.getElementById('bfile').click();}">
   <div class="dropmain"><div class="big">&#128230;</div><div id="bcount" class="hint">No labels added yet</div></div>
   <div class="dropcta">&#128073; Click here to add labels</div>
 </div>
 <input type="file" id="bfile" accept="image/*" multiple style="display:none">
 <details class="advanced"><summary>Optional — application details (for same-product batches)</summary>
 <div class="grid" style="margin-top:10px">
   <div>
     <label for="b_brand_name">Brand Name</label><input type="text" id="b_brand_name" placeholder="e.g. Old Tom Distillery">
     <label for="b_class_type">Class / Type</label><input type="text" id="b_class_type" placeholder="e.g. Kentucky Straight Bourbon Whiskey">
     <label for="b_alcohol_content">Alcohol Content</label><input type="text" id="b_alcohol_content" placeholder="e.g. 45">
   </div>
   <div>
     <label for="b_net_contents">Net Contents</label><input type="text" id="b_net_contents" placeholder="e.g. 750 mL">
     <label for="b_producer">Bottler / Producer</label><input type="text" id="b_producer" placeholder="e.g. Old Tom Distillery, Bardstown KY">
     <label for="b_origin">Country of Origin</label><input type="text" id="b_origin" placeholder="e.g. USA (imports only)">
   </div>
 </div>
 </details>
 <details class="advanced" open><summary>Privacy &amp; advanced options</summary>
 <div class="enginepick" role="radiogroup" aria-label="OCR engine for batch">
   <div class="englabel">&#9889; OCR engine <em style="font-weight:400;color:var(--mut)">— used for every label in the batch</em></div>
   <label class="engopt"><input type="radio" name="b_engine" value="paddle" checked onchange="engineChanged('b_')">
     <span class="engname">PaddleOCR (PP-OCRv5)<span class="engflag" tabindex="0" role="note" aria-label="Foreign software warning"><span aria-hidden="true">&#9888; Baidu &middot; China</span><span class="engtip">&#9888;&#65039; <b>PaddleOCR is developed by Baidu, a company based in China.</b> Selecting it runs Baidu&rsquo;s OCR models inside your browser. Your label image stays on your device, but the OCR software itself originates from China &mdash; review your organization&rsquo;s policy on foreign-developed software before using this option in a government environment.</span></span></span>
     <span class="engdesc">Most accurate. Deep-learning OCR on your device&rsquo;s GPU; images stay on your computer. Ideal for large batches.</span></label>
   <label class="engopt"><input type="radio" name="b_engine" value="tesseract" onchange="engineChanged('b_')">
     <span class="engname">Tesseract<span class="engok">open-source</span></span>
     <span class="engdesc">Fully open-source (Apache 2.0). Lighter, less accurate; runs on your device.</span></label>
   <label class="engopt"><input type="radio" name="b_engine" value="server" onchange="engineChanged('b_')">
     <span class="engname">Server</span>
     <span class="engdesc">Uploads each image and runs OCR on the TTB server &mdash; no in-browser engine.</span></label>
 </div>
 </details>
 <button class="go" id="bGoBtn" onclick="verifyBatch()">Start Batch Verification</button>
 <div class="result" id="bresult" role="status" aria-live="polite"></div>
</div>
</div>
<script>
/* =====================================================================================
   CLIENT-SIDE SCRIPT  (vanilla JS — no framework, no build step)
   -------------------------------------------------------------------------------------
   Responsibilities:
     - drive the two screens (Single Label / Batch)
     - optionally OCR the image IN THE BROWSER (Tesseract.js) and post only the text, so
       the image never leaves the user's machine; otherwise upload the image to the server
     - render the per-field result table (status, expected, detected + confidence)
   Server endpoints used:  /verify (image)  /verify_text (browser text)  /batch  /sample_images/<f>
   The server is always the source of truth; browser OCR is an advisory fast path that
   automatically falls back to the server if it can't read an image.
   Sections below are marked with banner comments:  ==== SECTION ====
   ===================================================================================== */
const $=id=>document.getElementById(id);
// App state: the chosen single-label File, the batch File list, and the last batch result
// (kept so "export CSV" / row expansion can re-read it without re-OCRing).
let singleFile=null, batchFiles=[], lastBatch=null;
function show(w){const s=w==='single';$('single').style.display=s?'':'none';$('batch').style.display=s?'none':'';$('tSingle').classList.toggle('on',s);$('tBatch').classList.toggle('on',!s);}  // toggle screens
function esc(x){return (x==null?'':String(x)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}  // HTML-escape every value we inject (XSS-safe)

// Capability detection — warn on old/limited browsers, and gracefully disable
// in-browser OCR when the browser lacks Web Workers (Tesseract.js needs them).
const CAP={fetch:typeof window.fetch==='function',
           file:!!(window.File&&window.FileReader&&window.Blob),
           worker:typeof window.Worker==='function',
           webgpu:!!(navigator.gpu)};
(function(){
  const missing=[];
  if(!CAP.fetch)missing.push('network requests (fetch)');
  if(!CAP.file)missing.push('local file reading (File API)');
  if(missing.length){const w=$('capWarn');if(w){w.style.display='';
    w.textContent='Your browser is missing: '+missing.join(', ')+'. Please update to a current browser (Chrome, Edge, Firefox, or Safari) to use this tool.';}}
  // In-browser engines (PaddleOCR / Tesseract.js) need Web Workers + WebAssembly. If absent,
  // disable them and force the "Server" engine.
  if(!CAP.worker || typeof WebAssembly!=='object'){
    ['engine','b_engine'].forEach(grp=>document.querySelectorAll('input[name="'+grp+'"]').forEach(r=>{
      if(r.value==='server'){r.checked=true;} else {r.disabled=true; const o=r.closest('.engopt'); if(o)o.style.opacity='.5';}
    }));
  }
  engineChanged(''); engineChanged('b_');   // reflect the initial selection (show the Baidu warning if default)
})();

/* ==== SINGLE LABEL: file selection, examples, preview ============================== */
const drop=$('drop');
// Application fields pre-filled by each "Try an example" link, so a click both loads the
// image AND populates the form, then runs a verification — a one-click end-to-end demo.
const EXAMPLE_FIELDS={
  'sample_correct.png':{brand_name:'Old Tom Distillery',class_type:'Kentucky Straight Bourbon Whiskey',alcohol_content:'45',net_contents:'750 mL'},
  'sample_bad_abv.png':{brand_name:'Old Tom Distillery',class_type:'Kentucky Straight Bourbon Whiskey',alcohol_content:'45',net_contents:'750 mL'},
  'sample_bad_warning.png':{brand_name:'Old Tom Distillery',class_type:'Kentucky Straight Bourbon Whiskey',alcohol_content:'45',net_contents:'750 mL'},
  'sample_sideways.png':{brand_name:'Old Tom Distillery',class_type:'Kentucky Straight Bourbon Whiskey',alcohol_content:'45',net_contents:'750 mL'},
  'sample_wine.png':{brand_name:'Chateau Marengo',class_type:'Napa Valley Cabernet Sauvignon',alcohol_content:'13.5',net_contents:'750 mL'}
};
$('file').addEventListener('change',e=>{singleFile=e.target.files[0];$('result').innerHTML='';preview();});
['dragover','dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.toggle('over',ev==='dragover');if(ev==='drop'){singleFile=e.dataTransfer.files[0];$('result').innerHTML='';preview();}}));
function preview(){if(!singleFile)return;if($('goBtn'))$('goBtn').disabled=false;const u=URL.createObjectURL(singleFile);drop.innerHTML='<div class="dropmain"><img draggable="false" src="'+u+'"><div class="hint">'+esc(singleFile.name)+'</div></div><div class="dropcta">&#128260; Try another image</div>';showImgTools(true);}

// ---- Interactive crop / rotate: a human assist for angled or curved-bottle photos ----------
// The edit is applied to the image BEFORE OCR (singleFile is replaced with the edited canvas),
// so the agent can straighten a sideways shot or crop to the flat, readable part of a label.
let cropMode=false,_cs=null,_cropRect=null;
function dropClick(){ if(cropMode) return; document.getElementById('file').click(); }   // don't re-open the picker mid-crop
function showImgTools(on){ const t=$('imgtools'); if(t)t.classList.toggle('on',!!on); if(!on&&cropMode)toggleCrop(); }
function fileToImg(file){ return new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=()=>rej(new Error('img'));im.src=URL.createObjectURL(file);}); }
function canvasToFile(c,name){ return new Promise(res=>c.toBlob(b=>res(new File([b],name||'edited.png',{type:(b&&b.type)||'image/png'})),'image/png',0.95)); }
async function rotateImg(deg){
  if(!singleFile||cropMode||!deg) return;
  const im=await fileToImg(singleFile), rad=deg*Math.PI/180, w=im.naturalWidth, h=im.naturalHeight;
  const aw=Math.abs(w*Math.cos(rad))+Math.abs(h*Math.sin(rad)), ah=Math.abs(w*Math.sin(rad))+Math.abs(h*Math.cos(rad));
  const c=document.createElement('canvas'); c.width=Math.round(aw); c.height=Math.round(ah);
  const x=c.getContext('2d'); x.fillStyle='#fff'; x.fillRect(0,0,c.width,c.height);   // white bg so rotated corners aren't black to the OCR
  x.translate(c.width/2,c.height/2); x.rotate(rad); x.drawImage(im,-w/2,-h/2);
  URL.revokeObjectURL(im.src);
  singleFile=await canvasToFile(c,singleFile.name); $('result').innerHTML=''; preview();
}
// "Straighten" slider: live-rotate the preview (CSS) while dragging; bake into the image on
// release, then reset the slider to 0 — each gesture nudges the current image by that angle.
function rotLive(v){ const im=_imgEl(); if(im){im.style.transformOrigin='center';im.style.transform='rotate('+v+'deg)';} $('rotVal').textContent=(v>0?'+':'')+v+'°'; }
async function rotBake(v){
  const sl=$('rotSlider'), ang=parseFloat(v)||0, im=_imgEl();
  if(im)im.style.transform=''; if(sl)sl.value=0; $('rotVal').textContent='0°';
  if(singleFile&&ang) await rotateImg(ang);
}
function toggleCrop(){
  cropMode=!cropMode;
  $('cropBtn').classList.toggle('active',cropMode);
  $('cropApply').style.display=cropMode?'':'none';
  $('cropHint').textContent=cropMode?'Drag a box over the part to keep, then “Apply crop”.':'';
  $('drop').style.cursor=cropMode?'crosshair':'';
  if(!cropMode){const s=$('drop').querySelector('.cropsel');if(s)s.style.display='none';_cropRect=null;_cs=null;}
}
function _imgEl(){ return $('drop').querySelector('img'); }
function _cropDown(e){ if(!cropMode||!_imgEl())return; e.preventDefault(); _cs={x:e.clientX,y:e.clientY}; }
function _cropMove(e){
  if(!cropMode||!_cs)return;
  const drop=$('drop'),dr=drop.getBoundingClientRect(); let s=drop.querySelector('.cropsel');
  if(!s){s=document.createElement('div');s.className='cropsel';drop.appendChild(s);}
  const x0=Math.min(_cs.x,e.clientX),y0=Math.min(_cs.y,e.clientY),x1=Math.max(_cs.x,e.clientX),y1=Math.max(_cs.y,e.clientY);
  s.style.display='block';s.style.left=(x0-dr.left)+'px';s.style.top=(y0-dr.top)+'px';s.style.width=(x1-x0)+'px';s.style.height=(y1-y0)+'px';
  _cropRect={x0,y0,x1,y1};
}
function _cropUp(){ _cs=null; }
async function applyCrop(){
  if(!cropMode||!_cropRect||!singleFile)return;
  const im=_imgEl(); if(!im)return; const ir=im.getBoundingClientRect();
  const dx0=Math.max(_cropRect.x0,ir.left),dy0=Math.max(_cropRect.y0,ir.top),dx1=Math.min(_cropRect.x1,ir.right),dy1=Math.min(_cropRect.y1,ir.bottom);
  if(dx1-dx0<8||dy1-dy0<8){$('cropHint').textContent='Selection too small — drag a larger box over the label.';return;}
  const src=await fileToImg(singleFile), sx=src.naturalWidth/ir.width, sy=src.naturalHeight/ir.height;
  const cw=(dx1-dx0)*sx, ch=(dy1-dy0)*sy;
  const c=document.createElement('canvas'); c.width=Math.round(cw); c.height=Math.round(ch);
  c.getContext('2d').drawImage(src,(dx0-ir.left)*sx,(dy0-ir.top)*sy,cw,ch,0,0,c.width,c.height);
  URL.revokeObjectURL(src.src);
  singleFile=await canvasToFile(c,singleFile.name);
  toggleCrop(); $('result').innerHTML=''; preview();
}
$('drop').addEventListener('mousedown',_cropDown);
$('drop').addEventListener('dragstart',e=>e.preventDefault());   // never let a real drag grab the image ghost (would hijack crop)
document.addEventListener('mousemove',_cropMove);
document.addEventListener('mouseup',_cropUp);
async function loadExample(name){
  const f=EXAMPLE_FIELDS[name]||{};
  ['brand_name','class_type','alcohol_content','net_contents','producer','origin'].forEach(k=>{$(k).value=f[k]||'';});
  $('result').innerHTML='<div class="hint">Loading example&hellip;</div>';
  try{
    const blob=await (await fetch('/sample_images/'+name)).blob();
    singleFile=new File([blob],name,{type:blob.type||'image/png'});
    preview(); verifyOne();
  }catch(e){$('result').innerHTML='<div class="err">Could not load example: '+esc(e.message)+'</div>';}
}

/* ==== BROWSER-SIDE OCR (beta, advisory) ===========================================
   Tesseract.js (WASM) reads the label on the USER'S OWN machine; only the extracted text
   is posted to /verify_text — the image never leaves the device. The engine is lazy-loaded
   from a CDN on first use (so the page itself stays light), and the caller falls back to
   the server if loading or reading fails. */
let tessReady=false;
function loadTesseract(){
  return new Promise((resolve,reject)=>{
    if(tessReady||window.Tesseract){tessReady=true;return resolve();}
    const s=document.createElement('script');
    s.src='https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js';
    s.onload=()=>{tessReady=true;resolve();};
    s.onerror=()=>reject(new Error('Could not load the in-browser OCR engine (check your connection).'));
    document.head.appendChild(s);
  });
}
function medianConf(words){
  const cs=words.filter(w=>/[A-Za-z]{2,}/.test(w[0])).map(w=>w[1]).sort((a,b)=>a-b);
  return cs.length?Math.round(cs[Math.floor(cs.length/2)]):0;
}
async function browserOcr(file,onProg){
  await loadTesseract();
  // Mirror the server's multi-PSM ladder: PSM 3 (auto) reads body paragraphs; PSM 11
  // (sparse) catches stylized titles a single pass misses. The PSM-3 pass also yields
  // per-word CONFIDENCE for the read-confidence display. Returns {text, words, meanConf}.
  const worker=await Tesseract.createWorker('eng',1,{logger:m=>{
    if(m.status==='recognizing text'&&onProg)onProg(Math.round((m.progress||0)*100));
  }});
  try{
    let combined='', words=[];
    for(const psm of ['3','11']){
      await worker.setParameters({tessedit_pageseg_mode:psm,user_defined_dpi:'300'});
      const {data}=await worker.recognize(file);
      if(data&&data.text)combined+=data.text+'\n';
      if(psm==='3'&&data&&data.words)words=data.words.map(w=>[w.text,w.confidence]).filter(x=>x[0]&&x[1]>=0);
    }
    return {text:combined, words:words, meanConf:medianConf(words)};
  }finally{await worker.terminate();}
}

/* ---- Deep browser OCR: PP-OCRv5 via ONNX Runtime Web (WebGPU, WASM fallback) ----------
   The MOST powerful engine we can run client-side, executed on the USER'S OWN hardware/GPU,
   so the shared server stays light. Loaded once via a dynamic ESM import from a CDN; the
   PP-OCRv5 models auto-download once and are browser-cached. Far more accurate than
   Tesseract.js on real label photos. If it can't load/run, we fall back to Tesseract.js,
   then (only if both fail) to the server. */
let paddleSvc=null, paddleLoad=null;
function loadPaddle(){
  if(paddleSvc) return Promise.resolve(paddleSvc);
  if(!paddleLoad) paddleLoad=(async()=>{
    const mod=await import('https://esm.sh/ppu-paddle-ocr/web');
    const svc=new mod.PaddleOcrService();
    await svc.initialize();                                 // downloads PP-OCRv5 models once (cached)
    paddleSvc=svc; return svc;
  })();
  return paddleLoad;
}
// Flatten one PP-OCR result into {lines:[text], words:[[word,conf]], confs:[]}.
function flattenPaddle(res){
  const lines=[],words=[],confs=[];
  for(const line of (res.lines||[])) for(const seg of (line||[])){
    const t=String(seg.text||'').trim(); if(!t) continue;
    const cf=Math.round((seg.confidence||0)*100); confs.push(cf); lines.push(t);
    for(const wd of t.split(/\s+/)) if(wd) words.push([wd,cf]);
  }
  return {lines,words,confs};
}
function pack(f){f.confs.sort((a,b)=>a-b);return {text:f.lines.join('\n'),words:f.words,meanConf:f.confs.length?f.confs[Math.floor(f.confs.length/2)]:0};}
async function paddleOcr(file,onProg){
  const svc=await loadPaddle();
  if(onProg)onProg(35);
  const img=new Image(); img.src=URL.createObjectURL(file);
  await new Promise((r,e)=>{img.onload=r;img.onerror=()=>e(new Error('decode failed'));});
  try{
    const W=img.width,H=img.height;
    const drawCanvas=(sx,sy,sw,sh,scale)=>{const c=document.createElement('canvas');c.width=Math.round(sw*scale);c.height=Math.round(sh*scale);c.getContext('2d').drawImage(img,sx,sy,sw,sh,0,0,c.width,c.height);return c;};
    const MAX=2000, dscale=Math.max(W,H)>MAX?MAX/Math.max(W,H):1;      // downscale big photos: fast + GPU-safe
    let best=flattenPaddle(await svc.recognize(drawCanvas(0,0,W,H,dscale)));   // whole image (fast path)
    if(onProg)onProg(70);
    // ESCALATE: a poor whole-image read usually means a curved/angled label whose text baselines
    // bend across the frame. Re-OCR in narrow VERTICAL STRIPS — the curve is ~flat over a small
    // horizontal span — and union the fragments. Recovers fields (ABV, net contents, …) the whole
    // image misses. Only runs on hard images, so easy labels stay fast.
    if(wordCount(best.lines.join(' '))<12){
      const N=3, overlap=0.12, seen=new Set(), merged={lines:[],words:[],confs:[]};
      best.lines.forEach(t=>{const k=t.toLowerCase().replace(/[^a-z0-9]/g,'');if(k)seen.add(k);});
      for(let i=0;i<N;i++){
        const x0=Math.max(0,(i/N-overlap))*W, x1=Math.min(1,((i+1)/N+overlap))*W, sw0=x1-x0;
        const r=flattenPaddle(await svc.recognize(drawCanvas(x0,0,sw0,H,Math.min(3,1400/sw0))));
        r.confs.forEach(c=>merged.confs.push(c)); r.words.forEach(wd=>merged.words.push(wd));
        for(const t of r.lines){const k=t.toLowerCase().replace(/[^a-z0-9]/g,'');if(k&&!seen.has(k)){seen.add(k);merged.lines.push(t);}}
        if(onProg)onProg(70+Math.round((i+1)/N*28));
      }
      if(wordCount(merged.lines.join(' '))>wordCount(best.lines.join(' '))) best=merged;  // use strips if they recovered more
    }
    if(onProg)onProg(100);
    return pack(best);
  } finally { URL.revokeObjectURL(img.src); }
}
// Best in-browser OCR: try the deep PP-OCR model first; if it can't load/run or reads too
// little, fall back to Tesseract.js. Both stay entirely on the user's machine.
async function bestBrowserOcr(file,onProg){
  try{
    const r=await paddleOcr(file,onProg);
    if(wordCount(r.text)>=4){ r.engine='paddle'; return r; }
  }catch(e){ /* deep engine unavailable (old browser / network) -> Tesseract.js */ }
  if(onProg)onProg(0);
  const r=await browserOcr(file,onProg); r.engine='tesseract'; return r;
}

// ---- Engine selector ----------------------------------------------------------------------
// Read the chosen engine from the radio group ('' = single panel, 'b_' = batch panel).
function selectedEngine(p){ const r=document.querySelector('input[name="'+(p?'b_engine':'engine')+'"]:checked'); return (r&&r.value)||'server'; }
// Show/hide the foreign-software (Baidu) warning when PaddleOCR is the selected engine.
function engineChanged(p){
  const w=$(p+'engWarn'); if(!w) return;
  if(selectedEngine(p)==='paddle'){
    w.style.display='block';
    w.innerHTML='&#9888;&#65039; <b>PaddleOCR is developed by Baidu, a company based in China.</b> Selecting it runs Baidu&rsquo;s OCR models inside your browser. Your label image stays on your device &mdash; but the OCR software itself originates from China. Review your organization&rsquo;s policy on foreign-developed software before using this option in a government environment. The open-source <b>Tesseract</b> and <b>Server</b> options avoid this.';
  } else { w.style.display='none'; w.innerHTML=''; }
}
// Run one in-browser engine by name. 'paddle' = deep PP-OCR (with a Tesseract.js safety net);
// 'tesseract' = Tesseract.js only (no Baidu code loaded).
async function runEngine(engine,file,onProg){
  if(engine==='tesseract'){ const r=await browserOcr(file,onProg); r.engine='tesseract'; return r; }
  return await bestBrowserOcr(file,onProg);
}

function wordCount(t){return (String(t||'').match(/[A-Za-z]{3,}/g)||[]).length;}
function setBusy(btn,on){if(btn){btn.disabled=on;btn.setAttribute('aria-busy',on?'true':'false');}}
function showProgress(p){const b=$('prog');if(b){b.style.display='block';b.classList.remove('indet');$('progbar').style.width=Math.max(3,p||0)+'%';}}
function hideProgress(){const b=$('prog');if(b){b.style.display='none';b.classList.remove('indet');$('progbar').style.width='0%';}}
// Immediate feedback the moment the user clicks: a "working" line where the answer
// will appear, scrolled into view, plus an indeterminate bar — so the screen never
// looks frozen during the 1-5s wait (Sarah: perceived speed matters as much as speed).
function showWorking(el,msg){el.innerHTML='<div class="working"><span id="wtxt">&#9203; '+esc(msg)+'</span><div class="workbar"><div id="wbar"></div></div></div>';el.scrollIntoView({behavior:'smooth',block:'start'});}
function updateWorking(done,total){const t=$('wtxt'),b=$('wbar');if(t)t.innerHTML='&#9203; Read '+done+' of '+total+' labels…';if(b){b.style.animation='none';b.style.width=Math.round(done/Math.max(1,total)*100)+'%';}}
function fallbackNote(reason){
  return '<div class="fallback">&#9888; In-browser scanning '+esc(reason)+', so we used <b>server processing</b> instead — your image was uploaded to complete this check.</div>';
}
async function serverVerifyImage(file){
  const fd=new FormData();fd.append('image',file);
  ['brand_name','class_type','alcohol_content','net_contents','producer','origin'].forEach(k=>fd.append(k,$(k).value));
  return (await fetch('/verify',{method:'POST',body:fd})).json();
}

/* ==== SINGLE LABEL: verify flow ====================================================
   Orchestrates one verification: validate the file looks like an image, then either OCR in
   the browser (and post text to /verify_text) or upload to /verify. If browser OCR yields
   too few words it throws __LOWREAD__ and we transparently fall back to the server, showing
   the user a note that the image was uploaded. Always restores the button + scrolls to the
   verdict at the end. */
async function verifyOne(){
  if(!singleFile){$('result').innerHTML='<div class="err">Please select a label image first.</div>';return;}
  if(!/^image\//.test(singleFile.type||'')&&!/\.(png|jpe?g|gif|bmp|tiff?|webp|heic)$/i.test(singleFile.name||'')){
    $('result').innerHTML='<div class="err">That file does not look like an image. Please choose a JPG or PNG photo of the label.</div>';return;}
  const engine=selectedEngine('');               // 'paddle' | 'tesseract' | 'server'
  const useBrowser=(engine!=='server');
  setBusy($('goBtn'),true);
  showWorking($('result'), useBrowser?'Reading the label in your browser…':'Checking the label…');
  let d=null, fellBack='';
  try{
    if(useBrowser){
      try{
        $('goBtn').textContent='Reading in your browser…';showProgress(0);
        const t0=Date.now();
        const res=await runEngine(engine,singleFile,p=>{showProgress(p);$('goBtn').textContent='Reading in your browser… '+p+'%';});
        if(wordCount(res.text)<6) throw new Error('__LOWREAD__');  // chosen browser engine failed -> server
        const body={ocr_text:res.text,filename:singleFile.name,client_ms:Date.now()-t0,words:res.words,mean_conf:res.meanConf};
        ['brand_name','class_type','alcohol_content','net_contents','producer','origin'].forEach(k=>body[k]=$(k).value);
        d=await (await fetch('/verify_text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
      }catch(err){
        // In-browser OCR had a problem -> fall back to the server, and tell the user.
        fellBack=(err&&err.message==='__LOWREAD__')?'could not read this image clearly':'could not run on this device';
        $('goBtn').textContent='Using server…';hideProgress();
        d=await serverVerifyImage(singleFile);
      }
    }else{
      $('goBtn').textContent='Processing Label…';
      d=await serverVerifyImage(singleFile);
    }
    if(d&&d.error){$('result').innerHTML='<div class="err">'+esc(d.error)+'</div>';}
    else $('result').innerHTML=(fellBack?fallbackNote(fellBack):'')+renderResult(d);
  }catch(e){
    $('result').innerHTML='<div class="err">'+esc(e.message||'Something went wrong')+'. Please try again.</div>';
  }
  hideProgress();setBusy($('goBtn'),false);$('goBtn').textContent='Verify Label Now';
  $('result').scrollIntoView({behavior:'smooth',block:'start'});
}

function clearSingle(){
  singleFile=null;
  if($('goBtn'))$('goBtn').disabled=true;
  ['brand_name','class_type','alcohol_content','net_contents','producer','origin'].forEach(k=>{if($(k))$(k).value='';});
  if($('file'))$('file').value='';
  $('result').innerHTML='';hideProgress();showImgTools(false);
  drop.innerHTML='<div class="dropmain"><div class="big">&#128247;</div><div class="hint">Drag a clear photo of the label here, or</div></div><div class="dropcta">&#128073; Click here to start</div>';
}

/* ==== RENDERING: result banner + per-field table ==================================
   Pure view code: turn a verifier result object into HTML. Every interpolated value goes
   through esc(). The big banner states the headline verdict in words + icon (never color
   alone — the 73-year-old bar), and the table shows one row per field. */
function fmtMs(ms){ms=ms||0;return ms<100?'under 0.1s':(ms/1000).toFixed(1)+'s';}  // friendly elapsed time
// A read-confidence pill: High (>=80) / Med (>=60) / Low (<60), or an em-dash when unknown.
function confPill(c){
  if(c==null)return '<span class="conf na">—</span>';
  const lvl=c>=80?'hi':c>=60?'md':'lo',word=c>=80?'High':c>=60?'Med':'Low';
  return '<span class="conf '+lvl+'" title="How confident the OCR was reading this field">'+word+' '+c+'%</span>';
}
function statusCell(s){
  const m={pass:'&#9989; PASS',fail:'&#10060; FAIL',lowconf:'&#128064; CHECK BY EYE'};
  return '<td class="s '+s+'">'+(m[s]||'&ndash; NOT CHECKED')+'</td>';
}
function resultsTable(results){
  let h='<table><thead><tr><th>Status</th><th>Label Field</th><th>Expected (App)</th><th>Detected (Label)</th><th>OCR Findings / Notes</th></tr></thead><tbody>';
  (results||[]).forEach(x=>{
    const det = x.detected ? ('<div class="fieldname">'+esc(x.detected)+'</div>' + confPill(x.confidence)) : '<span class="note">Not found</span>';
    // A skipped field we DID read from the label reads as "DETECTED" (not "NOT CHECKED").
    const sc = x.status==='skip' ? (x.detected?'<td class="s detected">&#128065; DETECTED</td>':'<td class="s skip">&ndash; NOT ENTERED</td>') : statusCell(x.status);
    h+='<tr>'+sc+'<td class="fieldname">'+esc(x.field)+'</td><td>'+esc(x.expected)+'</td><td>'+det+'</td><td class="note">'+esc(x.note)+'</td></tr>';});
  return h+'</tbody></table>';
}
// Headline banner, chosen by priority: COULDN'T READ (nothing legible) > FAIL (a real
// mismatch) > NEEDS A LOOK (something unreadable) > EXTRACTED (no app data was entered) >
// PASS. This ordering is deliberate so the most serious state always wins the headline.
function renderResult(d){
  const anyFail=d.results.some(x=>x.status==='fail');
  const anyLow=d.results.some(x=>x.status==='lowconf');
  const skipN=d.results.filter(x=>x.status==='skip').length;
  const provided=d.provided||0;
  let cls,icon,big,sub;
  const poorRead=d.unreadable||(typeof d.mean_conf==='number'&&d.mean_conf<40);  // very low OCR confidence = garbage read
  if(poorRead){cls='warn';icon='&#128247;';big='COULDN’T READ';sub='We couldn’t read this label clearly — the text came back too low-confidence to trust. If this is a photo of a <b>curved or angled bottle</b>, OCR can’t reliably read text that wraps around the glass. Please submit the <b>flat label image</b> (as used in a COLA application) or a straight-on, flattened photo of the label.';}
  else if(anyFail){cls='fail';icon='&#10060;';big='FAIL';sub='This label needs review — see the mismatches below.';}
  else if(anyLow){cls='warn';icon='&#128064;';big='NEEDS A LOOK';sub='Some fields couldn’t be read confidently — please verify the ones marked “check by eye”.';}
  else if(provided===0){cls='pass';icon='&#10024;';big='EXTRACTED';sub='No application details entered, so we extracted what we could from the label.';}
  else{cls='pass';icon='&#9989;';big='PASS';sub=skipN?('All '+provided+' field(s) you entered match. ('+skipN+' not entered.)'):'All checks passed.';}
  let h='<div class="banner '+cls+'"><span class="dot">'+icon+'</span><span class="bmsg"><span class="bigword">'+big+'</span><span class="subline">'+sub+'</span></span><span class="timing">Processed in '+fmtMs(d.total_ms)+'</span></div>';
  if(d.engine==='browser'){h+='<div class="privacy">&#128274; OCR ran entirely in your browser — the image was never uploaded.</div>';}
  h+=resultsTable(d.results);
  h+='<details><summary>View Raw Label Text (OCR)</summary><pre>'+esc(d.ocr_text||'(No text detected)')+'</pre></details>';
  return h;
}

/* ==== BATCH: many labels at once ==================================================
   Same engines, fanned out. Default is a Government-Warning sweep (one shared optional
   field template); browser-OCR batch reuses a single Tesseract worker across all images.
   Results render as a filterable, expandable table with CSV export. */
$('bfile').addEventListener('change',e=>{batchFiles=[...e.target.files];$('bresult').innerHTML='';$('bcount').innerHTML='<b>'+batchFiles.length+'</b> label'+(batchFiles.length===1?'':'s')+' ready — click to add or change';});
['dragover','dragleave','drop'].forEach(ev=>$('bdrop').addEventListener(ev,e=>{e.preventDefault();$('bdrop').classList.toggle('over',ev==='dragover');if(ev==='drop'){batchFiles=[...e.dataTransfer.files];$('bresult').innerHTML='';$('bcount').innerHTML='<b>'+batchFiles.length+'</b> label'+(batchFiles.length===1?'':'s')+' ready — click to add or change';}}));

async function verifyBatch(){
  if(!batchFiles.length){$('bresult').innerHTML='<div class="err">Please select at least one image.</div>';return;}
  const engine=selectedEngine('b_');               // 'paddle' | 'tesseract' | 'server'
  const useBrowser=(engine!=='server');
  $('bGoBtn').disabled=true;
  showWorking($('bresult'),'Checking '+batchFiles.length+' label'+(batchFiles.length>1?'s':'')+'… please keep this page open.');
  const fields={};['brand_name','class_type','alcohol_content','net_contents','producer','origin'].forEach(k=>fields[k]=$('b_'+k).value);
  const t0=Date.now();
  try{
    let items;
    if(useBrowser){
      items=await browserOcrBatch(batchFiles,fields,engine,(done,total)=>{$('bGoBtn').textContent='Reading in your browser… '+done+'/'+total;updateWorking(done,total);});
    }else{
      $('bGoBtn').textContent='Parallel Processing ('+batchFiles.length+' labels)...';
      const fd=new FormData();batchFiles.forEach(f=>fd.append('images',f));
      Object.keys(fields).forEach(k=>fd.append(k,fields[k]));
      const r=await fetch('/batch',{method:'POST',body:fd});
      const d=await r.json();
      if(d.error)throw new Error(d.error);
      items=d.items;
    }
    renderBatch(items,((Date.now()-t0)/1000).toFixed(1),useBrowser);
  }catch(e){
    const hint=useBrowser?' Uncheck “⚡ OCR in my browser” to use server processing instead.':'';
    $('bresult').innerHTML='<div class="err">'+esc(e.message||'Batch error')+hint+'</div>';
  }
  $('bGoBtn').disabled=false;$('bGoBtn').textContent='Start Batch Verification';
  $('bresult').scrollIntoView({behavior:'smooth',block:'start'});
}

// Batch OCR in the browser: ONE reused Tesseract worker reads every image
// sequentially (safe on weak devices; the server never touches the images). Each
// image's text is verified via /verify_text, mapped to the server batch row shape.
async function browserOcrBatch(files,fields,engine,onProg){
  const items=[];
  const verifyImageOnServer=async(file)=>{
    const fd=new FormData();fd.append('image',file);Object.keys(fields).forEach(k=>fd.append(k,fields[k]));
    return (await fetch('/verify',{method:'POST',body:fd})).json();
  };
  const row=(file,d,ms,engine)=>({filename:file.name,passed:!!d.passed,provided:d.provided||0,
    total_ms:ms,fails:(d.results||[]).filter(r=>r.status==='fail').map(r=>r.field),engine,
    results:d.results||[],ocr_text:(d.ocr_text||'').slice(0,3000)});
  for(let i=0;i<files.length;i++){
    const file=files[i];onProg(i,files.length);
    const it0=Date.now();
    try{
      const res=await runEngine(engine,file);        // chosen engine (paddle deep / tesseract)
      if(wordCount(res.text)<6){
        // Even the fallback engine failed on this image -> server.
        items.push(row(file,await verifyImageOnServer(file),Date.now()-it0,'server'));
      }else{
        const body={ocr_text:res.text,filename:file.name,words:res.words,mean_conf:res.meanConf};Object.keys(fields).forEach(k=>body[k]=fields[k]);
        const d=await (await fetch('/verify_text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
        items.push(row(file,d,Date.now()-it0,'browser'));
      }
    }catch(err){
      // In-browser OCR threw -> last-resort server attempt for this one file.
      try{ items.push(row(file,await verifyImageOnServer(file),Date.now()-it0,'server')); }
      catch(e2){ items.push({filename:file.name,passed:false,provided:0,total_ms:Date.now()-it0,fails:['Could not read this image'],engine:'error'}); }
    }
  }
  onProg(files.length,files.length);
  return items;
}

function whereLabel(engine){return engine==='server'?'server (uploaded)':engine==='error'?'unreadable':engine==='browser'?'in browser':'server';}
function renderBatch(items,elapsed,useBrowser){
  const pass=items.filter(x=>x.passed).length;
  const fail=items.length-pass;
  const warnSweep=items.length>0 && items.every(x=>(x.provided||0)===0);  // no app fields entered: a Government-Warning sweep
  const sentToServer=items.filter(x=>x.engine==='server'||x.engine==='error').length;
  const cls=fail===0?'pass':'fail', icon=fail===0?'&#9989;':'&#10060;';
  let big,sub;
  if(warnSweep){big=pass+' / '+items.length+' OK';sub='labels have the Government Warning. (No application details entered — only the warning was checked.)';}
  else if(fail===0){big='PASS';sub='All '+items.length+' labels match the application details.';}
  else{big=fail+' of '+items.length+' need review';sub=pass+' passed, '+fail+' failed.';}
  let h='<div class="banner '+cls+'"><span class="dot">'+icon+'</span><span class="bmsg"><span class="bigword">'+big+'</span><span class="subline">'+sub+'</span></span><span class="timing">Total time: '+elapsed+'s</span></div>';
  if(useBrowser){
    if(sentToServer===0)h+='<div class="privacy">&#128274; OCR ran entirely in your browser — the images were never uploaded.</div>';
    else h+='<div class="fallback">&#9888; '+sentToServer+' of '+items.length+' image'+(items.length>1?'s':'')+' couldn’t be read in your browser, so '+(sentToServer>1?'they were':'it was')+' sent to the <b>server</b> to finish. The rest stayed on your computer.</div>';
  }
  if(fail>0)h+='<label class="filter"><input type="checkbox" onchange="document.getElementById(\'bwrap\').classList.toggle(\'failonly\',this.checked)"> Show only the '+fail+' label'+(fail>1?'s':'')+' that need review</label>';
  h+='<div id="bwrap"><table style="font-size:16px"><thead><tr><th></th><th>Result</th><th>Filename</th><th>Detected / Issues</th><th>Where</th><th>Time</th></tr></thead><tbody>';
  items.forEach((x,i)=>{
    const warnOnly=x.passed&&(x.provided||0)===0;  // warning-sweep pass: the warning is present
    const hasFail=(x.results||[]).some(r=>r.status==='fail');
    const lowOnly=!x.passed&&!hasFail&&(x.results||[]).some(r=>r.status==='lowconf');
    const rowClass=x.passed?'pass':'fail';                       // controls the "show only failures" filter
    const badgeClass=x.passed?'pass':(lowOnly?'warn':'fail');
    const resLabel=warnOnly?'OK':(x.passed?'PASS':(lowOnly?'CHECK':'FAIL'));
    const brand = (x.results || []).find(r => r.field === 'Brand name');
    const brandTxt = brand && brand.detected ? brand.detected : '';
    const issue=x.unreadable?'Couldn’t read this image — try a clearer photo':(warnOnly? (brandTxt ? esc(brandTxt) + ' (Warning OK)' : 'Government Warning present'):(x.passed?'None':(lowOnly?'Some fields couldn’t be read — open to verify':'Missing/Mismatch: '+esc((x.fails||[]).join(', ')))));
    h+='<tr class="brow '+rowClass+'" onclick="toggleBatchRow('+i+')" title="Click to see the label image and full details"><td class="expcell"><span class="exp" id="exp_'+i+'">&#9656;</span></td><td><span class="badge '+badgeClass+'">'+resLabel+'</span></td><td class="fieldname">'+esc(x.filename)+'</td><td class="note">'+issue+'</td><td class="note">'+whereLabel(x.engine)+'</td><td class="note">'+((x.total_ms||0)/1000).toFixed(1)+'s</td></tr>';
    h+='<tr class="bdetail '+rowClass+'" id="bd_'+i+'"><td colspan="6"></td></tr>';
  });
  h+='</tbody></table></div>';
  h+='<button class="go" style="margin-top:18px;background:var(--green)" onclick="exportCsv()">&#11015; Download results as CSV</button>';
  lastBatch=items;$('bresult').innerHTML=h;
}
// Expand a batch row to show the label image (from the user's own copy — never re-stored
// server-side) and the full per-field results, so an agent can adjudicate a failure in place.
function toggleBatchRow(i){
  const row=$('bd_'+i); if(!row)return;
  const open=row.classList.toggle('open');
  const caret=$('exp_'+i); if(caret)caret.innerHTML=open?'&#9662;':'&#9656;';
  if(open && !row.dataset.rendered){
    const it=lastBatch[i]||{}; const cell=row.firstElementChild;
    let h='<div class="bdetailwrap">';
    if(batchFiles[i])h+='<img class="bthumb" src="'+URL.createObjectURL(batchFiles[i])+'" alt="label image">';
    h+='<div class="bdetailbody">';
    h+=(it.results&&it.results.length)?resultsTable(it.results):'<div class="hint">No field details for this image.</div>';
    if(it.ocr_text)h+='<details><summary>View Raw Label Text (OCR)</summary><pre>'+esc(it.ocr_text)+'</pre></details>';
    h+='</div></div>';
    cell.innerHTML=h; row.dataset.rendered='1';
  }
}
function exportCsv(){
  if(!lastBatch||!lastBatch.length)return;
  const head=['Filename','Result','Issues','Processing','Time (s)'];
  const rows=lastBatch.map(x=>{
    const res=x.passed?((x.provided||0)===0?'OK (warning present)':'PASS'):'FAIL';
    return [x.filename,res,(x.passed?'':(x.fails||[]).join('; ')),whereLabel(x.engine),(x.total_ms/1000).toFixed(1)];
  });
  const csv=[head,...rows].map(r=>r.map(c=>'"'+String(c).replace(/"/g,'""')+'"').join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='ttb_label_results.csv';document.body.appendChild(a);a.click();a.remove();
  URL.revokeObjectURL(url);
}
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
