"""Core label-verification engine for the TTB take-home.

Pure logic — no web or OCR dependencies. Given the expected application fields and
the text read off a label, it returns a per-field pass/fail with a human reason.
The matching rules come straight from the stakeholder interviews in the brief:

  * Brand / class / producer  -> FORGIVING. Dave's "STONE'S THROW" vs "Stone's Throw"
    must pass: case-insensitive, punctuation/whitespace-tolerant, fuzzy for OCR slips.
  * Alcohol content (ABV)      -> NUMERIC. Parse the number and compare with a small
    tolerance, so application "45" matches a label's "45% Alc./Vol. (90 Proof)".
  * Government Warning         -> STRICT. Jenny's rule: the exact statutory wording,
    and "GOVERNMENT WARNING:" must be ALL-CAPS. Title-case / altered / missing = fail.

Kept framework-free so it's unit-testable and reusable behind any UI.
"""
from __future__ import annotations

import re
import time

try:  # rapidfuzz is fast + handles OCR noise well; fall back to stdlib if absent.
    from rapidfuzz import fuzz

    def _ratio(a: str, b: str) -> float:
        return fuzz.ratio(a, b) / 100.0

    def _partial(a: str, b: str) -> float:
        return fuzz.partial_ratio(a, b) / 100.0
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _partial(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

# 27 CFR 16.21 — the exact mandated health warning statement.
GOV_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

PASS, FAIL, SKIP, LOWCONF = "pass", "fail", "skip", "lowconf"


def _loose(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for forgiving comparison.
    Also normalizes common OCR slips (0/O, 1/I/l)."""
    s = (s or "").lower()
    s = s.replace("0", "o").replace("1", "i").replace("l", "i")
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s)).strip()


def check_text(expected: str, ocr: str, threshold: float = 0.85) -> tuple[str, str, str | None]:
    """Forgiving presence check for a short field (brand, class, producer, net contents)."""
    e = _loose(expected)
    if not e:
        return SKIP, "no expected value provided", None
    t = _loose(ocr)
    if e in t:
        return PASS, f"“{expected}” found on the label", expected
    score = _partial(e, t)
    if score >= threshold:
        return PASS, f"matched (≈{int(score * 100)}%)", expected
    return FAIL, f"“{expected}” not found on the label (best {int(score * 100)}%)", None


def _normalize_numbers(s: str) -> str:
    """Normalize OCR noise in numbers like '1 2 . 5 %' -> '12.5%'"""
    s = s or ""
    # Remove space between digits
    s = re.sub(r"(\d)\s+(?=\d)", r"\1", s)
    # Remove space around decimal point/comma
    s = re.sub(r"(\d)\s*[\.,]\s*(\d)", r"\1.\2", s)
    return s


def _abv(s: str):
    """Extract a numeric ABV. Prefer an explicit percentage (e.g. '45%' or '45 Alc/Vol')
    over a proof reading, so 'application' wording like '45% Alc./Vol. (90 Proof)' resolves
    to 45 — not 22.5. Falls back to proof/2 only when no percentage is present."""
    s = _normalize_numbers((s or "").lower())
    # Explicit percentage or alc/vol number wins (3 digits so 100 isn't truncated to 10).
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|alc|abv|vol)", s)
    if m:
        return float(m.group(1))
    if "proof" in s:
        m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*proof", s) or re.search(r"(\d{1,3}(?:\.\d+)?)", s)
        if m:
            return float(m.group(1)) / 2.0
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def check_abv(expected: str, ocr: str, tol: float = 0.1) -> tuple[str, str, str | None]:
    """Numeric ABV check: find the percentage(s) on the label, compare to expected."""
    e = _abv(expected)
    if e is None:
        return SKIP, "no expected alcohol content", None
    
    # Pre-clean OCR text for better number extraction
    clean_ocr = _normalize_numbers(ocr or "")
    
    cands = []
    # Explicit %
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", clean_ocr):
        cands.append((float(m.group(1)), m.group(0)))
    # Proof
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*proof", clean_ocr.lower()):
        cands.append((float(m.group(1))/2.0, m.group(0)))
    # General alc/abv numbers
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*(?:alc|abv)", clean_ocr.lower()):
        cands.append((float(m.group(1)), m.group(0)))

    for val, raw in cands:
        if abs(val - e) <= tol:
            return PASS, f"{e:g}% alcohol matches the label", raw
    
    seen = ", ".join(f"{c:g}%" for c in sorted(set(v for v, r in cands))) or "none"
    detected = cands[0][1] if cands else None
    return FAIL, f"label shows {seen}; application says {e:g}%", detected


def check_warning(ocr: str) -> tuple[str, str]:
    """Strict Government Warning check: exact statutory wording + ALL-CAPS heading."""
    raw = ocr or ""
    flat_caps = re.sub(r"\s+", " ", raw)  # collapse OCR line-wraps so "GOVERNMENT\nWARNING:" still matches

    # Requirement: "GOVERNMENT WARNING:" present and ALL-CAPS. (Per 27 CFR the heading is the
    # all-caps part; the body wording must match but need not be uppercase.)
    if "GOVERNMENT WARNING:" not in flat_caps:
        if "government warning" in flat_caps.lower():
            return FAIL, "Present, but “GOVERNMENT WARNING:” is not in ALL-CAPS"
        return FAIL, "Government Warning header is missing"

    # Requirement: Exact statutory wording
    # We use a high threshold but also verify "Critical Keywords" that must be present.
    body_match = _partial(_loose(GOV_WARNING), _loose(raw))
    if body_match < 0.95: # Tightened from 0.92
        return FAIL, f"Statutory wording is incorrect or incomplete ({int(body_match * 100)}%)"
    
    # Critical keyword check to prevent a "forgiving" match on altered meaning.
    # Normalize whitespace first: OCR wraps the warning across lines, so a phrase
    # like "operate machinery" reads as "operate\nmachinery" — collapse runs of
    # whitespace to single spaces before the substring test, or valid labels fail.
    flat = re.sub(r"\s+", " ", raw).lower()
    criticals = ["SURGEON GENERAL", "SHOULD NOT DRINK", "PREGNANCY", "BIRTH DEFECTS",
                 "OPERATE MACHINERY", "DRIVE A CAR", "HEALTH PROBLEMS"]
    missing = [word for word in criticals if word.lower() not in flat]
    if missing:
        return FAIL, f"Missing critical statutory terms: {', '.join(missing)}"

    return PASS, "Present, exact statutory wording, ALL-CAPS heading"


_UNIT = {"ml": "ml", "milliliter": "ml", "milliliters": "ml", "l": "l", "liter": "l",
         "litre": "l", "liters": "l", "litres": "l", "cl": "cl", "floz": "floz", "oz": "floz", "ozs": "floz"}
_QTY_RE = r"(\d+(?:\.\d+)?)\s*(ml|milliliters?|liters?|litres?|l|cl|fl\.?\s*oz|ozs?)\b"


# Convert each unit to milliliters so equal volumes compare equal regardless of how the
# label writes them (imports often use cL; spirits sometimes show fl oz).
_TO_ML = {"ml": 1.0, "cl": 10.0, "l": 1000.0, "floz": 29.5735}


def _qty_ml(s):
    """Parse a net-contents quantity to milliliters: 750 mL == 75 cL == 0.75 L."""
    m = re.search(_QTY_RE, (s or "").lower())
    if not m:
        return None
    unit = _UNIT.get(re.sub(r"[^a-z]", "", m.group(2)), "")
    return float(m.group(1)) * _TO_ML[unit] if unit in _TO_ML else None


def check_quantity(expected, ocr) -> tuple[str, str, str | None]:
    """Net-contents by VOLUME (normalized to mL): 750 mL matches a label's 75 cL or
    0.75 L, but NOT 1750 mL. 1% tolerance absorbs rounding/OCR slips."""
    exp = _qty_ml(expected)
    if exp is None:
        return check_text(expected, ocr)
    tol = max(1.0, exp * 0.01)
    
    first_detected = None
    for m in re.finditer(_QTY_RE, (ocr or "").lower()):
        unit = _UNIT.get(re.sub(r"[^a-z]", "", m.group(2)), "")
        if unit in _TO_ML:
            val_ml = float(m.group(1)) * _TO_ML[unit]
            if first_detected is None:
                first_detected = m.group(0)
            if abs(val_ml - exp) <= tol:
                return PASS, f"{expected} found on the label", m.group(0)
    return FAIL, f"net contents {expected} not found on the label", first_detected


# (application key, display label, checker)
FIELDS = [
    ("brand_name", "Brand name", check_text),
    ("class_type", "Class / type", check_text),
    ("alcohol_content", "Alcohol content", check_abv),
    ("net_contents", "Net contents", check_quantity),
    ("producer", "Bottler / producer", check_text),
    ("origin", "Country of origin", check_text),
]


LOW_CONF = 60   # below this median word-confidence, an UN-FOUND field is "couldn't read", not "failed"
# Markers that say "this is a non-US-market label" — so a missing US warning is a real
# compliance gap, not an OCR miss. (Italian/French/German + UK-specific phrases.)
_FOREIGN = re.compile(r"chief medical|consommation|contiene|prodotto|produkt|sulfit[ie]|drinkaware|verbraucher", re.I)


def _words_conf(value, words):
    """Confidence of the OCR words that spell out `value` (a matched field), or None if
    those words can't be located. Sliding window over the (word, conf) list."""
    e = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    if not e or not words:
        return None
    toks = [(re.sub(r"[^a-z0-9]", "", w.lower()), c) for w, c in words]
    for i in range(len(toks)):
        acc, cs = "", []
        for j in range(i, min(i + 8, len(toks))):
            acc += toks[j][0]; cs.append(toks[j][1])
            if e in acc:
                return sum(cs) / len(cs)
    return None


def _absent(key, expected, ocr):
    """Was the field's subject not located on the label at all (vs. found-but-mismatched)?
    Only an ABSENT field on a LOW-confidence read becomes 'couldn't read'."""
    low = (ocr or "").lower()
    if key == "alcohol_content":
        return not re.search(r"\d{1,3}(?:\.\d+)?\s*(?:%|proof|alc|abv|vol)", low)
    if key == "net_contents":
        return not re.search(_QTY_RE, low)
    if key == "__warning__":
        return "government warning" not in low
    return _partial(_loose(expected), _loose(ocr)) < 0.55


# Class/type lexicon — LONGEST specific phrases first so "kentucky straight bourbon whiskey"
# wins over "whiskey" and "cabernet sauvignon" over "wine". (Codex: a real lexicon, not 3 words.)
_CLASS_LEXICON = [
    "kentucky straight bourbon whiskey", "single malt scotch whisky", "blended scotch whisky",
    "straight bourbon whiskey", "straight rye whiskey", "tennessee whiskey", "irish whiskey",
    "bourbon whiskey", "rye whiskey", "scotch whisky", "cabernet sauvignon", "sauvignon blanc",
    "pinot grigio", "pinot noir", "chardonnay", "merlot", "zinfandel", "riesling", "syrah",
    "india pale ale", "pale ale", "amber ale", "imperial stout", "wheat beer", "pilsner",
    "whiskey", "whisky", "bourbon", "vodka", "gin", "tequila", "mezcal", "rum", "brandy",
    "cognac", "liqueur", "red wine", "white wine", "sparkling wine", "rose", "wine", "lager",
    "ale", "stout", "porter", "cider", "sake", "beer",
]


# A brand line must NOT be a warning, regulatory, award, or field-value line.
_NOT_BRAND = re.compile(
    r"government|warning|smoking|alcohol abuse|consumption|injurious|responsibly|chief medical|"
    r"surgeon|dangerous|health|denominaz|indicazione|controllat|garantit|award|medall|contains|"
    r"sulfit|gluten|allerg|product of|bottled|imported|distilled|brewed|"
    r"\d{2,}\s*%|\bproof\b|\b\d+\s*(?:ml|cl|l)\b", re.I)


def _extract_detected(key, ocr_text, words=None, mean_conf=None):
    """Best-effort read of what the LABEL says for a field, for the 'Detected' column.

    GATED BY READ CONFIDENCE + PLAUSIBILITY so we surface real values on legible labels and
    stay SILENT (rather than show OCR garbage) on hard ones — showing a confidently-wrong
    brand is worse than showing nothing. Conservative: leaves a field blank when unsure."""
    low = (ocr_text or "").lower()
    lines = [l.strip() for l in (ocr_text or "").splitlines() if l.strip()]
    mc = mean_conf if mean_conf is not None else 100

    if mc < 40:
        return None                                       # read too poor to extract anything reliably

    if key == "alcohol_content":
        for m in re.finditer(r"(\d{1,2}(?:\.\d+)?)\s*%", _normalize_numbers(low)):
            if 4.0 <= float(m.group(1)) <= 70.0:          # plausible ABV (not 100% grape, not a 2% misread)
                return m.group(0).strip()
        m = re.search(r"(\d{2,3})\s*proof", low)
        return m.group(0).strip() if (m and 40 <= int(m.group(1)) <= 200) else None
    if key == "net_contents":
        for m in re.finditer(_QTY_RE, low):
            unit = _UNIT.get(re.sub(r"[^a-z]", "", m.group(2)), "")
            if unit in _TO_ML and 50 <= float(m.group(1)) * _TO_ML[unit] <= 2000:   # plausible bottle size
                return m.group(0).strip()
        return None

    if mc < 60:
        return None                                       # legible enough for numbers, not free-text fields

    if key == "origin":
        for l in lines:                                   # line-by-line: never run into the next line
            m = re.search(r"\b(?:product|produced|made|bottled)\s+(?:of|in)\s+([A-Za-z][A-Za-z .'-]{2,28})", l, re.I)
            if m:
                return ("Product of " + m.group(1).strip().rstrip(".")).strip()
        return None
    if key == "producer":
        for l in lines:                                   # ONLY when anchored — guessing here makes bad data
            m = re.search(r"\b(?:bottled|produced|distilled|brewed|imported)\s+(?:by|for)\b\s*(.+)", l, re.I)
            if m and m.group(1).strip():
                return m.group(1).strip()[:60]
        return None
    if key == "class_type":
        for kw in _CLASS_LEXICON:                         # longest specific phrase wins
            if re.search(r"\b" + re.escape(kw) + r"\b", low):   # word-bounded: "ale" must not match "male"
                for l in lines:
                    if re.search(r"\b" + re.escape(kw) + r"\b", l.lower()):
                        return l[:60]
                return kw.title()
        return None
    if key == "brand_name":
        for l in lines[:8]:                               # a prominent top line that isn't warning/regulatory
            if _NOT_BRAND.search(l):
                continue
            letters = re.sub(r"[^A-Za-z]", "", l)
            if len(letters) < 4 or not re.search(r"[A-Za-z]{4,}", l):
                continue                                  # too short / no real word -> OCR fragment
            if len(letters) / max(1, len(l.replace(" ", ""))) < 0.6:
                continue                                  # mostly punctuation/symbols -> garbage
            return l[:60]
        return None
    return None


def verify(fields: dict, ocr_text: str, words=None, mean_conf=None) -> dict:
    """Run every provided field check + the mandatory warning check.

    Optional `words` ([(text, conf)]) and `mean_conf` (median word confidence) enable a
    per-field READ CONFIDENCE and a 'couldn't read — verify by eye' (LOWCONF) state, so a
    field the OCR simply couldn't read isn't reported as a hard compliance FAIL. Returns
    {results:[{field, expected, status, note, confidence, detected}], passed, provided,
    unreadable, mean_conf, elapsed_ms}. `passed` is true only when every checked field is PASS.
    """
    t0 = time.time()
    mc = float(mean_conf) if mean_conf is not None else None
    results = []
    provided = 0

    real = words is not None   # real OCR path? (the matrix / unit tests pass text only)

    def _conf_for(detected):
        # Confidence is about the value we READ off the label (the detected value): the
        # confidence of its exact source words, falling back to the overall median.
        if not detected:
            return None
        if words:
            c = _words_conf(detected, words)
            if c is not None:
                return round(c)
        return round(mc) if mc is not None else None

    for key, label, checker in FIELDS:
        expected = (fields or {}).get(key)
        detected = None
        if expected in (None, ""):
            detected = _extract_detected(key, ocr_text, words, mc)
            note = "read from label — no application value to verify against" if detected else "— not entered —"
            results.append({"field": label, "expected": "—", "status": SKIP,
                            "note": note, "confidence": _conf_for(detected), "detected": detected})
            continue
        provided += 1
        status, note, detected = checker(expected, ocr_text)
        # A field whose value isn't on the label AT ALL is ambiguous — unreadable vs. truly
        # absent — so on the real OCR path we flag it "verify by eye" rather than asserting a
        # hard FAIL. Hard FAIL stays for genuine MISMATCHES (a value that contradicts the app).
        if status == FAIL and real and _absent(key, expected, ocr_text):
            status, note = LOWCONF, "Couldn’t read this field on the image — verify by eye"
        results.append({"field": label, "expected": expected, "status": status,
                        "note": note, "confidence": _conf_for(detected), "detected": detected})

    ws, wn = check_warning(ocr_text)
    low_text = (ocr_text or "").lower()
    us_absent = "government warning" not in low_text
    # A NON-US health warning (UK "drink responsibly", EU, South Africa, etc.) is present but
    # does NOT satisfy the mandatory US Government Warning — flag it explicitly instead of
    # reporting "Missing", which wrongly reads as "no warning at all".
    foreign_warn = bool(re.search(
        r"drink responsibly|chief medical|drinkaware|alcohol abuse|consommation|à consommer|schwanger|embarazo",
        low_text))
    if ws == FAIL and us_absent and foreign_warn:
        ws = FAIL  # the US warning is mandatory regardless of any foreign warning
        wn = ("A non-US health warning is present (e.g. UK “drink responsibly” / Chief Medical "
              "Officers), but the mandatory US Government Warning is missing — looks like a "
              "foreign-market label.")
        w_detected = "Non-US warning only"
    elif ws == FAIL and us_absent and real and mc is not None and mc < LOW_CONF:
        ws, wn = LOWCONF, "Couldn’t confirm the Government Warning — verify by eye"
        w_detected = "Unclear — verify by eye"
    elif ws == PASS:
        w_detected = "Present"
    elif us_absent:
        w_detected = "Missing"
    else:
        w_detected = "Present, but not compliant"
    results.append({"field": "Government Warning", "expected": "statutory text, ALL-CAPS",
                    "status": ws, "note": wn, "detected": w_detected,
                    "confidence": round(mc) if (ws in (PASS, FAIL) and mc is not None) else None})

    # Don't surface the same OCR line as both Brand and Class.
    bd = next((r for r in results if r["field"] == "Brand name"), None)
    cd = next((r for r in results if r["field"] == "Class / type"), None)
    if bd and cd and bd.get("detected") and cd.get("detected") == bd.get("detected"):
        cd["detected"] = None
        if cd["status"] == SKIP:
            cd["confidence"], cd["note"] = None, "— not entered —"

    # PASS only if every checked field is PASS (SKIP = not entered). FAIL or LOWCONF -> not passed.
    passed = all(r["status"] in (PASS, SKIP) for r in results)
    unreadable = len(re.findall(r"[A-Za-z]{3,}", ocr_text or "")) < 5
    return {"results": results, "passed": passed, "provided": provided,
            "unreadable": unreadable, "mean_conf": round(mc) if mc is not None else None,
            "elapsed_ms": int((time.time() - t0) * 1000)}
