"""Core label-verification engine for the TTB take-home.

WHAT THIS FILE IS
-----------------
The compliance brain of the tool. It is PURE LOGIC — no Flask, no Tesseract, no file
I/O — which is what makes it unit-testable (see tests/test_verifier.py) and reusable
behind any front end. It takes two inputs:

  1. the application fields a reviewer typed (what the COLA application *claims*), and
  2. the raw text an OCR engine read off the label image,

and returns a per-field PASS / FAIL / "verify by eye" verdict, each with a plain-English
reason and a read-confidence number. `verify()` at the bottom is the single entry point;
everything above it is a helper it composes.

TWO JOBS ON ONE SCREEN (kept deliberately separate)
---------------------------------------------------
  * VERIFICATION  — does the label match the application? (the core task)
  * EXTRACTION    — independently, what does the label itself say for each field?
    (the "Detected on label" column, produced by `_extract_detected`)
Expected (application) and Detected (label) are two INDEPENDENT sources. We never check
the label against itself — that would be circular and meaningless — so extraction is a
separate, confidence-gated read used only for display, never to decide a PASS.

THE STATUS MODEL
----------------
  PASS    — the entered application value matches the label.
  FAIL    — a genuine, confident MISMATCH (label says 40%, application says 45%), or a
            mandatory element (the US Government Warning) that is truly absent/altered.
  SKIP    — the reviewer left this field blank; nothing to verify (never a failure).
  LOWCONF — the OCR could not read this field, so we say "verify by eye" instead of
            asserting a FAIL we are not sure of. Crucial distinction: an UNREADABLE field
            is ambiguous (unreadable vs. truly absent), and a compliance tool must not
            cry "FAIL" on a bad photo. (Maps to Jenny's "if we can't read it, ask for a
            better image.") Only a confident contradiction earns a hard FAIL.

DESIGN PRINCIPLES — straight from the stakeholder interviews in the brief
------------------------------------------------------------------------
  * Brand / class / producer  -> FORGIVING. Dave's "STONE'S THROW" vs "Stone's Throw"
    must pass: case-insensitive, punctuation/whitespace-tolerant, fuzzy for OCR slips.
  * Alcohol content (ABV)      -> NUMERIC. Parse the number and compare with a small
    tolerance, so application "45" matches a label's "45% Alc./Vol. (90 Proof)".
  * Net contents               -> VOLUME-NORMALIZED. 750 mL, 75 cL and 0.75 L are equal.
  * Government Warning         -> STRICT. Jenny's rule: the exact statutory wording, and
    "GOVERNMENT WARNING:" must be ALL-CAPS. Title-case / altered / missing = fail. A
    NON-US warning (UK "drink responsibly", etc.) is reported distinctly, not as "missing".
  * Confidence everywhere      -> when given per-word OCR confidence, we expose it per
    field and let it gate both the LOWCONF state and what extraction dares to display.

Statuses are plain string constants (below) so callers and tests can compare directly.
"""
from __future__ import annotations

import re
import time

# Fuzzy string similarity backend.
# We prefer rapidfuzz: it is fast (C-backed) and tolerant of the character-level noise
# OCR produces. If it isn't installed we transparently fall back to the standard library's
# difflib so the engine still runs anywhere — no hard dependency on a native wheel.
#   _ratio(a, b)   -> full-string similarity in 0..1 (whole strings must be alike)
#   _partial(a, b) -> best-substring similarity in 0..1 (a short needle inside a long
#                     haystack scores high — the right tool for "is the brand somewhere
#                     in this page of label text?")
try:  # rapidfuzz is fast + handles OCR noise well; fall back to stdlib if absent.
    from rapidfuzz import fuzz

    def _ratio(a: str, b: str) -> float:
        return fuzz.ratio(a, b) / 100.0

    def _partial(a: str, b: str) -> float:
        return fuzz.partial_ratio(a, b) / 100.0
except Exception:  # pragma: no cover  (difflib is slower and has no partial_ratio, so we
    from difflib import SequenceMatcher          # approximate partial with full ratio)

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _partial(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

# 27 CFR 16.21 — the EXACT mandated health-warning statement, verbatim. This is the
# single source of truth the strict warning check (`check_warning`) matches against; the
# sample-label generators import it too, so test labels and the checker never drift apart.
GOV_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

# Per-field verdict constants. Plain strings (not an enum) so callers, JSON, and unit
# tests can compare against them directly without importing a type. See "THE STATUS
# MODEL" in the module docstring for the meaning of each.
PASS, FAIL, SKIP, LOWCONF = "pass", "fail", "skip", "lowconf"


def _loose(s: str) -> str:
    """Canonicalize a string for FORGIVING comparison — the heart of "STONE'S THROW"
    matching "Stone's Throw".

    Pipeline: lowercase → fold the OCR look-alikes that Tesseract most often confuses
    (0↔O, 1↔I↔l) onto a single canonical letter → drop every non-alphanumeric character →
    collapse runs of whitespace to single spaces → trim. Two strings that differ only in
    case, punctuation, spacing, or those specific OCR slips become byte-identical, so a
    plain substring test (`in`) suffices for the common case.

    Args:    s: any string (None is treated as "").
    Returns: the normalized form, e.g. "Stone's Throw!" -> "stones throw".
    """
    s = (s or "").lower()
    s = s.replace("0", "o").replace("1", "i").replace("l", "i")
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s)).strip()


def check_text(expected: str, ocr: str, threshold: float = 0.85) -> tuple[str, str, str | None]:
    """Forgiving PRESENCE check for a short identity field (brand, class, producer).

    "Present" rather than "equal": we ask whether the expected value appears ANYWHERE in
    the label's text, because the application value is one phrase while the OCR text is the
    whole label. Two-step: first an exact substring test on the `_loose`-normalized strings
    (fast, handles case/punctuation/OCR-slips); if that misses, a fuzzy `_partial` ratio
    catches residual OCR noise, passing when it clears `threshold` (default 0.85).

    Args:
        expected:  the application value (blank -> SKIP, nothing to verify).
        ocr:       the full text read off the label.
        threshold: minimum fuzzy similarity (0..1) to accept when the exact test misses.
    Returns:
        (status, human_reason, detected) — a 3-tuple shared by every checker. `detected`
        echoes the expected value on a PASS (it was confirmed present) and is None on FAIL.
    """
    e = _loose(expected)
    if not e:
        return SKIP, "no expected value provided", None
    t = _loose(ocr)
    if e in t:                                       # exact (post-normalization) substring hit
        return PASS, f"“{expected}” found on the label", expected
    score = _partial(e, t)                           # fall back to fuzzy: absorb OCR noise
    if score >= threshold:
        return PASS, f"matched (≈{int(score * 100)}%)", expected
    return FAIL, f"“{expected}” not found on the label (best {int(score * 100)}%)", None


def _normalize_numbers(s: str) -> str:
    """Repair OCR's two classic number-mangling habits before we parse digits.

    OCR frequently sprays spaces through numbers ("1 2 . 5 %") because it segments glyphs
    individually. Left alone, a regex would read "12.5%" as a "1" and a "2". We (1) delete
    spaces sitting between two digits and (2) tidy spaces around a decimal point/comma, so
    "1 2 . 5 %" becomes "12.5%". Everything else is left untouched.

    Args:    s: raw OCR text (None -> "").
    Returns: the same text with intra-number spacing repaired.
    """
    s = s or ""
    s = re.sub(r"(\d)\s+(?=\d)", r"\1", s)            # join digits split by spaces: "4 5" -> "45"
    s = re.sub(r"(\d)\s*[\.,]\s*(\d)", r"\1.\2", s)   # tidy the decimal: "12 . 5" -> "12.5"
    return s


def _abv(s: str):
    """Parse ONE alcohol-by-volume number out of a string, resolving the %-vs-proof ambiguity.

    A spirit label commonly prints both forms — "45% Alc./Vol. (90 Proof)" — and proof is
    exactly twice the percentage. If we naively grabbed the first number we might return 90,
    or halve it to 45 by luck; if we grabbed proof we'd return 22.5 for a 45% spirit. So the
    rule is explicit and deterministic:

        1. An explicit PERCENTAGE / alc / abv / vol number wins (this is the real ABV).
        2. Only if there is NO percentage do we read "proof" and divide by two.
        3. Failing both, take any bare number as a last resort.

    The `\\d{1,3}` (not `\\d{1,2}`) matters: it lets "100" and "151" (overproof rum) parse
    whole instead of truncating to "10"/"15".

    Args:    s: a string that may contain an ABV (the application value, or a label snippet).
    Returns: the ABV as a float, or None if no number is present.
    """
    s = _normalize_numbers((s or "").lower())
    # (1) Explicit percentage / alc-vol number wins — this is the true ABV.
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|alc|abv|vol)", s)
    if m:
        return float(m.group(1))
    # (2) No percentage anywhere -> a proof reading, converted to ABV (proof / 2).
    if "proof" in s:
        m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*proof", s) or re.search(r"(\d{1,3}(?:\.\d+)?)", s)
        if m:
            return float(m.group(1)) / 2.0
    # (3) Last resort: any bare number.
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def check_abv(expected: str, ocr: str, tol: float = 0.1) -> tuple[str, str, str | None]:
    """Numeric ABV check: collect every alcohol figure on the label and see if one equals
    the application's claimed value (within a small tolerance).

    Why collect ALL candidates instead of one: a label can show the percentage and the proof
    and sometimes a serving figure, in any order. We gather each as a (value, raw_text) pair —
    percentages as-is, proof halved, alc/abv numbers as-is — then PASS if any candidate lands
    within `tol` of the expected ABV. If none match, the reason lists what we did see, which
    is far more useful to a reviewer than a bare "mismatch".

    Args:
        expected: the application's alcohol content (e.g. "45", "45%", "90 proof").
        ocr:      the full label text.
        tol:      absolute ABV tolerance in percentage points (default 0.1) — absorbs
                  rounding and OCR jitter without letting 40% pass for 45%.
    Returns:
        (status, human_reason, detected). On PASS, `detected` is the exact label substring
        that matched; on FAIL, the first alcohol figure seen (or None if the label had none).
    """
    e = _abv(expected)
    if e is None:
        return SKIP, "no expected alcohol content", None

    # Repair split digits once, up front, so all three patterns below see clean numbers.
    clean_ocr = _normalize_numbers(ocr or "")

    cands = []  # [(abv_value, raw_label_text), ...] — every alcohol figure found on the label
    # Explicit percentages: "45%", "13.5 %".
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", clean_ocr):
        v = float(m.group(1))
        cands.append((v, m.group(0)))
        # OCR sometimes drops the decimal point ("13.5%" read as "135%"). A literal ABV >= 100
        # is physically impossible, so when we see one that ISN'T a round hundred (those are
        # usually content claims like "100% agave"), also offer the decimal-restored reading
        # (135 -> 13.5) as a candidate. Harmless: it only matches if the application entered it.
        if v > 100 and v % 100 != 0:
            cands.append((v / 10.0, m.group(0)))
    # Proof readings, converted to ABV (proof / 2): "90 Proof" -> 45.
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*proof", clean_ocr.lower()):
        cands.append((float(m.group(1))/2.0, m.group(0)))
    # Numbers explicitly tagged alc/abv but without a "%" sign.
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*(?:alc|abv)", clean_ocr.lower()):
        cands.append((float(m.group(1)), m.group(0)))

    for val, raw in cands:
        if abs(val - e) <= tol:
            return PASS, f"{e:g}% alcohol matches the label", raw

    # No candidate matched — report what the label actually showed, for the reviewer.
    seen = ", ".join(f"{c:g}%" for c in sorted(set(v for v, r in cands))) or "none"
    detected = cands[0][1] if cands else None
    return FAIL, f"label shows {seen}; application says {e:g}%", detected


def check_warning(ocr: str) -> tuple[str, str]:
    """Strict Government Warning check — the one element the brief says must be exact.

    This is intentionally the LEAST forgiving check in the file (Jenny's rule). It clears
    two hurdles, each of which can fail the label on its own:

      1. ALL-CAPS HEADING. The literal "GOVERNMENT WARNING:" must appear in upper case. We
         first collapse whitespace so an OCR line-wrap ("GOVERNMENT\\nWARNING:") still counts,
         then test case-sensitively. If the words are present but not all-caps (a title-case
         "Government Warning:"), that is a specific, reportable defect — not "missing".
      2. EXACT STATUTORY WORDING — a CONTIGUOUS-SUBSTRING test, NOT a fuzzy ratio. The canonical
         27 CFR 16.21 text must appear contiguously inside the OCR after `_loose` normalization
         (lowercase, punctuation/whitespace collapsed, OCR slips 0↔O/1↔I↔l folded). This is what
         rejects MEANING-INVERTED warnings — "ENHANCES your ability to drive a car", "MEN should
         not drink", "ABSENCE of birth defects" — which a forgiving fuzzy ratio wrongly accepts
         (a one-word inversion still scores >0.95). Anything short of an exact (slip-tolerant)
         match is reported as altered/defective, never PASS; on the real-OCR path verify() softens
         a low-confidence near-miss to "verify by eye".

    Why so strict: the health warning is statutorily mandated word-for-word, so "close enough"
    is the wrong default here even though it is the right default for a brand name.

    Args:    ocr: the full label text.
    Returns: (status, human_reason). Only PASS or FAIL — no SKIP (the warning is never
             optional) and no LOWCONF here (the confidence-aware "verify by eye" downgrade is
             applied by `verify()`, which knows the read's mean confidence).
    """
    raw = ocr or ""
    flat_caps = re.sub(r"\s+", " ", raw)  # collapse OCR line-wraps so "GOVERNMENT\nWARNING:" still matches

    # Hurdle 1 — the ALL-CAPS heading. (Per 27 CFR the heading is the all-caps part; the body
    # wording must match but need not itself be uppercase.)
    if "GOVERNMENT WARNING:" not in flat_caps:
        if "government warning" in flat_caps.lower():
            return FAIL, "Present, but “GOVERNMENT WARNING:” is not in ALL-CAPS"
        return FAIL, "Government Warning header is missing"

    # Hurdle 2 — EXACT wording via CONTIGUOUS SUBSTRING (the fix for meaning-inverted warnings).
    # _loose() lowercases, strips punctuation, collapses whitespace (so OCR line-wraps are fine)
    # and folds OCR slips 0↔O/1↔I↔l. If the canonical warning appears CONTIGUOUSLY in the
    # normalized OCR, the wording is exact-enough; otherwise it's altered/incomplete/too garbled
    # to trust. A fuzzy ratio was used here before and wrongly PASSed one-word inversions.
    loose_raw = _loose(raw)
    if _loose(GOV_WARNING) in loose_raw:
        return PASS, "Present, exact statutory wording, ALL-CAPS heading"

    # Present but NOT an exact match — surface which load-bearing phrases are missing/altered.
    # This both explains the failure and pinpoints meaning inversions (impairs↔enhances,
    # women↔men, risk-of↔absence-of). Still a FAIL; verify() may soften it to "verify by eye"
    # on a low-confidence real read.
    criticals = ["women should not drink", "during pregnancy", "risk of birth defects",
                 "impairs your ability", "operate machinery", "drive a car", "health problems",
                 "surgeon general"]
    missing = [c for c in criticals if _loose(c) not in loose_raw]
    if missing:
        return FAIL, f"Government Warning wording is altered or incomplete (missing/changed: {', '.join(missing)})"
    return FAIL, "Government Warning wording does not exactly match 27 CFR 16.21 — verify by eye"


# Net-contents unit handling.
# _UNIT maps the many ways a label can spell a unit ("milliliters", "litre", "ozs", …) onto
# a small set of canonical keys. _QTY_RE pulls "<number> <unit>" pairs out of label text
# (note "fl.? oz" allows "fl oz" / "fl. oz" / "floz"). _TO_ML then converts a canonical unit
# to milliliters — the common denominator — so that 750 mL, 75 cL and 0.75 L all reduce to
# the same 750.0 and compare equal regardless of how a given label chose to print the volume
# (imports often use cL; some spirits show fl oz).
_UNIT = {"ml": "ml", "milliliter": "ml", "milliliters": "ml", "l": "l", "liter": "l",
         "litre": "l", "liters": "l", "litres": "l", "cl": "cl", "floz": "floz", "oz": "floz", "ozs": "floz"}
_QTY_RE = r"(\d+(?:\.\d+)?)\s*(ml|milliliters?|liters?|litres?|l|cl|fl\.?\s*oz|ozs?)\b"
_TO_ML = {"ml": 1.0, "cl": 10.0, "l": 1000.0, "floz": 29.5735}


def _qty_ml(s):
    """Parse the first "<number> <unit>" quantity in a string into milliliters.

    Collapses the unit's spelling to a canonical key via `_UNIT`, then scales by `_TO_ML`,
    so 750 mL, 75 cL and 0.75 L all return 750.0.

    Args:    s: a string that may contain a volume (None -> None).
    Returns: the volume in mL as a float, or None if no recognizable quantity is found.
    """
    m = re.search(_QTY_RE, (s or "").lower())
    if not m:
        return None
    unit = _UNIT.get(re.sub(r"[^a-z]", "", m.group(2)), "")
    return float(m.group(1)) * _TO_ML[unit] if unit in _TO_ML else None


def check_quantity(expected, ocr) -> tuple[str, str, str | None]:
    """Net-contents check by VOLUME rather than by string, so unit differences don't matter.

    Both sides are reduced to milliliters and compared with a 1% tolerance (floored at 1 mL):
    an application "750 mL" matches a label printed as "75 cL" or "0.75 L", but NOT "1750 mL".
    If the expected value isn't a parseable volume (some net-contents fields are free text,
    e.g. "1 PINT 9 FL OZ"), we gracefully fall back to the forgiving `check_text` path.

    Args:
        expected: the application's net contents.
        ocr:      the full label text.
    Returns:
        (status, human_reason, detected). On PASS, `detected` is the matching label substring;
        on FAIL, the first quantity seen on the label (or None).
    """
    exp = _qty_ml(expected)
    if exp is None:
        return check_text(expected, ocr)          # expected isn't a volume -> forgiving text match
    tol = max(1.0, exp * 0.01)                     # 1% tolerance, but never tighter than 1 mL

    first_detected = None
    for m in re.finditer(_QTY_RE, (ocr or "").lower()):
        unit = _UNIT.get(re.sub(r"[^a-z]", "", m.group(2)), "")
        if unit in _TO_ML:
            val_ml = float(m.group(1)) * _TO_ML[unit]
            if first_detected is None:
                first_detected = m.group(0)        # remember the first quantity, for the FAIL note
            if abs(val_ml - exp) <= tol:
                return PASS, f"{expected} found on the label", m.group(0)
    return FAIL, f"net contents {expected} not found on the label", first_detected


# The field registry that drives `verify()`: one row per checkable field, as
# (application_key, human_display_label, checker_function). Order here is the order results
# appear in the UI. To add a field you add a row — the engine loops over this table.
FIELDS = [
    ("brand_name", "Brand name", check_text),
    ("class_type", "Class / type", check_text),
    ("alcohol_content", "Alcohol content", check_abv),
    ("net_contents", "Net contents", check_quantity),
    ("producer", "Bottler / producer", check_text),
    ("origin", "Country of origin", check_text),
]


# Median word-confidence threshold. At or below this, a field we could NOT find on the label
# is treated as "couldn't read" (LOWCONF / verify by eye) rather than a hard FAIL — because on
# a poor read, absence is ambiguous. Above it, a genuinely missing value can be trusted as a
# real mismatch.
LOW_CONF = 60
# Markers that say "this is a non-US-market label" — so a missing US warning is a real
# compliance gap, not an OCR miss. (Italian/French/German + UK-specific phrases.) Currently
# kept for reference/future use; the live foreign-warning decision in verify() uses a more
# targeted phrase set tuned to the warning sentence itself.
_FOREIGN = re.compile(r"chief medical|consommation|contiene|prodotto|produkt|sulfit[ie]|drinkaware|verbraucher", re.I)


def _words_conf(value, words):
    """Find the OCR confidence of the specific words that spell out `value`.

    Used to attach a read-confidence to the value we DETECTED for a field (rather than the
    whole-label median). We strip both `value` and each OCR token down to bare alphanumerics,
    then slide an up-to-8-token window across the word list, concatenating as we go, until the
    normalized value appears inside the concatenation — and return the mean confidence of the
    tokens that formed it.

    Args:
        value: the detected/matched string whose source words we want to score.
        words: the OCR word list as [(text, confidence_0_100), ...].
    Returns:
        mean confidence (0..100) of the words spelling `value`, or None if they can't be
        located (or there are no words — the text-only unit-test path).
    """
    e = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    if not e or not words:
        return None
    toks = [(re.sub(r"[^a-z0-9]", "", w.lower()), c) for w, c in words]
    for i in range(len(toks)):
        acc, cs = "", []
        for j in range(i, min(i + 8, len(toks))):     # window of up to 8 tokens (multi-word values)
            acc += toks[j][0]; cs.append(toks[j][1])
            if e in acc:
                return sum(cs) / len(cs)
    return None


def _absent(key, expected, ocr):
    """Distinguish "the label never mentions this" from "found it, but it's wrong".

    This is what lets `verify()` downgrade an unreadable field to LOWCONF instead of FAIL: a
    FAIL only deserves to stand if the subject was actually PRESENT and contradicted the
    application. For the structured fields we look for the SHAPE of a value (any ABV-like
    number, any quantity, the warning header); for free-text fields we ask whether even a
    loose fuzzy trace of the expected value exists (< 0.55 similarity == effectively absent).

    Args:
        key:      the field's application key (selects the absence test).
        expected: the application value (used only for the free-text fuzzy test).
        ocr:      the full label text.
    Returns:
        True if the field's subject appears to be missing from the label entirely.
    """
    low = (ocr or "").lower()
    if key == "alcohol_content":
        return not re.search(r"\d{1,3}(?:\.\d+)?\s*(?:%|proof|alc|abv|vol)", low)   # no ABV-shaped number
    if key == "net_contents":
        return not re.search(_QTY_RE, low)                                          # no quantity at all
    if key == "__warning__":
        return "government warning" not in low
    return _partial(_loose(expected), _loose(ocr)) < 0.55                           # no fuzzy trace -> absent


# Class/type lexicon used by extraction to recognize the beverage type on a label.
# Ordered LONGEST / most-specific phrase FIRST and scanned in order, so the most precise
# match wins: "kentucky straight bourbon whiskey" beats the bare "whiskey", and "cabernet
# sauvignon" beats "wine". Without this ordering a generic word would shadow the specific one.
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


# Brand-name extraction picks a prominent top line — but the most prominent lines are often
# NOT the brand. This pattern rejects any candidate line that is really a warning, regulatory
# notice, appellation, award, allergen, process statement, or a field value (ABV / proof /
# volume). If a line matches this, it is skipped as a brand candidate.
_NOT_BRAND = re.compile(
    r"government|warning|smoking|alcohol abuse|consumption|injurious|responsibly|chief medical|"
    r"surgeon|dangerous|health|denominaz|indicazione|controllat|garantit|award|medall|contains|"
    r"sulfit|gluten|allerg|product of|bottled|imported|distilled|brewed|"
    r"\d{2,}\s*%|\bproof\b|\b\d+\s*(?:ml|cl|l)\b", re.I)


def _extract_detected(key, ocr_text, words=None, mean_conf=None):
    """Best-effort read of what the LABEL itself says for one field — the "Detected" column.

    This powers EXTRACTION (independent of verification). Its guiding principle is
    "silence beats a confident wrong answer": on a hard or low-confidence read it returns
    None and the UI simply shows nothing, which is far better for a compliance tool than
    displaying OCR garbage that looks authoritative.

    Two gates keep it honest:
      * READ CONFIDENCE — below 40 we extract nothing at all; numbers (ABV, net contents)
        are allowed down to 40, but free-text fields (brand, class, producer, origin) need
        60+, since a wrong word is more misleading than a wrong-looking number.
      * PLAUSIBILITY — values must fall in sane ranges (ABV 4–70%, bottle 50–2000 mL) so a
        "100%" that is really grape content, or a "70 L" misread, is ignored.

    Per-key strategy: numbers via tolerant regex; origin/producer ONLY when anchored by a
    cue word ("Product of", "Bottled by") and read line-by-line so we never bleed into the
    next line; class via the longest-phrase lexicon with word boundaries; brand as the first
    prominent top line that survives the `_NOT_BRAND` filter and looks like a real word.

    Args:
        key:       the field's application key (selects the strategy).
        ocr_text:  the full label text.
        words:     optional [(text, conf)] (unused here, accepted for signature symmetry).
        mean_conf: median read confidence (0..100); the confidence gate. None == treat as 100.
    Returns:
        the detected string for display, or None when unsure / gated out.
    """
    low = (ocr_text or "").lower()
    lines = [l.strip() for l in (ocr_text or "").splitlines() if l.strip()]
    mc = mean_conf if mean_conf is not None else 100

    if mc < 40:
        return None                                       # read too poor to extract anything reliably

    if key == "alcohol_content":
        # (?<!\d) so a 3-digit reading like "135%" is NOT misread as a partial "35%".
        for m in re.finditer(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%", _normalize_numbers(low)):
            v = float(m.group(1))
            if v > 100 and v % 100 != 0:                  # OCR dropped a decimal: 135% -> 13.5%
                v = v / 10.0
            if 4.0 <= v <= 70.0:                          # plausible ABV (not 100% grape, not a 2% misread)
                return f"{v:g}%"
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
