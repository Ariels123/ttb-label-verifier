"""Unit tests for verifier.py — the compliance-critical matching logic.

These lock in the rules from the brief's interviews and the bugs surfaced in review:
the strict Government Warning (exact + ALL-CAPS heading, line-wrap tolerant), numeric ABV
(proof handling, no 3-digit truncation), and net contents by normalized volume.

Run:  python3 -m pytest tests/        (or plain:  python3 tests/test_verifier.py)
"""
import os
import sys

# verifier.py lives in the repo's code/ folder (sibling of this tests/ folder), so add
# <repo-root>/code to the import path before importing it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
import verifier as v  # noqa: E402

WARN = ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
        "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems.")


def test_warning_exact_passes():
    assert v.check_warning(WARN)[0] == v.PASS


def test_warning_titlecase_header_fails():
    assert v.check_warning(WARN.replace("GOVERNMENT WARNING:", "Government Warning:"))[0] == v.FAIL


def test_warning_line_wrapped_header_passes():
    # OCR often wraps the heading across lines — a valid label must still pass.
    assert v.check_warning(WARN.replace("GOVERNMENT WARNING:", "GOVERNMENT\nWARNING:"))[0] == v.PASS


def test_warning_negation_flip_fails():
    assert v.check_warning(WARN.replace("should not drink", "should drink"))[0] == v.FAIL


def test_warning_missing_fails():
    assert v.check_warning("Just a nice bourbon label")[0] == v.FAIL


def test_abv_percent_passes():
    assert v.check_abv("45", "45% Alc./Vol.")[0] == v.PASS


def test_abv_proof_resolves():
    assert v.check_abv("45", "90 Proof")[0] == v.PASS


def test_abv_three_digit_not_truncated():
    assert v._abv("100") == 100.0
    assert v._abv("151") == 151.0


def test_abv_prefers_percent_over_proof():
    assert v._abv("45% Alc./Vol. (90 Proof)") == 45.0


def test_abv_mismatch_fails():
    assert v.check_abv("45", "40% Alc./Vol.")[0] == v.FAIL


def test_quantity_equal_volumes_pass():
    assert v.check_quantity("750 mL", "net 75 cL")[0] == v.PASS
    assert v.check_quantity("1 L", "1000 mL")[0] == v.PASS


def test_quantity_distinct_volumes_fail():
    assert v.check_quantity("750 mL", "1750 mL")[0] == v.FAIL
    assert v.check_quantity("70 mL", "700 mL")[0] == v.FAIL


def test_text_forgiving_case_and_punctuation():
    assert v.check_text("Stone's Throw", "STONES THROW DISTILLERY")[0] == v.PASS


def test_text_absent_fails():
    assert v.check_text("Acme Vodka", "totally different label")[0] == v.FAIL


def test_verify_unreadable_flag():
    assert v.verify({}, "")["unreadable"] is True
    assert v.verify({"brand_name": "X"}, WARN + " Acme Brand Whiskey 45%")["unreadable"] is False


def test_verify_skips_unentered_fields():
    r = v.verify({"alcohol_content": "45"}, "Acme 45% " + WARN)
    statuses = {x["field"]: x["status"] for x in r["results"]}
    assert statuses["Brand name"] == v.SKIP  # not entered -> skipped, never failed


def test_extraction_heuristics():
    ocr = "Old Tom Distillery \n 750 mL \n 45% Alc/Vol \n Product of USA \n " + WARN
    r = v.verify({}, ocr)
    res = {x["field"]: x["detected"] for x in r["results"]}
    assert res["Brand name"] == "Old Tom Distillery"
    assert "45%" in res["Alcohol content"]
    assert "750 ml" in res["Net contents"].lower()
    assert "product of usa" in res["Country of origin"].lower()


def test_foreign_warning_is_flagged_not_missing():
    uk = "PLEASE DRINK RESPONSIBLY: UK Chief Medical Officers recommend adults do not exceed limits."
    r = v.verify({}, "Highland Glen 40% Alc/Vol " + uk, words=[("x", 90)], mean_conf=90)
    w = next(x for x in r["results"] if x["field"] == "Government Warning")
    assert w["status"] == v.FAIL and w["detected"] == "Non-US warning only"


def test_extraction_ignores_implausible_abv():
    # "100% uve Dolcetto" is grape content, not alcohol -> use the real 13.5% instead
    assert v._extract_detected("alcohol_content", "Vino rosso 100% uve Dolcetto 13.5% vol", mean_conf=90) == "13.5%"


def test_extraction_ignores_implausible_net_contents():
    assert v._extract_detected("net_contents", "contents 70 l net", mean_conf=90) is None  # 70 L is not a bottle
    assert v._extract_detected("net_contents", "750 ml", mean_conf=90) == "750 ml"


def test_extraction_is_silent_on_low_confidence():
    assert v._extract_detected("brand_name", "Xqz Wbll\nfoo bar", mean_conf=30) is None  # poor read -> no guess


def test_extraction_brand_skips_warning_lines():
    assert v._extract_detected("brand_name", "SMOKING AND ALCOHOL CONSUMPTION IS INJURIOUS\nAcme Distillery", mean_conf=90) == "Acme Distillery"


def test_extraction_recovers_dropped_decimal_abv():
    # OCR sometimes drops the decimal: "13.5% vol" reads as "135% vol". Restore to 13.5%,
    # and never surface a partial "35%".
    assert v._extract_detected("alcohol_content", "135% vol", mean_conf=90) == "13.5%"
    assert v._extract_detected("alcohol_content", "alc 135%", mean_conf=90) == "13.5%"


def test_extraction_keeps_round_hundred_as_content_not_abv():
    # "100% agave" is a content claim, not a dropped-decimal ABV -> ignore it, use the real 40%.
    assert v._extract_detected("alcohol_content", "100% agave 40% alc/vol", mean_conf=90) == "40%"


def test_abv_check_tolerates_dropped_decimal():
    # application says 13.5; label OCR'd as "135% vol" (decimal dropped) still PASSes...
    assert v.check_abv("13.5", "135% vol")[0] == v.PASS
    # ...but a genuine mismatch still FAILs.
    assert v.check_abv("13.5", "145% vol")[0] == v.FAIL


if __name__ == "__main__":
    tests = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
            print("ok  ", fn.__name__)
        except AssertionError:
            print("FAIL", fn.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
