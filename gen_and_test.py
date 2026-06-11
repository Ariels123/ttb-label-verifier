"""Generate a varied set of test labels + run the full verification matrix.

Covers more beverage types (wine, beer, imported scotch) and edge cases
(missing warning, proof-only ABV) so we know the engine holds up beyond bourbon.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ocr
import verifier

OUT = HERE / "samples"
EX = HERE / "examples"
OUT.mkdir(exist_ok=True)
EX.mkdir(exist_ok=True)
WARN = verifier.GOV_WARNING


def F(sz, bold=False):
    for p in ([f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
               "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap(d, text, fnt, w):
    out, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if d.textlength(t, font=fnt) <= w:
            cur = t
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def make(name, lines, warning):
    W, H = 820, 1040
    img = Image.new("RGB", (W, H), (243, 234, 214))
    d = ImageDraw.Draw(img)
    d.rectangle([18, 18, W - 18, H - 18], outline=(90, 74, 42), width=5)
    y = 120
    for text, size, bold in lines:
        d.text((W // 2, y), text, font=F(size, bold), fill=(58, 42, 20), anchor="mm")
        y += size + 26
    if warning:
        fnt = F(22, True)
        bx0, by0, bx1 = 60, 660, W - 60
        wl = wrap(d, warning, fnt, bx1 - bx0 - 24)
        d.rectangle([bx0, by0, bx1, by0 + len(wl) * 30 + 24], outline=(90, 74, 42), width=2)
        yy = by0 + 14
        for ln in wl:
            d.text((bx0 + 12, yy), ln, font=fnt, fill=(40, 30, 15))
            yy += 30
    img.save(OUT / name)
    img.save(EX / name)
    return OUT / name


BOURBON = {"brand_name": "Old Tom Distillery", "class_type": "Kentucky Straight Bourbon Whiskey",
           "alcohol_content": "45", "net_contents": "750 mL"}

# Existing samples (already on disk) + the fields to test them with + expected verdict.
EXISTING = [
    ("sample_correct.png", BOURBON, "PASS"),
    ("sample_bad_abv.png", BOURBON, "FAIL"),
    ("sample_bad_warning.png", BOURBON, "FAIL"),
    ("sample_sideways.png", BOURBON, "PASS"),
]

# New samples: (file, draw-lines, warning, fields, expected)
NEW = [
    ("sample_wine.png",
     [("Chateau Marengo", 48, True), ("Napa Valley Cabernet Sauvignon", 26, False),
      ("13.5% Alc./Vol.", 32, True), ("750 mL", 24, False)],
     WARN, {"brand_name": "Chateau Marengo", "class_type": "Napa Valley Cabernet Sauvignon",
            "alcohol_content": "13.5", "net_contents": "750 mL"}, "PASS"),
    ("sample_beer.png",
     [("Hop Harbor Brewing", 46, True), ("India Pale Ale", 28, False),
      ("6.2% Alc./Vol.", 32, True), ("12 FL OZ", 24, False)],
     WARN, {"brand_name": "Hop Harbor Brewing", "class_type": "India Pale Ale",
            "alcohol_content": "6.2", "net_contents": "12 FL OZ"}, "PASS"),
    ("sample_import.png",
     [("Highland Glen", 48, True), ("Blended Scotch Whisky", 26, False),
      ("43% Alc./Vol. (86 Proof)", 30, True), ("700 mL", 24, False), ("Product of Scotland", 22, False)],
     WARN, {"brand_name": "Highland Glen", "class_type": "Blended Scotch Whisky",
            "alcohol_content": "43", "net_contents": "700 mL", "origin": "Scotland"}, "PASS"),
    ("sample_no_warning.png",
     [("Old Tom Distillery", 48, True), ("Kentucky Straight Bourbon Whiskey", 24, False),
      ("45% Alc./Vol.", 32, True), ("750 mL", 24, False)],
     None, {"brand_name": "Old Tom Distillery", "alcohol_content": "45"}, "FAIL"),
    ("sample_proof_only.png",
     [("Cask Strength Co", 48, True), ("Straight Rye Whiskey", 26, False),
      ("100 Proof", 36, True), ("750 mL", 24, False)],
     WARN, {"brand_name": "Cask Strength Co", "alcohol_content": "50"}, "PASS"),
]

print("=== generating new samples ===")
for f, lines, warn, fields, exp in NEW:
    make(f, lines, warn)
    print("  wrote", f)

print("\n=== FULL TEST MATRIX (ocr + verify) ===")
ok_all = True
for fname, fields, exp in EXISTING + [(f, fld, e) for f, _, _, fld, e in NEW]:
    txt = ocr.extract_text(str(OUT / fname))
    r = verifier.verify(fields, txt)
    fails = [x["field"] for x in r["results"] if x["status"] == "fail"]
    if not r["passed"]:
        actual = "FAIL"
    elif r.get("provided", 0) == 0:
        actual = "INCOMPLETE"
    else:
        actual = "PASS"
    ok = "OK " if actual == exp else "XX "
    if actual != exp:
        ok_all = False
    print(f"  {ok}{fname:26} -> {actual:10} (exp {exp:5})  fails={fails or '-'}  ({r['elapsed_ms']}ms)")
print("\nALL MATCH EXPECTED" if ok_all else "\n!! SOME MISMATCHED — review above")
