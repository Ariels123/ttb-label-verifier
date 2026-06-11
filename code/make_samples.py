"""Generate synthetic TTB label images for testing — one correct, two deliberately broken.

Saves into sample_images/ (the labels the app's "Try an example" links serve, which also holds
the real-world test photos). Lets you exercise the verifier end-to-end without hunting for real
label photos. The brief even suggests AI-generated test labels; these are clean synthetic ones
drawn with Pillow. (See gen_and_test.py for the fuller set + the OCR/verify self-test.)
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent.parent / "sample_images"  # at the repo root, above this code/ folder
OUT.mkdir(exist_ok=True)

WARNING_CAPS = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. (2) "
    "Consumption of alcoholic beverages impairs your ability to drive a car or operate "
    "machinery, and may cause health problems."
)


def font(size: int, bold: bool = False):
    cands = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
             if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"])
    for p in cands:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    out, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def make(name, brand, cls, abv_line, net, warning):
    W, H = 820, 1040
    img = Image.new("RGB", (W, H), (243, 234, 214))
    d = ImageDraw.Draw(img)
    d.rectangle([18, 18, W - 18, H - 18], outline=(90, 74, 42), width=5)
    cx = W // 2
    d.text((cx, 120), brand.upper(), font=font(52, True), fill=(58, 42, 20), anchor="mm")
    d.text((cx, 188), cls, font=font(28), fill=(80, 60, 30), anchor="mm")
    d.text((cx, 330), abv_line, font=font(34, True), fill=(58, 42, 20), anchor="mm")
    d.text((cx, 392), net, font=font(26), fill=(80, 60, 30), anchor="mm")
    fnt = font(22, True)
    bx0, by0, bx1 = 60, 640, W - 60
    lines = wrap(d, warning, fnt, bx1 - bx0 - 24)
    d.rectangle([bx0, by0, bx1, by0 + len(lines) * 30 + 24], outline=(90, 74, 42), width=2)
    y = by0 + 14
    for ln in lines:
        d.text((bx0 + 12, y), ln, font=fnt, fill=(40, 30, 15))
        y += 30
    img.save(OUT / name)
    print("wrote", OUT / name)


make("sample_correct.png", "Old Tom Distillery", "Kentucky Straight Bourbon Whiskey",
     "45% Alc./Vol. (90 Proof)", "750 mL", WARNING_CAPS)
make("sample_bad_abv.png", "Old Tom Distillery", "Kentucky Straight Bourbon Whiskey",
     "40% Alc./Vol. (80 Proof)", "750 mL", WARNING_CAPS)
make("sample_bad_warning.png", "Old Tom Distillery", "Kentucky Straight Bourbon Whiskey",
     "45% Alc./Vol. (90 Proof)", "750 mL",
     WARNING_CAPS.replace("GOVERNMENT WARNING:", "Government Warning:"))
