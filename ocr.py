"""Local OCR for label images — Tesseract binary via subprocess.

Why shell out instead of pytesseract: the binary needs no Python ML stack (no
pandas/NumPy), so it stays fast, dependency-light, and trivial to install in a
container (apt-get install tesseract-ocr). Everything runs ON THE BOX — no cloud
vision API — which is the whole point given TTB's outbound-firewall constraint
and the 5-second SLA.

Hard labels escalate through a ladder: best-of-PSM Tesseract first, then OpenCV-
preprocessed variants (CLAHE / adaptive threshold / Otsu / deskew), then an optional
heavier model — so clean labels stay fast and only difficult ones pay the extra cost.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import time

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 64_000_000  # decompression-bomb guard (~64 MP); enforced because we
# open every upload through PIL in _working_image (PIL raises past 2x this on a pixel bomb).

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except Exception:  # OpenCV is optional — plain Tesseract still works without it
    HAVE_CV2 = False

# Detect PaddleOCR WITHOUT importing it — its import pulls in a heavy dependency tree
# (and can be slow even when it ultimately fails). The real import happens lazily in
# _paddle_ocr, so `import ocr` stays fast whether or not Paddle is installed. Set
# TTB_DISABLE_PADDLE=1 to force it off (small box, or to match the no-Paddle production
# image during local testing).
_PADDLE_OFF = os.environ.get("TTB_DISABLE_PADDLE", "").lower() in ("1", "true", "yes")
HAVE_PADDLE = (importlib.util.find_spec("paddleocr") is not None) and not _PADDLE_OFF

_PADDLE = None

TESSERACT = shutil.which("tesseract") or "tesseract"


def _run(path: str, timeout: int, psm: int = 3) -> str:
    """Run Tesseract on a file path; return decoded stdout (best-effort)."""
    try:
        # --psm 3: fully automatic segmentation.
        # --psm 6: assume a single uniform block of text.
        proc = subprocess.run(
            [TESSERACT, path, "stdout", "--psm", str(psm)],
            capture_output=True, timeout=timeout,
        )
        return (proc.stdout or b"").decode("utf-8", "replace")
    except Exception:
        return ""


def _score(text: str) -> int:
    """Quality proxy: count of word-like tokens (3+ letters). Distinguishes real
    text from OCR garbage far better than raw character count (garbage has lots of
    characters but few real words)."""
    return len(re.findall(r"[A-Za-z]{3,}", text or ""))


def _run_tsv(path: str, timeout, psm: int = 3):
    """Like _run but with Tesseract's TSV output, so we get PER-WORD confidence.
    Returns (reconstructed_text, [(word, conf 0-100), ...]). Confidence drives the
    read-quality signal and the per-field 'can we trust this read' display."""
    try:
        proc = subprocess.run(
            [TESSERACT, path, "stdout", "--psm", str(psm), "tsv"],
            capture_output=True, timeout=timeout,
        )
        out = (proc.stdout or b"").decode("utf-8", "replace")
    except Exception:
        return "", []
    words, lines, cur, buf = [], [], None, []
    for ln in out.splitlines()[1:]:           # skip the TSV header row
        c = ln.split("\t")
        if len(c) < 12 or not c[11].strip():
            continue
        key = (c[2], c[3], c[4])              # block, paragraph, line
        if cur is not None and key != cur:
            lines.append(" ".join(buf)); buf = []
        cur = key
        buf.append(c[11])
        try:
            conf = float(c[10])
        except ValueError:
            conf = -1.0
        if conf >= 0:
            words.append((c[11], conf))
    if buf:
        lines.append(" ".join(buf))
    return "\n".join(lines), words


def _mean_conf(words) -> float:
    """Median confidence over real (multi-letter) words — robust to a few junk tokens.
    0 when nothing legible was read."""
    cs = sorted(c for w, c in words if c >= 0 and len(re.findall(r"[A-Za-z]", w)) >= 2)
    return float(cs[len(cs) // 2]) if cs else 0.0


def _deskew(binary):
    """Rotate a binarized image so its text is horizontal (skew from text pixels).
    Returns None if there is no meaningful, plausible skew."""
    try:
        coords = np.column_stack(np.where(cv2.bitwise_not(binary) > 0))
        if len(coords) < 80:
            return None
        angle = cv2.minAreaRect(coords)[-1]
        angle = (90 + angle) if angle < -45 else angle
        if abs(angle) < 0.7 or abs(angle) > 30:
            return None
        h, w = binary.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return None


def _cv_variants(path: str):
    """OpenCV-preprocessed grayscale variants for a hard image: CLAHE contrast,
    adaptive threshold, Otsu binarization, plus a deskewed copy. Cheap CPU ops.
    Returns [] if OpenCV is unavailable or the file cannot be read."""
    if not HAVE_CV2:
        return []
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 1800:                                  # upscale small text
        s = 1800.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants = [clahe, adaptive, otsu]
    desk = _deskew(otsu)
    if desk is not None:
        variants.append(desk)
    return variants


def _paddle_ocr(path: str) -> str:
    """Heavy but accurate local fallback (PaddleOCR, deep-learning) for ornate
    labels Tesseract cannot read. Lazy-loaded + cached; "" if unavailable/on error."""
    global _PADDLE
    if not HAVE_PADDLE:
        return ""
    try:
        if _PADDLE is None:
            from paddleocr import PaddleOCR  # heavy import, only when Tier 3 is reached
            _PADDLE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        res = _PADDLE.ocr(path, cls=True)
        out = []
        for page in (res or []):
            for line in (page or []):
                try:
                    out.append(line[1][0])
                except Exception:
                    pass
        return "\n".join(out)
    except Exception:
        return ""


def _read_useful(text: str) -> bool:
    """Have we read enough to verify the KEY fields, or should we escalate to the OpenCV
    pass? We require a real ABV NUMBER (or the Government Warning) — not merely the word
    'alcohol'. A read full of body text but missing the ABV/brand (e.g. a glary or curved
    title, like the wm02 DOGLIANI label) should still escalate. (Codex's first-consult
    point: gate on field usefulness, not raw text volume.)"""
    low = (text or "").lower()
    if "government" in low and "warning" in low:
        return True
    has_abv = bool(re.search(r"\b\d{1,2}(?:\.\d)?\s*%", low) or re.search(r"\bproof\b", low))
    return has_abv and _score(text) >= 15


def _union(texts) -> str:
    """Merge several OCR reads into one, de-duplicating by normalized line. Different
    page-segmentation modes catch different regions — PSM 3 the body paragraphs, PSM 11
    a big stylized title — so the UNION covers fields that a single best-scoring pass
    drops. (This is exactly why the in-browser 2-PSM union out-read the old best-of-PSM
    server path on labels like DOGLIANI, where the brand title sits above the body.)"""
    seen, out = set(), []
    for t in texts:
        for line in (t or "").splitlines():
            key = re.sub(r"[^a-z0-9]", "", line.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(line.rstrip())
    return "\n".join(out)


def _working_image(path: str):
    """Open every upload through PIL FIRST: this enforces the decompression-bomb guard (the
    rest of the pipeline shells out to Tesseract/OpenCV, which don't), applies EXIF orientation
    so portrait phone photos aren't read sideways (Jenny's "weird angles"), and downscales very
    large shots (~2600px is plenty for label text and keeps OCR within the ~5s budget).
    Returns (path_to_use, tmp_to_clean); (None, None) rejects a pixel bomb; falls back to the
    original path on any other error."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)            # honor phone-photo rotation
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 2600:
            s = 2600.0 / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        fd, tp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tp)
        return tp, tp
    except Image.DecompressionBombError:
        return None, None                             # reject pixel bombs — never hand them to OpenCV/Tesseract
    except Exception:
        return path, None


def read_label(path: str, timeout: int = 3) -> dict:
    """Read a label image, returning {text, words, mean_conf}.

    The PRIMARY PSM-3 pass uses Tesseract TSV so we capture per-word CONFIDENCE; PSM 11
    (sparse) catches stylized titles and is unioned in. We escalate to OpenCV variants —
    then, if installed, a heavier model (PaddleOCR) — when the read is poor OR
    low-confidence (mean_conf gates the deep tier, per the Gemini/Codex consults).

    SPEED-FIRST (Sarah: ~5s or nobody uses it): tightly time-bounded; a pixel bomb or
    unreadable input returns an empty read so the UI can show "couldn't read clearly".
    """
    budget = 3.5
    t0 = time.time()
    work, work_tmp = _working_image(path)
    if work is None:
        return {"text": "", "words": [], "mean_conf": 0.0}
    rem = lambda: max(0.4, budget - (time.time() - t0))  # cap each pass so the chain can't blow ~5s
    try:
        text3, words = _run_tsv(work, min(timeout, rem()), 3)   # text + per-word confidence
        parts = [text3] if _score(text3) >= 1 else []
        mean_conf = _mean_conf(words)
        if (time.time() - t0) < budget:
            t11 = _run(work, min(timeout, rem()), 11)           # sparse: stylized titles
            if _score(t11) >= 1:
                parts.append(t11)
        best = _union(parts)
        # A poor OR low-confidence read earns the OpenCV preprocessing passes.
        if (not _read_useful(best) or mean_conf < 60) and (time.time() - t0) < budget:
            t6 = _run(work, min(timeout, rem()), 6)
            if _score(t6) >= 1:
                parts.append(t6)
            for arr in _cv_variants(work)[:2]:
                if (time.time() - t0) >= budget:
                    break
                fd, vp = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                try:
                    cv2.imwrite(vp, arr)
                    for psm in (6, 11):
                        if (time.time() - t0) >= budget:
                            break
                        t = _run(vp, min(timeout, rem()), psm)
                        if _score(t) >= 1:
                            parts.append(t)
                finally:
                    os.unlink(vp)
            best = _union(parts)
        # Deep model for ornate labels Tesseract + OpenCV still can't read — gated on a
        # poor/low-confidence read AND availability (off by default on a small box).
        if (not _read_useful(best) or mean_conf < 50) and HAVE_PADDLE:
            pt = _paddle_ocr(work)
            if _score(pt) > _score(best):
                best = pt
        return {"text": best, "words": words, "mean_conf": mean_conf}
    finally:
        if work_tmp:
            try:
                os.unlink(work_tmp)
            except OSError:
                pass


def extract_text(path: str, timeout: int = 3) -> str:
    """Back-compat wrapper: just the unioned text (for callers that don't need confidence)."""
    return read_label(path, timeout)["text"]
