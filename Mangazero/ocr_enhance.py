"""Enhanced OCR helpers for MangaZero/COMIX panel text extraction.

Implements the comic-OCR improvement recipe for data that already has text
box coordinates:

1. crop with padding to avoid clipped first/last characters;
2. optionally upscale small crops before recognition;
3. optionally try rotated variants for vertical or rotated text;
4. optionally fuse one or more OCR engines by choosing the best candidate.
"""

from __future__ import annotations

from typing import Callable

from PIL import Image


def crop_with_padding(
    page_image: Image.Image,
    bbox: list[int] | tuple[int, int, int, int],
    padding_ratio: float = 0.05,
    min_padding: int = 2,
) -> Image.Image:
    """Crop a text bbox with extra surrounding context."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(min_padding, int(round(width * padding_ratio)))
    pad_y = max(min_padding, int(round(height * padding_ratio)))
    page_w, page_h = page_image.size
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(page_w, x2 + pad_x)
    bottom = min(page_h, y2 + pad_y)
    if right <= left or bottom <= top:
        return page_image.crop((x1, y1, x2, y2)).convert("RGB")
    return page_image.crop((left, top, right, bottom)).convert("RGB")


def upscale_crop(image: Image.Image, scale: float = 2.0) -> Image.Image:
    """Resize while preserving aspect ratio; useful for small text crops."""
    if scale <= 1.0:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def image_variants(
    image: Image.Image,
    with_upscale: bool = False,
    with_rotations: bool = False,
    upscale_scale: float = 2.0,
) -> list[Image.Image]:
    """Return one or more recognition variants for a text crop."""
    variants = [image]
    if with_upscale:
        variants.append(upscale_crop(image, scale=upscale_scale))
    if with_rotations:
        variants.append(image.rotate(90, expand=True))
        variants.append(image.rotate(-90, expand=True))
    return variants


def best_text(candidates: list[str]) -> str:
    """Pick the most complete non-empty OCR line."""
    cleaned = [text.strip() for text in candidates if text and text.strip()]
    if not cleaned:
        return ""
    return max(cleaned, key=lambda text: (len(text), text))


def run_ocr_variants(
    image: Image.Image,
    recognizers: list[Callable[[Image.Image], str]],
    with_upscale: bool = False,
    with_rotations: bool = False,
    upscale_scale: float = 2.0,
) -> str:
    """Run every recognizer over every image variant and return the best text."""
    candidates: list[str] = []
    for variant in image_variants(
        image,
        with_upscale=with_upscale,
        with_rotations=with_rotations,
        upscale_scale=upscale_scale,
    ):
        for recognizer in recognizers:
            try:
                text = recognizer(variant)
            except Exception:
                text = ""
            if text and text.strip():
                candidates.append(text.strip())
    return best_text(candidates)


class MangaOCRRecognizer:
    """Optional extra recognizer using kha-white/manga-ocr."""

    def __init__(self) -> None:
        try:
            from manga_ocr import MangaOcr
        except ImportError as exc:
            raise ImportError("manga-ocr is not installed; cannot use --ocr-extra-engines manga-ocr") from exc
        self.model = MangaOcr()

    def __call__(self, image: Image.Image) -> str:
        return str(self.model(image)).strip()


class EasyOCRRecognizer:
    """Optional extra recognizer using easyocr."""

    def __init__(self, langs: tuple[str, ...] = ("ja", "en")) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError("easyocr is not installed; cannot use --ocr-extra-engines easyocr") from exc
        self.reader = easyocr.Reader(list(langs))

    def __call__(self, image: Image.Image) -> str:
        result = self.reader.readtext(image, detail=0, paragraph=False)
        return " ".join(str(text).strip() for text in result if text and str(text).strip())


class TesseractRecognizer:
    """Optional extra recognizer using pytesseract."""

    def __init__(self, lang: str = "eng") -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError("pytesseract is not installed; cannot use --ocr-extra-engines tesseract") from exc
        self.lang = lang
        self.pytesseract = pytesseract

    def __call__(self, image: Image.Image) -> str:
        return self.pytesseract.image_to_string(image, lang=self.lang, config="--psm 7").strip()


def build_extra_recognizers(names: str | list[str]) -> list[Callable[[Image.Image], str]]:
    """Build only the requested optional recognizers; skip unavailable ones."""
    if isinstance(names, str):
        raw = [name.strip() for name in names.split(",") if name.strip()]
    else:
        raw = [str(name).strip() for name in names if str(name).strip()]
    recognizers: list[Callable[[Image.Image], str]] = []
    for name in raw:
        lowered = name.lower()
        try:
            if lowered in ("manga-ocr", "manga_ocr", "manga"):
                recognizers.append(MangaOCRRecognizer())
            elif lowered in ("easyocr", "easy_ocr"):
                recognizers.append(EasyOCRRecognizer())
            elif lowered in ("tesseract", "pytesseract"):
                recognizers.append(TesseractRecognizer())
        except ImportError as exc:
            print(f"WARN: skip OCR engine {name}: {exc}", flush=True)
    return recognizers