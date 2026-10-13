#!/usr/bin/env python3
"""Test script to verify COMIX OCR functionality."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw, ImageFont
from comix.comix_pipeline import RecognitionOCREngine


def create_test_image(text: str = "Hello World") -> Image.Image:
    """Create a simple test image with text."""
    img = Image.new("RGB", (200, 50), color="white")
    draw = ImageDraw.Draw(img)
    # Use default font
    draw.text((10, 10), text, fill="black")
    return img


def main():
    # Create mock args
    args = argparse.Namespace(
        ocr_lang="en",
        ocr_version="PP-OCRv6",
        ocr_device="cpu",
        ocr_use_angle_cls=False,
        ocr_enable_pir=False,
        ocr_enable_mkldnn=False,
    )

    print("Initializing OCR engine...")
    try:
        ocr_engine = RecognitionOCREngine(args)
        print("OCR engine initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize OCR engine: {e}")
        return 1

    # Test with a simple image
    print("\nCreating test image with text...")
    test_img = create_test_image("HELLO")

    print("Running OCR...")
    try:
        result = ocr_engine.recognize(test_img, context="test")
        print(f"OCR Result: '{result}'")

        if result.strip():
            print("\n✓ OCR is working! Successfully recognized text.")
            return 0
        else:
            print("\n✗ OCR returned empty result. This indicates the bug is still present.")
            return 1
    except Exception as e:
        print(f"\n✗ OCR failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        ocr_engine.close()


if __name__ == "__main__":
    sys.exit(main())
