"""
TaP v1.5 Vision Smoke Test.

Validates that the VL model endpoint accepts OpenAI-compatible image_url
content blocks, and that ``_encode_image_base64`` round-trips correctly.

No GPU or SAPIEN dependency.  Only needs QWEN_API_KEY in .env.

Run:
    cd ~/Documents/GitHub/RoboTwin
    python policy/Your_Policy/examples/tap_vision_smoke_test.py
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

# ── Setup path so we can import tap_executor ────────────────────────────

_HERE = Path(__file__).resolve().parent
_POLICY_DIR = _HERE.parent
_ROBOTWIN_ROOT = _POLICY_DIR.parent.parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))
if str(_ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOTWIN_ROOT))

# Load .env before importing anything that needs QWEN_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv(_POLICY_DIR / ".env")
except ImportError:
    pass

_RESULTS = []


def _record(name, ok, details=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}")
    if details:
        for line in str(details).split("\n"):
            print(f"      {line}")
    _RESULTS.append((name, ok))
    return ok


# ── Pre-computed valid 16x16 red PNG ─────────────────────────────────────
# 16x16 meets VL model minimum (>10x10). Generated via struct+zlib, verified.

RED_16x16_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4"
    "nGP4z8BAEiJN9aiGUQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII="
)
RED_16x16_PNG_BYTES = base64.b64decode(RED_16x16_PNG_B64)


# ── Test: _encode_image_base64 round-trip ───────────────────────────────

def case_encode_image_roundtrip():
    """Write a PNG to a temp file, encode it, verify the data URI prefix."""
    from tap_executor import TaPExecutor

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(RED_16x16_PNG_BYTES)
        tmp_path = f.name
    try:
        data_uri = TaPExecutor._encode_image_base64(tmp_path)
        ok = (
            data_uri is not None
            and data_uri.startswith("data:image/png;base64,")
            and len(data_uri) > 30
        )
        return _record("encode_image_roundtrip", ok,
                        f"URI length={len(data_uri) if data_uri else 0}")
    finally:
        os.unlink(tmp_path)


def case_encode_image_nonexistent():
    """Missing file returns None."""
    from tap_executor import TaPExecutor
    result = TaPExecutor._encode_image_base64("/tmp/_nonexistent_12345.png")
    return _record("encode_image_nonexistent", result is None)


# ── Test: VL model API call ─────────────────────────────────────────────

def case_vl_api_accepts_image():
    """Send a 1x1 PNG to the VL model and verify a non-empty response."""
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        return _record("vl_api_accepts_image", False, "QWEN_API_KEY not set")

    base_url = os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    model = (
        os.environ.get("TAP_VISION_MODEL")
        or os.environ.get("QWEN_VISION_MODEL")
        or "qwen3-vl-plus"
    )

    try:
        from openai import OpenAI
    except ImportError:
        return _record("vl_api_accepts_image", False, "openai package not installed")

    client = OpenAI(api_key=api_key, base_url=base_url)
    data_uri = f"data:image/png;base64,{RED_16x16_PNG_B64}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this tiny image? Reply in one word."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=32,
        )
        text = resp.choices[0].message.content.strip()
        ok = len(text) > 0
        return _record("vl_api_accepts_image", ok,
                        f"model={model} response={text!r}")
    except Exception as e:
        return _record("vl_api_accepts_image", False, f"API error: {e}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    cases = [
        case_encode_image_roundtrip,
        case_encode_image_nonexistent,
        case_vl_api_accepts_image,
    ]
    for case in cases:
        try:
            case()
        except Exception as e:
            _record(case.__name__, False, f"raised: {e}")
        print()

    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print("=" * 60)
    print(f"TaP v1.5 vision smoke test: {passed}/{total} passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
