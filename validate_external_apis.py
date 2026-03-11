#!/usr/bin/env python3
"""Smoke-test the official PodcastAI hybrid API stack."""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

from google import genai
from openai import OpenAI


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def check_openai(model: Optional[str]) -> CheckResult:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return CheckResult("openai", "skipped", "OPENAI_API_KEY is not set")

    selected_model = model or os.getenv("OPENAI_TEXT_MODEL") or "gpt-5.4"
    try:
        client = OpenAI(api_key=key)
        response = client.responses.create(
            model=selected_model,
            input="Reply with exactly: ok",
            max_output_tokens=16,
        )
        text = (response.output_text or "").strip()
        if text:
            return CheckResult("openai", "passed", f"model={selected_model} text generation call succeeded")
        return CheckResult("openai", "passed", f"model={selected_model} call succeeded (empty text)")
    except Exception as exc:
        return CheckResult("openai", "failed", f"{selected_model}: {exc}")


def check_banana(model: Optional[str]) -> CheckResult:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return CheckResult("banana", "skipped", "GEMINI_API_KEY is not set")

    selected_model = model or os.getenv("BANANA_IMAGE_MODEL") or "nano-banana-pro-preview"
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=selected_model,
            contents=[
                "Saubere weisse Infografik mit einem einfachen blauen Kreis, ohne Text, ohne medizinische Bildaesthetik."
            ],
            config={"image_config": {"aspect_ratio": "1:1", "image_size": "1K"}},
        )
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) if content is not None else []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    return CheckResult("banana", "passed", f"model={selected_model} image generation call succeeded")
        return CheckResult("banana", "failed", f"{selected_model}: no inline image data returned")
    except Exception as exc:
        return CheckResult("banana", "failed", f"{selected_model}: {exc}")


def print_result(result: CheckResult) -> None:
    symbols = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}
    print(f"[{symbols[result.status]}] {result.name}: {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "banana", "all"], default="all")
    parser.add_argument("--openai-model", default=None, help="Optional OpenAI model override (default gpt-5.4)")
    parser.add_argument("--banana-model", default=None, help="Optional Banana image model override")
    args = parser.parse_args()

    checks = []
    if args.provider in {"openai", "all"}:
        checks.append(check_openai(args.openai_model))
    if args.provider in {"banana", "all"}:
        checks.append(check_banana(args.banana_model))

    for result in checks:
        print_result(result)

    if any(r.status == "failed" for r in checks):
        return 1
    if all(r.status == "skipped" for r in checks):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
