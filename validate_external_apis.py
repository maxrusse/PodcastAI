#!/usr/bin/env python3
"""Smoke-test external API connectivity for PodcastAI workflows."""

import argparse
import os
import sys
from dataclasses import dataclass

from google import genai
from openai import OpenAI


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def check_openai() -> CheckResult:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return CheckResult("openai", "skipped", "OPENAI_API_KEY is not set")

    try:
        client = OpenAI(api_key=key)
        response = client.responses.create(
            model="gpt-5.2-mini",
            input="Reply with exactly: ok",
            max_output_tokens=16,
        )
        text = (response.output_text or "").strip().lower()
        if "ok" not in text:
            return CheckResult("openai", "failed", f"unexpected output: {text!r}")
        return CheckResult("openai", "passed", "text generation call succeeded")
    except Exception as exc:
        return CheckResult("openai", "failed", str(exc))


def check_gemini() -> CheckResult:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return CheckResult("gemini", "skipped", "GEMINI_API_KEY is not set")

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with exactly: ok",
            config={"max_output_tokens": 16},
        )
        text = (response.text or "").strip().lower()
        if "ok" not in text:
            return CheckResult("gemini", "failed", f"unexpected output: {text!r}")
        return CheckResult("gemini", "passed", "text generation call succeeded")
    except Exception as exc:
        return CheckResult("gemini", "failed", str(exc))


def print_result(result: CheckResult) -> None:
    symbols = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}
    print(f"[{symbols[result.status]}] {result.name}: {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "all"],
        default="all",
        help="Provider to validate",
    )
    args = parser.parse_args()

    checks = []
    if args.provider in {"openai", "all"}:
        checks.append(check_openai())
    if args.provider in {"gemini", "all"}:
        checks.append(check_gemini())

    for result in checks:
        print_result(result)

    if any(r.status == "failed" for r in checks):
        return 1
    if all(r.status == "skipped" for r in checks):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
