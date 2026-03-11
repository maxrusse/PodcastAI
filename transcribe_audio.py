#!/usr/bin/env python3
"""Generate transcript text from MP3/MP4 using the official PodcastAI hybrid workflow."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI


OUT_ROOT = Path("out_course")
INTERMEDIATE_DIR_NAME = "intermediate"
DEFAULT_TRANSCRIPT_PATH = Path("transcript.txt")

OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
OPENAI_TRANSCRIBE_COMPAT_MODEL = os.getenv("OPENAI_TRANSCRIBE_COMPAT_MODEL", "gpt-4o-transcribe")


def ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def resolve_ffmpeg_bin() -> str | None:
    for candidate in [
        os.path.expanduser("~/.local/bin/ffmpeg"),
        shutil.which("ffmpeg"),
        "/software/anaconda3/bin/ffmpeg",
    ]:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def split_audio_for_openai(audio_file: Path, chunk_seconds: int = 1200) -> list[Path]:
    ffmpeg_bin = resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found for long-audio chunking.")

    chunk_dir = Path(tempfile.mkdtemp(prefix="podcastai_audio_chunks_"))
    out_pattern = chunk_dir / "chunk_%03d.mp3"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(audio_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("ffmpeg chunking produced no audio chunks.")
    return chunks


def slug_mode(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-").lower()
    return cleaned or "default"


def default_run_mode() -> str:
    return f"transcribe-openai-{slug_mode(OPENAI_TRANSCRIBE_MODEL)}"


def run_dir(run_mode: str, run_timestamp: str) -> Path:
    return OUT_ROOT / run_mode / run_timestamp


def resolve_transcript_out(transcript_out: Path | None, current_run_dir: Path) -> Path:
    if transcript_out:
        return transcript_out
    return current_run_dir / INTERMEDIATE_DIR_NAME / DEFAULT_TRANSCRIPT_PATH


def resolve_diarized_out(diarized_out: Path | None, current_run_dir: Path) -> Path:
    if diarized_out:
        return diarized_out
    return current_run_dir / INTERMEDIATE_DIR_NAME / "diarized_transcript.json"


def write_latest_markers(current_run_dir: Path, transcript_out: Path, diarized_out: Path | None) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "LATEST_TRANSCRIBE_RUN.txt").write_text(str(current_run_dir), encoding="utf-8")
    if transcript_out.exists():
        (OUT_ROOT / "LATEST_TRANSCRIBE_TRANSCRIPT.txt").write_text(str(transcript_out), encoding="utf-8")
    if diarized_out and diarized_out.exists():
        (OUT_ROOT / "LATEST_TRANSCRIBE_DIARIZED.txt").write_text(str(diarized_out), encoding="utf-8")


def transcribe_openai(audio_path: Path, language: str | None, output_path: Path, diarized_output: Path | None) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    request_kwargs: dict[str, Any] = {
        "model": OPENAI_TRANSCRIBE_MODEL,
        "response_format": "json",
    }
    if language:
        request_kwargs["language"] = language

    def run_chunked_fallback(compat_kwargs: dict[str, Any]) -> dict[str, Any]:
        print("Long-audio fallback: chunking locally with ffmpeg for OpenAI transcription.")
        transcript_parts: list[str] = []
        for chunk_path in split_audio_for_openai(audio_path):
            with chunk_path.open("rb") as chunk_file:
                chunk_response = client.audio.transcriptions.create(file=chunk_file, **compat_kwargs)
            chunk_text = chunk_response if isinstance(chunk_response, str) else getattr(chunk_response, "text", "") or ""
            if chunk_text.strip():
                transcript_parts.append(chunk_text.strip())
        return {"text": "\n".join(transcript_parts).strip(), "segments": []}

    with audio_path.open("rb") as audio_file:
        try:
            response = client.audio.transcriptions.create(
                file=audio_file,
                chunking_strategy="auto",
                **request_kwargs,
            )
        except TypeError as exc:
            if "chunking_strategy" not in str(exc):
                raise
            audio_file.seek(0)
            try:
                response = client.audio.transcriptions.create(file=audio_file, **request_kwargs)
            except BadRequestError as bre:
                details = str(bre)
                requires_chunking = "chunking_strategy is required" in details
                uses_diarize_model = "diarize" in OPENAI_TRANSCRIBE_MODEL
                if "longer than" in details:
                    compat_kwargs: dict[str, Any] = {"model": OPENAI_TRANSCRIBE_COMPAT_MODEL}
                    if language:
                        compat_kwargs["language"] = language
                    response = run_chunked_fallback(compat_kwargs)
                    details = ""
                if details and not (requires_chunking and uses_diarize_model):
                    raise
                if details:
                    print(
                        "OpenAI SDK compatibility fallback: using "
                        f"{OPENAI_TRANSCRIBE_COMPAT_MODEL} without diarization."
                    )
                    audio_file.seek(0)
                    compat_kwargs = {"model": OPENAI_TRANSCRIBE_COMPAT_MODEL}
                    if language:
                        compat_kwargs["language"] = language
                    try:
                        response = client.audio.transcriptions.create(
                            file=audio_file,
                            chunking_strategy="auto",
                            **compat_kwargs,
                        )
                    except TypeError as compat_exc:
                        if "chunking_strategy" not in str(compat_exc):
                            raise
                        audio_file.seek(0)
                        try:
                            response = client.audio.transcriptions.create(file=audio_file, **compat_kwargs)
                        except BadRequestError as compat_bre:
                            if "longer than" not in str(compat_bre):
                                raise
                            response = run_chunked_fallback(compat_kwargs)
        except BadRequestError as bre:
            if "longer than" not in str(bre):
                raise
            compat_kwargs = {"model": OPENAI_TRANSCRIBE_COMPAT_MODEL}
            if language:
                compat_kwargs["language"] = language
            response = run_chunked_fallback(compat_kwargs)

    if isinstance(response, str):
        transcript_text = response
    elif isinstance(response, dict):
        transcript_text = str(response.get("text", "") or "")
    else:
        transcript_text = getattr(response, "text", "") or ""
    if not transcript_text:
        raise RuntimeError("OpenAI transcription returned empty text.")

    write_text(output_path, transcript_text)

    if diarized_output:
        if isinstance(response, dict):
            segments = response.get("segments", [])
            duration = response.get("duration")
        else:
            segments = getattr(response, "segments", None)
            duration = getattr(response, "duration", None)
        payload = {
            "duration": duration,
            "text": transcript_text,
            "segments": segments if segments is not None else [],
        }
        write_json(diarized_output, payload)

    print(f"Transcript written to: {output_path}")
    if diarized_output:
        print(f"Diarization JSON written to: {diarized_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-file", type=Path, required=True, help="Local MP3/MP4 path")
    parser.add_argument("--language", default="de", help="ISO-639-1 language hint (default: de)")
    parser.add_argument("--dry-run", action="store_true", help="Validate args/config and exit without API call")
    parser.add_argument("--run-mode", help="Optional run mode folder name under out_course")
    parser.add_argument("--run-timestamp", help="Optional run timestamp folder (default: UTC YYYYMMDD_HHMMSS)")
    parser.add_argument("--transcript-out", type=Path, help="Optional transcript output path")
    parser.add_argument("--diarized-json-out", type=Path, help="Optional diarization JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")

    current_run_mode = args.run_mode or default_run_mode()
    current_run_timestamp = args.run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    current_run_dir = run_dir(current_run_mode, current_run_timestamp)
    transcript_out = resolve_transcript_out(args.transcript_out, current_run_dir)
    diarized_out = resolve_diarized_out(args.diarized_json_out, current_run_dir)

    if args.dry_run:
        print(f"Resolved run directory: {current_run_dir}")
        print(f"Resolved transcript output: {transcript_out}")
        print(f"Resolved diarization output: {diarized_out}")
        print("Dry run successful: argument validation passed.")
        return

    transcribe_openai(
        audio_path=args.audio_file,
        language=args.language,
        output_path=transcript_out,
        diarized_output=diarized_out,
    )
    write_latest_markers(current_run_dir, transcript_out, diarized_out)


if __name__ == "__main__":
    main()
