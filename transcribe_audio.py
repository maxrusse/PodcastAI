#!/usr/bin/env python3
"""Generate transcript text from MP3/MP4 using OpenAI or Google Speech-to-Text v2."""

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from google.api_core.client_options import ClientOptions
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from openai import OpenAI


DEFAULT_TRANSCRIPT_PATH = Path("transcript.txt")


def ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def transcribe_openai(audio_path: Path, language: str | None, output_path: Path, diarized_output: Path | None) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI()

    request_kwargs: dict[str, Any] = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
    }
    if language:
        request_kwargs["language"] = language

    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(file=audio_file, **request_kwargs)

    transcript_text = getattr(response, "text", "") or ""
    if not transcript_text:
        raise RuntimeError("OpenAI transcription returned empty text.")

    write_text(output_path, transcript_text)

    if diarized_output:
        segments = getattr(response, "segments", None)
        payload = {
            "duration": getattr(response, "duration", None),
            "text": transcript_text,
            "segments": segments if segments is not None else [],
        }
        write_json(diarized_output, payload)

    print(f"OpenAI transcript written to: {output_path}")
    if diarized_output:
        print(f"OpenAI diarization JSON written to: {diarized_output}")


def transcribe_google(
    audio_gcs_uri: str,
    source_label: str,
    output_gcs_prefix: str,
    region: str,
    language_code: str,
    operation_timeout_s: int,
    response_manifest_path: Path | None,
) -> None:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise EnvironmentError("GOOGLE_CLOUD_PROJECT is not set.")

    endpoint = f"{region}-speech.googleapis.com"
    client = SpeechClient(client_options=ClientOptions(api_endpoint=endpoint))

    recognizer = f"projects/{project_id}/locations/{region}/recognizers/_"
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        model="chirp_3",
        language_codes=[language_code],
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            diarization_config=cloud_speech.SpeakerDiarizationConfig(),
        ),
    )

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=recognizer,
        config=config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=audio_gcs_uri)],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            gcs_output_config=cloud_speech.GcsOutputConfig(uri=output_gcs_prefix)
        ),
    )

    op = client.batch_recognize(request=request)
    response = op.result(timeout=operation_timeout_s)

    result_manifest: dict[str, Any] = {}
    for uri, file_result in response.results.items():
        source_key = source_label if uri == audio_gcs_uri else uri
        if file_result.cloud_storage_result:
            result_manifest[source_key] = file_result.cloud_storage_result.uri
        else:
            result_manifest[source_key] = str(file_result)

    print("Google BatchRecognize finished. Output written in GCS:")
    for src, dst in result_manifest.items():
        print(f"  {src} -> {dst}")

    if response_manifest_path:
        write_json(response_manifest_path, result_manifest)
        print(f"Google result manifest written to: {response_manifest_path}")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI (must start with gs://): {uri}")
    bucket_and_path = uri[5:]
    if "/" not in bucket_and_path:
        return bucket_and_path, ""
    bucket, blob = bucket_and_path.split("/", 1)
    return bucket, blob


def upload_local_audio_to_temp_gcs(audio_file: Path, output_gcs_prefix: str, temp_prefix: str) -> tuple[str, str, str]:
    bucket_name, _ = parse_gcs_uri(output_gcs_prefix)
    object_name = f"{temp_prefix.rstrip('/')}/{uuid.uuid4()}-{audio_file.name}"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(audio_file))

    uploaded_uri = f"gs://{bucket_name}/{object_name}"
    print(f"Uploaded local audio to temporary GCS object: {uploaded_uri}")
    return uploaded_uri, bucket_name, object_name


def delete_gcs_object(bucket_name: str, object_name: str) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(object_name).delete()
    print(f"Deleted temporary GCS object: gs://{bucket_name}/{object_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "google"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate args/config and exit without API call")

    parser.add_argument("--audio-file", type=Path, help="Local MP3/MP4 path (OpenAI, or Google upload alternative)")
    parser.add_argument("--language", default="de", help="ISO-639-1 language for OpenAI (default: de)")
    parser.add_argument("--transcript-out", type=Path, default=DEFAULT_TRANSCRIPT_PATH)
    parser.add_argument("--diarized-json-out", type=Path, help="Output file for OpenAI diarization JSON")

    parser.add_argument("--audio-gcs-uri", help="Input GCS URI, e.g. gs://bucket/podcast.mp3 (Google only; alternative to --audio-file)")
    parser.add_argument("--gcs-output-prefix", help="Output GCS prefix, e.g. gs://bucket/stt_out/ (Google only)")
    parser.add_argument("--google-region", default="eu", help="Google Speech region, e.g. eu or us")
    parser.add_argument("--google-language-code", default="de-DE", help="Google BCP-47 language code")
    parser.add_argument("--google-timeout", type=int, default=3600, help="Operation timeout in seconds")
    parser.add_argument(
        "--google-manifest-out",
        type=Path,
        help="Optional local JSON to store source->result GCS mapping",
    )
    parser.add_argument(
        "--google-temp-upload-prefix",
        default="tmp/transcribe_audio_uploads",
        help="GCS object prefix for temporary uploads when using --audio-file with Google",
    )
    parser.add_argument(
        "--google-delete-temp-upload",
        action="store_true",
        help="Delete temporary uploaded object after Google transcription finishes",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.provider == "openai":
        if not args.audio_file:
            raise ValueError("--audio-file is required when --provider openai.")
        if not args.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
        if args.audio_gcs_uri:
            raise ValueError("--audio-gcs-uri is not supported when --provider openai.")
    elif args.provider == "google":
        provided_sources = int(bool(args.audio_gcs_uri)) + int(bool(args.audio_file))
        if provided_sources != 1:
            raise ValueError("Provide exactly one of --audio-gcs-uri or --audio-file when --provider google.")
        if args.audio_file and not args.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
        if not args.gcs_output_prefix:
            raise ValueError("--gcs-output-prefix is required when --provider google.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.dry_run:
        print("Dry run successful: argument validation passed.")
        return

    if args.provider == "openai":
        transcribe_openai(
            audio_path=args.audio_file,
            language=args.language,
            output_path=args.transcript_out,
            diarized_output=args.diarized_json_out,
        )
        return

    temp_upload_ref: tuple[str, str] | None = None
    google_audio_uri = args.audio_gcs_uri
    source_label = args.audio_gcs_uri

    if args.audio_file:
        google_audio_uri, bucket_name, object_name = upload_local_audio_to_temp_gcs(
            audio_file=args.audio_file,
            output_gcs_prefix=args.gcs_output_prefix,
            temp_prefix=args.google_temp_upload_prefix,
        )
        source_label = str(args.audio_file)
        temp_upload_ref = (bucket_name, object_name)

    try:
        transcribe_google(
            audio_gcs_uri=google_audio_uri,
            source_label=source_label,
            output_gcs_prefix=args.gcs_output_prefix,
            region=args.google_region,
            language_code=args.google_language_code,
            operation_timeout_s=args.google_timeout,
            response_manifest_path=args.google_manifest_out,
        )
    finally:
        if temp_upload_ref and args.google_delete_temp_upload:
            delete_gcs_object(bucket_name=temp_upload_ref[0], object_name=temp_upload_ref[1])


if __name__ == "__main__":
    main()
