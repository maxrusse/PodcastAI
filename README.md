# PodcastAI

PodcastAI converts a German podcast/lecture transcript into a complete learning package:

- Structured lesson sections with learning objectives and exam-style questions
- Optional generated section images and static radiology teaching images
- Episode-level after-learning package (blueprint 1-8)
- Optional Z3-SMP style oral exam material for the configured subject
- Combined DOCX export

This repository is the **first full version** built from `brainstorming.md`.

## Repository structure

- `openai_course_builder_2026.py` - OpenAI pipeline (text + images)
- `google_course_builder_2026.py` - Gemini pipeline (text + images)
- `requirements.txt` - Python dependencies
- `transcript.txt.example` - Input template
- `transcribe_audio.py` - Build transcript from MP3/MP4 (OpenAI or Google STT v2)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp transcript.txt.example transcript.txt
```

### Build transcript from MP3/MP4 first

`transcribe_audio.py` is the first step before running either course builder.

#### Quick usage summary

- **OpenAI provider**: local file input via `--audio-file` (for example `podcast.mp3` or `podcast.mp4`), writes local `transcript.txt`.
- **Google provider**: input via `--audio-gcs-uri` **or** local `--audio-file` (uploaded to a temporary GCS object), writes transcript JSON in GCS.
- Use `--dry-run` to validate parameters and required combinations without making API calls.

Example validation:

```bash
python transcribe_audio.py --provider openai --audio-file ./podcast.mp4 --dry-run
python transcribe_audio.py --provider google --audio-gcs-uri gs://YOUR_BUCKET/podcast.mp3 --gcs-output-prefix gs://YOUR_BUCKET/stt_out/ --dry-run
python transcribe_audio.py --provider google --audio-file ./podcast.mp4 --gcs-output-prefix gs://YOUR_BUCKET/stt_out/ --dry-run
```

#### OpenAI (local MP3/MP4 -> `transcript.txt` + optional diarization JSON)

```bash
export OPENAI_API_KEY="..."
python transcribe_audio.py \
  --provider openai \
  --audio-file ./podcast.mp3 \
  --language de \
  --transcript-out transcript.txt \
  --diarized-json-out out_openai/diarized_transcript.json
```

MP4 example:

```bash
python transcribe_audio.py --provider openai --audio-file ./lecture.mp4 --language de
```

#### Google Speech-to-Text v2 (GCS URI or local MP3/MP4 -> transcript JSON in GCS)

```bash
export GOOGLE_CLOUD_PROJECT="..."
python transcribe_audio.py \
  --provider google \
  --audio-gcs-uri gs://YOUR_BUCKET/podcast.mp3 \
  --gcs-output-prefix gs://YOUR_BUCKET/stt_out/ \
  --google-region eu \
  --google-language-code de-DE \
  --google-manifest-out out_google/stt_manifest.json
```

MP4 example:

```bash
python transcribe_audio.py \
  --provider google \
  --audio-gcs-uri gs://YOUR_BUCKET/lecture.mp4 \
  --gcs-output-prefix gs://YOUR_BUCKET/stt_out/
```

Local file upload example (Google mode):

```bash
export GOOGLE_CLOUD_PROJECT="..."
python transcribe_audio.py \
  --provider google \
  --audio-file ./lecture.mp4 \
  --gcs-output-prefix gs://YOUR_BUCKET/stt_out/ \
  --google-temp-upload-prefix tmp/transcribe_audio_uploads \
  --google-delete-temp-upload \
  --google-manifest-out out_google/stt_manifest.json
```

For local upload in Google mode, your credentials must allow both Speech and Cloud Storage operations:

- `GOOGLE_CLOUD_PROJECT` must be set.
- IAM permissions needed include uploading/deleting temporary objects in the target bucket (for example `storage.objects.create` and optionally `storage.objects.delete` when using `--google-delete-temp-upload`).
- Permissions for Speech-to-Text v2 batch recognition are still required.

Google batch mode writes recognition output to GCS. Use the reported output URI (or `--google-manifest-out`) to fetch JSON and generate/merge `transcript.txt` for downstream course building.

### Run OpenAI version

```bash
export OPENAI_API_KEY="..."
python openai_course_builder_2026.py
```

### Run Gemini version

```bash
export GEMINI_API_KEY="..."
python google_course_builder_2026.py
```

## Output

OpenAI outputs under `out_openai/` and Gemini outputs under `out_google/`.
Each run creates:

- `course_plan.json`
- `sections/*.md`
- `sections/*_openai.png` or `sections/*_gemini.png` (optional)
- `after_episode_package.json` + `.md`
- `materials/*.png` (optional)
- `z3_smp_<fach-slug>.json` + `.md` (for example `z3_smp_zahnaerztliche-prothetik.json`)
- `course.docx`

## Notes

- Code is ASCII-only; generated content may contain umlauts.
- For long transcripts, prefer splitting transcript into chunks and merging section plans.
- Generated medical content should be reviewed by qualified professionals before use.
