# PodcastAI

PodcastAI converts a German podcast/lecture transcript into a complete learning package:

- Structured lesson sections with learning objectives and exam-style questions
- Optional generated section images and static radiology teaching images
- Episode-level after-learning package (blueprint 1-8)
- Z3-SMP style oral exam material for dental prosthetics
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

#### OpenAI (local MP3/MP4 -> `transcript.txt` + diarization JSON)

```bash
export OPENAI_API_KEY="..."
python transcribe_audio.py \
  --provider openai \
  --audio-file ./podcast.mp3 \
  --language de \
  --transcript-out transcript.txt \
  --diarized-json-out out_openai/diarized_transcript.json
```

#### Google Speech-to-Text v2 (GCS MP3/MP4 -> transcript JSON in GCS)

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
- `z3_smp_prothetik.json` + `.md`
- `course.docx`

## Notes

- Code is ASCII-only; generated content may contain umlauts.
- For long transcripts, prefer splitting transcript into chunks and merging section plans.
- Generated medical content should be reviewed by qualified professionals before use.
