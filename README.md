# PodcastAI

PodcastAI turns a medical or dental podcast into exam-oriented learning material for advanced students through one official hybrid workflow:

- OpenAI for transcription, planning, text generation, and final language polishing
- Nano Banana for non-clinical explanatory infographics

Hard rule: never generate medical images. Visuals must stay conceptual, infographic-like, and non-clinical.

## Official Stack

- Text model: `gpt-5.4`
- Image model: `nano-banana-pro-preview`
- Final text polish: `FINAL_TEXT_RECHALLENGE=1`
- Canonical output root: `out_course/`

## Files

- `hybrid_course_builder.py` - main build pipeline
- `transcribe_audio.py` - transcript generation from local MP3/MP4
- `image_prompt_policy.py` - shared image prompt policy
- `build_short_script_from_outputs.py` - optional short-script / slide helper
- `validate_external_apis.py` - API smoke test for the official stack

## Workflow

1. Transcribe the podcast.
2. Build the course outputs.
3. Review the run folder in `out_course/`.

## Transcript

```bash
export OPENAI_API_KEY="..."
python transcribe_audio.py \
  --audio-file ./podcast.mp4 \
  --language de
```

Transcript runs are written to:

- `out_course/transcribe-openai-.../<timestamp>/intermediate/transcript.txt`

Latest transcript markers are written under:

- `out_course/LATEST_TRANSCRIBE_RUN.txt`
- `out_course/LATEST_TRANSCRIBE_TRANSCRIPT.txt`
- `out_course/LATEST_TRANSCRIBE_DIARIZED.txt`

## Build

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export OPENAI_TEXT_MODEL="gpt-5.4"
export BANANA_IMAGE_MODEL="nano-banana-pro-preview"
export FINAL_TEXT_RECHALLENGE="1"
python hybrid_course_builder.py
```

Official behavior:

- uses the latest transcript from `out_course/` unless `COURSE_TRANSCRIPT_PATH` is set
- writes one hybrid build run under `out_course/hybrid-course/<timestamp>/`
- generates only selective non-clinical explanatory images
- runs a final text-polish pass before export

## Outputs

Each build run creates:

- `course_plan.json`
- `sections/*.md`
- `sections/*.png`
- `need_to_know.md`
- `need_to_know.docx`
- `question_bank.json`
- `question_bank.md`
- `question_bank.docx`
- `course.docx`

Latest build markers are written under:

- `out_course/LATEST_BUILD_RUN.txt`
- `out_course/LATEST_BUILD_COURSE_DOCX.txt`

## Validation

```bash
python validate_external_apis.py --provider all
```

Optional overrides:

```bash
python validate_external_apis.py --provider openai --openai-model gpt-5.4
python validate_external_apis.py --provider banana --banana-model nano-banana-pro-preview
```
