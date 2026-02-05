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

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp transcript.txt.example transcript.txt
```

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

### Validate external API connectivity

```bash
python validate_external_apis.py --provider all
```

This runs lightweight smoke tests against OpenAI and Gemini and reports `PASS`, `FAIL`, or `SKIP` depending on API key availability and call success.

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
