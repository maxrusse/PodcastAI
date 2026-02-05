# openai_course_builder_2026.py
# End-to-end (OpenAI) as of 01/2026.
# Input: transcript.txt (UTF-8 text)
# Output:
#   out_openai/sections/*.md
#   out_openai/sections/*_openai.png
#   out_openai/after_episode_package.md
#   out_openai/materials/*.png
#   out_openai/z3_smp_prothetik.md
#   out_openai/course.docx
#
# Notes:
# - Code is ASCII-only. Model outputs may contain umlauts.
# - Requires: openai, python-docx, pillow

import json
import re
import base64
from pathlib import Path

from openai import OpenAI
from docx import Document
from docx.shared import Inches


# -----------------------------
# Config
# -----------------------------

IN_TRANSCRIPT = Path("transcript.txt")

OUT_DIR = Path("out_openai")
SECTIONS_DIR = OUT_DIR / "sections"
MATERIALS_DIR = OUT_DIR / "materials"

OPENAI_TEXT_MODEL = "gpt-5.2-thinking"
OPENAI_REASONING_EFFORT = "medium"
OPENAI_IMAGE_MODEL = "gpt-image-1"

MAKE_SECTION_IMAGES = True
MAKE_STATIC_MATERIAL_IMAGES = True

# Z3-SMP
MAKE_Z3_SMP = True
SMP_N_CASES = 1  # n = 1, modifiable
SMP_FACH = "Zahnaerztliche Prothetik"
SMP_DAUER_MIN = 20


# -----------------------------
# Helpers
# -----------------------------

def ensure_dirs() -> None:
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_text(path: Path, s: str) -> None:
    path.write_text(s, encoding="utf-8")


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] if s else "item"


def safe_json_load(s: str) -> dict:
    return json.loads(s)


def openai_generate_png(prompt: str, out_path: Path) -> None:
    client = OpenAI()
    img = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
    )
    b64 = img.data[0].b64_json
    out_path.write_bytes(base64.b64decode(b64))


# -----------------------------
# OpenAI: course plan (sections + written questions + per-section image prompt)
# -----------------------------

def openai_plan_course(transcript: str) -> dict:
    client = OpenAI()

    schema = {
        "type": "object",
        "properties": {
            "course_title": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "title": {"type": "string"},
                        "learning_objectives": {"type": "array", "items": {"type": "string"}},
                        "bullets": {"type": "array", "items": {"type": "string"}},
                        "key_terms": {"type": "array", "items": {"type": "string"}},
                        "image_brief": {"type": "string"},
                        "image_prompt": {"type": "string"},
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["mcq_sba", "short_answer", "case_vignette"]},
                                    "stem": {"type": "string"},
                                    "options": {
                                        "type": "object",
                                        "properties": {
                                            "A": {"type": "string"},
                                            "B": {"type": "string"},
                                            "C": {"type": "string"},
                                            "D": {"type": "string"},
                                            "E": {"type": "string"},
                                        },
                                        "required": ["A", "B", "C", "D", "E"],
                                        "additionalProperties": False,
                                    },
                                    "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                                    "rationale": {"type": "string"},
                                    "prompt": {"type": "string"},
                                    "expected_points": {"type": "array", "items": {"type": "string"}},
                                    "vignette": {"type": "string"},
                                    "tasks": {"type": "array", "items": {"type": "string"}},
                                    "answer_key": {"type": "array", "items": {"type": "string"}},
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "index",
                        "title",
                        "learning_objectives",
                        "bullets",
                        "key_terms",
                        "image_brief",
                        "image_prompt",
                        "questions",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["course_title", "sections"],
        "additionalProperties": False,
    }

    instructions = """
You generate a German medical learning handout from a lecture/podcast transcript.

Rules:
- Write in German. Output may include umlauts.
- Split into coherent lesson sections.
- For each section:
  - 3-6 learning objectives
  - detailed bullet points (not vague)
  - key terms
  - 3-8 exam questions aligned with German written exam style:
    * MCQ single best answer with exactly 5 options (A-E) and exactly one best answer
    * short-answer prompts when useful
    * one clinical case vignette when useful
- Be faithful to the transcript. If a claim is not supported, mark it as "unsicher" in the rationale instead of inventing facts.
- Provide an educational image_brief and an image_prompt per section (infographic/figure).
- Do not include any Markdown in JSON fields; plain text only.
""".strip()

    resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        instructions=instructions,
        input=[{"role": "user", "content": [{"type": "input_text", "text": transcript}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": "course_plan",
                "schema": schema,
                "strict": True,
            }
        },
    )

    return safe_json_load(resp.output_text)


# -----------------------------
# OpenAI: after-episode learning package (didactic blueprint)
# -----------------------------

def openai_after_episode_package(transcript: str) -> dict:
    client = OpenAI()

    schema = {
        "type": "object",
        "properties": {
            "learning_objectives_1_3": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}},
            "case_summary": {
                "type": "object",
                "properties": {
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "radiologic_problem": {"type": "string"},
                    "clinical_relevance": {"type": "string"},
                    "interdisciplinary_aspects": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key_findings", "radiologic_problem", "clinical_relevance", "interdisciplinary_aspects"],
                "additionalProperties": False,
            },
            "static_radiologic_materials": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "modality": {"type": "string", "enum": ["PA", "BW", "OPG", "CBCT"]},
                        "caption": {"type": "string"},
                        "image_prompt": {"type": "string"},
                    },
                    "required": ["modality", "caption", "image_prompt"],
                    "additionalProperties": False,
                },
            },
            "key_reasoning_steps_5": {
                "type": "object",
                "properties": {
                    "identify": {"type": "array", "items": {"type": "string"}},
                    "interpret": {"type": "array", "items": {"type": "string"}},
                    "prioritize": {"type": "array", "items": {"type": "string"}},
                    "integrate": {"type": "array", "items": {"type": "string"}},
                    "plan": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["identify", "interpret", "prioritize", "integrate", "plan"],
                "additionalProperties": False,
            },
            "take_home_messages_3_5": {"type": "array", "minItems": 3, "maxItems": 5, "items": {"type": "string"}},
            "micro_reflection_prompt_1": {"type": "string"},
            "optional_self_test_mcq_1": {
                "type": "object",
                "properties": {
                    "stem": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                            "E": {"type": "string"},
                        },
                        "required": ["A", "B", "C", "D", "E"],
                        "additionalProperties": False,
                    },
                    "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                    "rationale": {"type": "string"},
                },
                "required": ["stem", "options", "answer", "rationale"],
                "additionalProperties": False,
            },
            "key_resources_max_3": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["guideline", "NKLZ", "IMPP_GK", "reference", "other"]},
                        "citation": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["kind", "citation", "note"],
                    "additionalProperties": False,
                },
            },
            "traceability_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "learning_objectives_1_3",
            "case_summary",
            "static_radiologic_materials",
            "key_reasoning_steps_5",
            "take_home_messages_3_5",
            "micro_reflection_prompt_1",
            "optional_self_test_mcq_1",
            "key_resources_max_3",
            "traceability_notes",
        ],
        "additionalProperties": False,
    }

    instructions = """
Create an after-episode learning package in German from the transcript.

Didactic blueprint:
1) Learning objectives (1-3), competency-based, concise
2) Case summary: key findings, radiologic problem, clinical relevance, interdisciplinary aspects
3) Static radiologic materials: PA/BW/OPG/CBCT with short captions
   - Provide image_prompt for each material, suitable for generating a synthetic educational image (no real patient data).
4) Key reasoning steps using 5-step model: Identify -> Interpret -> Prioritize -> Integrate -> Plan
5) Take-home messages (3-5)
6) Micro-reflection prompt (1)
7) Optional self-test item (1 MCQ, A-E, single best answer)
8) Key resources (max 3): guidelines, NKLZ mapping, IMPP Gegenstandkatalog, essential references

Rules:
- Output may include umlauts.
- Be faithful to transcript; do not invent specifics. If a resource is not explicitly mentioned, write a generic placeholder and mark as "unsicher" in traceability_notes.
- For imaging prompts: describe a schematic or synthetic radiograph-like educational figure, not a real patient image.
- Do not include any Markdown in JSON fields; plain text only.
""".strip()

    resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        instructions=instructions,
        input=[{"role": "user", "content": [{"type": "input_text", "text": transcript}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": "after_episode_package",
                "schema": schema,
                "strict": True,
            }
        },
    )

    return safe_json_load(resp.output_text)


# -----------------------------
# OpenAI: Z3-SMP dental template (n cases, additive)
# -----------------------------

def openai_generate_z3_smp(transcript: str, n_cases: int, fach: str, dauer_min: int) -> dict:
    client = OpenAI()

    schema = {
        "type": "object",
        "properties": {
            "fach": {"type": "string"},
            "dauer_min": {"type": "integer"},
            "cases": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "aufgabe_nr": {"type": "string"},
                        "pruefungsziele": {"type": "array", "items": {"type": "string"}},
                        "pruefungsgrundlage": {"type": "array", "items": {"type": "string"}},
                        "pruefungsaufgabe_kurz": {"type": "string"},
                        "pruefer_intro": {"type": "string"},
                        "fall": {"type": "string"},
                        "aufgaben": {"type": "array", "items": {"type": "string"}},
                        "fragen_genau_2": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}},
                        "erwartungshorizont": {
                            "type": "object",
                            "properties": {
                                "note1_plus": {"type": "array", "items": {"type": "string"}},
                                "note3": {"type": "array", "items": {"type": "string"}},
                                "note5_minus": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["note1_plus", "note3", "note5_minus"],
                            "additionalProperties": False,
                        },
                        "prueferkommentar": {"type": "string"},
                        "anhaenge": {"type": "array", "items": {"type": "string"}},
                        "traceability_notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "aufgabe_nr",
                        "pruefungsziele",
                        "pruefungsgrundlage",
                        "pruefungsaufgabe_kurz",
                        "pruefer_intro",
                        "fall",
                        "aufgaben",
                        "fragen_genau_2",
                        "erwartungshorizont",
                        "prueferkommentar",
                        "anhaenge",
                        "traceability_notes",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["fach", "dauer_min", "cases"],
        "additionalProperties": False,
    }

    instructions = f"""
You generate fictional but plausible structured oral exam cases in German following a Z3-SMP template.

Template requirements:
- Fach: {fach}
- Dauer: {dauer_min} Minuten
- Produce EXACTLY {n_cases} case(s).
- Include:
  * Pruefungsziele
  * Pruefungsgrundlage (NKLZ / IMPP Gegenstandkatalog / Leitlinie if supported; otherwise placeholders + "unsicher" in traceability_notes)
  * Pruefungsaufgabe (klinisch-theoretisch) for ~{dauer_min} minutes
  * Pruefer intro line: "Ich werde Ihnen einen Fall beschreiben und anschliessend zwei Fragen stellen, wir haben {dauer_min} Min. Zeit."
  * Fall
  * Aufgaben
  * Exactly 2 questions (fragen_genau_2)
  * Erwartungshorizont split into three bands:
    - Fuehrt zur Notenverbesserung (bis Note 1)
    - Erwartete Loesungen (Note 3)
    - Fuehrt zur Notenverschlechterung (bis Note 5)
  * Prueferkommentar
  * Anhaenge list (what would be attached: e.g., PA/BW/OPG/CBCT text materials, checklists)
- Be faithful to transcript; do not invent specific numbers or drug doses unless stated.
""".strip()

    resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        instructions=instructions,
        input=[{"role": "user", "content": [{"type": "input_text", "text": transcript}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": "z3_smp",
                "schema": schema,
                "strict": True,
            }
        },
    )

    data = safe_json_load(resp.output_text)
    data["cases"] = data["cases"][:n_cases]
    data["fach"] = fach
    data["dauer_min"] = dauer_min
    return data


# -----------------------------
# Renderers: Markdown
# -----------------------------

def render_section_md(section: dict, image_rel: str | None) -> str:
    lines = []
    lines.append(f"# {int(section['index']):02d}. {section['title']}")
    lines.append("")
    if image_rel:
        lines.append(f"![Figure]({image_rel})")
        lines.append("")
    lines.append("## Lernziele")
    for lo in section["learning_objectives"]:
        lines.append(f"- {lo}")
    lines.append("")
    lines.append("## Inhalt (detaillierte Bulletpoints)")
    for b in section["bullets"]:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Key Terms")
    lines.append(", ".join(section["key_terms"]) if section["key_terms"] else "-")
    lines.append("")
    lines.append("## Testfragen (schriftlich orientiert)")
    qn = 0
    for q in section["questions"]:
        qn += 1
        t = q["type"]
        if t == "mcq_sba":
            lines.append(f"### {qn}. MCQ (Bestantwort, A-E)")
            lines.append(q.get("stem", "").strip())
            lines.append("")
            opts = q.get("options", {})
            for k in ["A", "B", "C", "D", "E"]:
                lines.append(f"- **{k}**: {opts.get(k, '')}")
            lines.append("")
            lines.append(f"**Loesung:** {q.get('answer', '')}")
            lines.append(f"**Begruendung:** {q.get('rationale', '').strip()}")
            lines.append("")
        elif t == "short_answer":
            lines.append(f"### {qn}. Kurzantwort")
            lines.append(q.get("prompt", "").strip())
            lines.append("")
            lines.append("**Erwartete Punkte:**")
            for p in q.get("expected_points", []):
                lines.append(f"- {p}")
            lines.append("")
        elif t == "case_vignette":
            lines.append(f"### {qn}. Fallvignette")
            lines.append(q.get("vignette", "").strip())
            lines.append("")
            lines.append("**Aufgaben:**")
            for tsk in q.get("tasks", []):
                lines.append(f"- {tsk}")
            lines.append("")
            lines.append("**Loesungsschluessel:**")
            for ak in q.get("answer_key", []):
                lines.append(f"- {ak}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_after_episode_md(pkg: dict, material_files: dict) -> str:
    lines = []
    lines.append("# After-Episode Learning Package")
    lines.append("")
    lines.append("## 1) Learning Objectives (1-3)")
    for x in pkg["learning_objectives_1_3"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 2) Case Summary")
    cs = pkg["case_summary"]
    lines.append("**Key findings:**")
    for x in cs["key_findings"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append(f"**Radiologic problem:** {cs['radiologic_problem']}")
    lines.append("")
    lines.append(f"**Clinical relevance:** {cs['clinical_relevance']}")
    lines.append("")
    lines.append("**Interdisciplinary aspects:**")
    for x in cs["interdisciplinary_aspects"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 3) Static Radiologic Materials")
    for item in pkg["static_radiologic_materials"]:
        mod = item["modality"]
        cap = item["caption"]
        img_rel = material_files.get(mod)
        if img_rel:
            lines.append(f"### {mod}")
            lines.append(f"![{mod}]({img_rel})")
            lines.append(cap)
            lines.append("")
        else:
            lines.append(f"- {mod}: {cap}")
    lines.append("")
    lines.append("## 4) Key Reasoning Steps (5-step model)")
    rs = pkg["key_reasoning_steps_5"]
    lines.append("### Identify")
    for x in rs["identify"]:
        lines.append(f"- {x}")
    lines.append("### Interpret")
    for x in rs["interpret"]:
        lines.append(f"- {x}")
    lines.append("### Prioritize")
    for x in rs["prioritize"]:
        lines.append(f"- {x}")
    lines.append("### Integrate")
    for x in rs["integrate"]:
        lines.append(f"- {x}")
    lines.append("### Plan")
    for x in rs["plan"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 5) Take-Home Messages (3-5)")
    for x in pkg["take_home_messages_3_5"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 6) Micro-Reflection Prompt (1)")
    lines.append(pkg["micro_reflection_prompt_1"])
    lines.append("")
    lines.append("## 7) Optional Self-Test Item (1 MCQ)")
    mcq = pkg["optional_self_test_mcq_1"]
    lines.append(mcq["stem"])
    lines.append("")
    for k in ["A", "B", "C", "D", "E"]:
        lines.append(f"- **{k}**: {mcq['options'][k]}")
    lines.append("")
    lines.append(f"**Loesung:** {mcq['answer']}")
    lines.append(f"**Begruendung:** {mcq['rationale']}")
    lines.append("")
    lines.append("## 8) Key Resources (max 3)")
    for r in pkg["key_resources_max_3"]:
        lines.append(f"- [{r['kind']}] {r['citation']} - {r['note']}")
    lines.append("")
    lines.append("## Traceability Notes")
    for t in pkg["traceability_notes"]:
        lines.append(f"- {t}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_z3_smp_md(z3: dict) -> str:
    lines = []
    lines.append("# Z3-SMP")
    lines.append("")
    lines.append(f"Fach: {z3['fach']}\tAufgabe Nr.: __________")
    lines.append(f"Datum: ________________")
    lines.append(f"Gruppe: ____________\tPruefling: ____________________")
    lines.append(f"Pruefer: ________________")
    lines.append(f"Beisitzer: ________________\t\t\tProtokoll")
    lines.append("")
    lines.append("## Pruefungsziele:")
    lines.append("")
    case0 = z3["cases"][0]
    for x in case0["pruefungsziele"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Pruefungsgrundlage: (aus NKLZ, IMPP-Gegenstandkatalog oder Leitlinie)")
    lines.append("")
    for x in case0["pruefungsgrundlage"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append(f"## Pruefungsaufgabe (klinisch-theoretisch): ungefaehre Zeitangabe: {z3['dauer_min']} Minuten")
    lines.append("")
    lines.append(case0["pruefer_intro"])
    lines.append("")
    lines.append("### Fall:")
    lines.append(case0["fall"])
    lines.append("")
    lines.append("### Aufgaben:")
    for x in case0["aufgaben"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("### Fragen (genau 2):")
    for i, q in enumerate(case0["fragen_genau_2"], start=1):
        lines.append(f"{i}. {q}")
    lines.append("")
    lines.append("## Erwartungshorizont (erwartete Loesungen):")
    lines.append("")
    eh = case0["erwartungshorizont"]
    lines.append("| Fuehrt zur Notenverbesserung (bis Note 1) | Erwartete Loesungen (Note 3) | Fuehrt zur Notenverschlechterung (bis Note 5) |")
    lines.append("|---|---|---|")
    rows = max(len(eh["note1_plus"]), len(eh["note3"]), len(eh["note5_minus"]))
    for i in range(rows):
        a = eh["note1_plus"][i] if i < len(eh["note1_plus"]) else ""
        b = eh["note3"][i] if i < len(eh["note3"]) else ""
        c = eh["note5_minus"][i] if i < len(eh["note5_minus"]) else ""
        lines.append(f"| {a} | {b} | {c} |")
    lines.append("")
    lines.append("## Prueferkommentar:")
    lines.append("")
    lines.append(case0["prueferkommentar"] if case0["prueferkommentar"] else "______________________________")
    lines.append("")
    lines.append("## Bewertung in Noten 1-5:")
    lines.append("")
    lines.append("_____")
    lines.append("")
    lines.append("Unterschrift Pruefer: ____________________")
    lines.append("")
    lines.append("Unterschrift Beisitzer: __________________")
    lines.append("")
    lines.append("## Anhaenge")
    for x in case0["anhaenge"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Traceability Notes")
    for x in case0["traceability_notes"]:
        lines.append(f"- {x}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# -----------------------------
# DOCX builder
# -----------------------------

def build_docx(course_title: str, sections: list, after_md: str, z3_md: str | None, material_paths: list) -> None:
    doc = Document()
    doc.add_heading(course_title, level=0)

    # Course sections
    for section in sections:
        idx = int(section["index"])
        title = section["title"]
        slug = slugify(title)

        doc.add_heading(f"{idx:02d}. {title}", level=1)

        img_path = SECTIONS_DIR / f"{idx:02d}_{slug}_openai.png"
        if img_path.exists():
            doc.add_picture(str(img_path), width=Inches(5.8))

        doc.add_heading("Lernziele", level=2)
        for lo in section["learning_objectives"]:
            doc.add_paragraph(lo, style="List Bullet")

        doc.add_heading("Inhalt", level=2)
        for b in section["bullets"]:
            doc.add_paragraph(b, style="List Bullet")

        doc.add_heading("Key Terms", level=2)
        doc.add_paragraph(", ".join(section["key_terms"]) if section["key_terms"] else "-")

        doc.add_heading("Testfragen (schriftlich orientiert)", level=2)
        qn = 0
        for q in section["questions"]:
            qn += 1
            t = q["type"]
            if t == "mcq_sba":
                doc.add_paragraph(f"{qn}. MCQ (Bestantwort)")
                doc.add_paragraph(q.get("stem", "").strip())
                opts = q.get("options", {})
                for k in ["A", "B", "C", "D", "E"]:
                    doc.add_paragraph(f"{k}: {opts.get(k, '')}", style="List Bullet")
                doc.add_paragraph(f"Loesung: {q.get('answer','')}")
                doc.add_paragraph(f"Begruendung: {q.get('rationale','').strip()}")
            elif t == "short_answer":
                doc.add_paragraph(f"{qn}. Kurzantwort")
                doc.add_paragraph(q.get("prompt", "").strip())
                doc.add_paragraph("Erwartete Punkte:")
                for p in q.get("expected_points", []):
                    doc.add_paragraph(p, style="List Bullet")
            elif t == "case_vignette":
                doc.add_paragraph(f"{qn}. Fallvignette")
                doc.add_paragraph(q.get("vignette", "").strip())
                doc.add_paragraph("Aufgaben:")
                for x in q.get("tasks", []):
                    doc.add_paragraph(x, style="List Bullet")
                doc.add_paragraph("Loesungsschluessel:")
                for x in q.get("answer_key", []):
                    doc.add_paragraph(x, style="List Bullet")

        doc.add_page_break()

    # After-episode package
    doc.add_heading("After-Episode Learning Package", level=1)
    for line in after_md.splitlines():
        doc.add_paragraph(line)

    # Insert static materials images at end
    if material_paths:
        doc.add_heading("Static Radiologic Materials", level=2)
        for p in material_paths:
            if p.exists():
                doc.add_picture(str(p), width=Inches(5.8))
    doc.add_page_break()

    # Z3-SMP
    if z3_md:
        doc.add_heading("Z3-SMP", level=1)
        for line in z3_md.splitlines():
            doc.add_paragraph(line)
        doc.add_page_break()

    doc.save(str(OUT_DIR / "course.docx"))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ensure_dirs()
    transcript = read_text(IN_TRANSCRIPT)

    # 1) Course plan (sections)
    plan = openai_plan_course(transcript)
    write_text(OUT_DIR / "course_plan.json", json.dumps(plan, indent=2, ensure_ascii=False))

    sections = plan["sections"]

    # Write sections: images + md
    for section in sections:
        idx = int(section["index"])
        title = section["title"]
        slug = slugify(title)

        md_path = SECTIONS_DIR / f"{idx:02d}_{slug}.md"
        img_path = SECTIONS_DIR / f"{idx:02d}_{slug}_openai.png"

        image_rel = None
        if MAKE_SECTION_IMAGES:
            try:
                openai_generate_png(section["image_prompt"], img_path)
                image_rel = img_path.name
            except Exception as e:
                print(f"[WARN] section image failed {idx}: {e}")

        md = render_section_md(section, image_rel)
        write_text(md_path, md)

    # 2) After-episode learning package + static materials images
    pkg = openai_after_episode_package(transcript)
    write_text(OUT_DIR / "after_episode_package.json", json.dumps(pkg, indent=2, ensure_ascii=False))

    material_files_rel = {}
    material_paths = []
    if MAKE_STATIC_MATERIAL_IMAGES:
        for item in pkg["static_radiologic_materials"]:
            mod = item["modality"]
            prompt = item["image_prompt"]
            out_name = f"{mod.lower()}_{slugify(item['caption'])[:40]}_openai.png"
            out_path = MATERIALS_DIR / out_name
            try:
                openai_generate_png(prompt, out_path)
                material_files_rel[mod] = str((Path("materials") / out_name).as_posix())
                material_paths.append(out_path)
            except Exception as e:
                print(f"[WARN] material image failed {mod}: {e}")

    after_md = render_after_episode_md(pkg, material_files_rel)
    write_text(OUT_DIR / "after_episode_package.md", after_md)

    # 3) Z3-SMP (dental prothetik template), additive
    z3_md = None
    if MAKE_Z3_SMP:
        z3 = openai_generate_z3_smp(transcript, SMP_N_CASES, SMP_FACH, SMP_DAUER_MIN)
        write_text(OUT_DIR / "z3_smp_prothetik.json", json.dumps(z3, indent=2, ensure_ascii=False))
        z3_md = render_z3_smp_md(z3)
        write_text(OUT_DIR / "z3_smp_prothetik.md", z3_md)

    # 4) DOCX
    build_docx(plan["course_title"], sections, after_md, z3_md, material_paths)

    print("DONE -> out_openai/")


if __name__ == "__main__":
    # Needs: export OPENAI_API_KEY="..."
    main()
# google_course_builder_2026.py
# End-to-end (Google Gemini) as of 01/2026.
# Input: transcript.txt (UTF-8 text)
# Output:
#   out_google/sections/*.md
#   out_google/sections/*_gemini.png
#   out_google/after_episode_package.md
#   out_google/materials/*.png
#   out_google/z3_smp_prothetik.md
#   out_google/course.docx
#
# Notes:
# - Code is ASCII-only. Model outputs may contain umlauts.
# - Requires: google-genai, python-docx, pillow

import json
import re
from pathlib import Path

from google import genai
from docx import Document
from docx.shared import Inches


# -----------------------------
# Config
# -----------------------------

IN_TRANSCRIPT = Path("transcript.txt")

OUT_DIR = Path("out_google")
SECTIONS_DIR = OUT_DIR / "sections"
MATERIALS_DIR = OUT_DIR / "materials"

GEMINI_TEXT_MODEL = "gemini-2.5-pro"
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"  # Nano Banana Pro (preview)
# Alternative faster images:
# GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"

MAKE_SECTION_IMAGES = True
MAKE_STATIC_MATERIAL_IMAGES = True

MAKE_Z3_SMP = True
SMP_N_CASES = 1
SMP_FACH = "Zahnaerztliche Prothetik"
SMP_DAUER_MIN = 20


# -----------------------------
# Helpers
# -----------------------------

def ensure_dirs() -> None:
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MATERIALS_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_text(path: Path, s: str) -> None:
    path.write_text(s, encoding="utf-8")


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] if s else "item"


def extract_first_json(text: str) -> dict:
    # Prefer a full JSON object at end; fallback to first/last braces.
    m = re.search(r"\{.*\}\s*$", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    a = text.find("{")
    b = text.rfind("}")
    if a < 0 or b < 0 or b <= a:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[a : b + 1])


def gemini_generate_png(prompt: str, out_path: Path) -> None:
    client = genai.Client()
    resp = client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=[prompt])
    for part in resp.parts:
        if getattr(part, "inline_data", None) is not None:
            img = part.as_image()
            img.save(str(out_path))
            return
    raise RuntimeError("No image returned by Gemini image model.")


# -----------------------------
# Gemini: course plan JSON
# -----------------------------

def gemini_plan_course(transcript: str) -> dict:
    client = genai.Client()
    prompt = f"""
Return ONLY valid JSON (no markdown, no commentary).

You generate a German medical learning handout from a lecture/podcast transcript.

Rules:
- Write in German. Output may include umlauts.
- Split into coherent lesson sections.
- For each section:
  - 3-6 learning objectives
  - detailed bullet points (not vague)
  - key terms
  - 3-8 exam questions aligned with German written exam style:
    * MCQ single best answer with exactly 5 options (A-E) and exactly one best answer
    * short-answer prompts when useful
    * one clinical case vignette when useful
- Be faithful to transcript. If a claim is not supported, mark it as "unsicher" in the rationale instead of inventing facts.
- Provide image_brief and image_prompt per section (infographic/figure).
- Do not include any Markdown inside JSON fields; plain text only.

JSON shape:
{{
  "course_title": "...",
  "sections": [
    {{
      "index": 1,
      "title": "...",
      "learning_objectives": ["..."],
      "bullets": ["..."],
      "key_terms": ["..."],
      "image_brief": "...",
      "image_prompt": "...",
      "questions": [
        {{
          "type": "mcq_sba",
          "stem": "...",
          "options": {{"A":"...","B":"...","C":"...","D":"...","E":"..."}},
          "answer": "C",
          "rationale": "..."
        }},
        {{
          "type": "short_answer",
          "prompt": "...",
          "expected_points": ["..."]
        }},
        {{
          "type": "case_vignette",
          "vignette": "...",
          "tasks": ["..."],
          "answer_key": ["..."]
        }}
      ]
    }}
  ]
}}

Transcript:
{transcript}
""".strip()

    resp = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=[prompt],
        config={"response_mime_type": "application/json"},
    )
    text = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
    return extract_first_json(text)


def gemini_after_episode_package(transcript: str) -> dict:
    client = genai.Client()
    prompt = f"""
Return ONLY valid JSON (no markdown, no commentary).

Create an after-episode learning package in German from the transcript.

Didactic blueprint:
1) Learning objectives (1-3), competency-based, concise
2) Case summary: key findings, radiologic problem, clinical relevance, interdisciplinary aspects
3) Static radiologic materials: PA/BW/OPG/CBCT with short captions
   - Provide image_prompt for each material, suitable for generating a synthetic educational image (no real patient data).
4) Key reasoning steps using 5-step model: Identify -> Interpret -> Prioritize -> Integrate -> Plan
5) Take-home messages (3-5)
6) Micro-reflection prompt (1)
7) Optional self-test item (1 MCQ, A-E, single best answer)
8) Key resources (max 3): guidelines, NKLZ mapping, IMPP Gegenstandkatalog, essential references

Rules:
- Be faithful to transcript; do not invent specifics. If a resource is not mentioned, write a placeholder and mark as "unsicher" in traceability_notes.
- Imaging prompts should produce schematic/synthetic radiograph-like educational figures.

JSON shape:
{{
  "learning_objectives_1_3": ["..."],
  "case_summary": {{
    "key_findings": ["..."],
    "radiologic_problem": "...",
    "clinical_relevance": "...",
    "interdisciplinary_aspects": ["..."]
  }},
  "static_radiologic_materials": [
    {{"modality":"PA","caption":"...","image_prompt":"..."}},
    {{"modality":"BW","caption":"...","image_prompt":"..."}},
    {{"modality":"OPG","caption":"...","image_prompt":"..."}},
    {{"modality":"CBCT","caption":"...","image_prompt":"..."}}
  ],
  "key_reasoning_steps_5": {{
    "identify": ["..."],
    "interpret": ["..."],
    "prioritize": ["..."],
    "integrate": ["..."],
    "plan": ["..."]
  }},
  "take_home_messages_3_5": ["..."],
  "micro_reflection_prompt_1": "...",
  "optional_self_test_mcq_1": {{
    "stem": "...",
    "options": {{"A":"...","B":"...","C":"...","D":"...","E":"..."}},
    "answer": "C",
    "rationale": "..."
  }},
  "key_resources_max_3": [
    {{"kind":"guideline","citation":"...","note":"..."}}
  ],
  "traceability_notes": ["..."]
}}

Transcript:
{transcript}
""".strip()

    resp = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=[prompt],
        config={"response_mime_type": "application/json"},
    )
    text = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
    return extract_first_json(text)


def gemini_generate_z3_smp(transcript: str, n_cases: int, fach: str, dauer_min: int) -> dict:
    client = genai.Client()
    prompt = f"""
Return ONLY valid JSON (no markdown, no commentary).

Generate fictional but plausible structured oral exam cases in German following a Z3-SMP template.

Template requirements:
- Fach: {fach}
- Dauer: {dauer_min} Minuten
- Produce EXACTLY {n_cases} case(s).
- Include:
  * Pruefungsziele
  * Pruefungsgrundlage (NKLZ / IMPP-Gegenstandkatalog / Leitlinie if supported; otherwise placeholders + "unsicher" in traceability_notes)
  * Pruefungsaufgabe_kurz (klinisch-theoretisch)
  * Pruefer intro line: "Ich werde Ihnen einen Fall beschreiben und anschliessend zwei Fragen stellen, wir haben {dauer_min} Min. Zeit."
  * Fall
  * Aufgaben
  * Exactly 2 questions
  * Erwartungshorizont split into note1_plus / note3 / note5_minus
  * Prueferkommentar
  * Anhaenge list
  * Traceability notes

JSON shape:
{{
  "fach": "{fach}",
  "dauer_min": {dauer_min},
  "cases": [
    {{
      "aufgabe_nr": "Z3-01",
      "pruefungsziele": ["..."],
      "pruefungsgrundlage": ["..."],
      "pruefungsaufgabe_kurz": "...",
      "pruefer_intro": "Ich werde Ihnen einen Fall beschreiben und anschliessend zwei Fragen stellen, wir haben {dauer_min} Min. Zeit.",
      "fall": "...",
      "aufgaben": ["..."],
      "fragen_genau_2": ["...","..."],
      "erwartungshorizont": {{
        "note1_plus": ["..."],
        "note3": ["..."],
        "note5_minus": ["..."]
      }},
      "prueferkommentar": "...",
      "anhaenge": ["..."],
      "traceability_notes": ["..."]
    }}
  ]
}}

Transcript:
{transcript}
""".strip()

    resp = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=[prompt],
        config={"response_mime_type": "application/json"},
    )
    text = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
    data = extract_first_json(text)
    data["cases"] = data.get("cases", [])[:n_cases]
    data["fach"] = fach
    data["dauer_min"] = dauer_min
    return data


# -----------------------------
# Renderers: Markdown
# -----------------------------

def render_section_md(section: dict, image_rel: str | None) -> str:
    lines = []
    lines.append(f"# {int(section['index']):02d}. {section['title']}")
    lines.append("")
    if image_rel:
        lines.append(f"![Figure]({image_rel})")
        lines.append("")
    lines.append("## Lernziele")
    for lo in section.get("learning_objectives", []):
        lines.append(f"- {lo}")
    lines.append("")
    lines.append("## Inhalt (detaillierte Bulletpoints)")
    for b in section.get("bullets", []):
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Key Terms")
    kt = section.get("key_terms", [])
    lines.append(", ".join(kt) if kt else "-")
    lines.append("")
    lines.append("## Testfragen (schriftlich orientiert)")
    qn = 0
    for q in section.get("questions", []):
        qn += 1
        t = q.get("type", "")
        if t == "mcq_sba":
            lines.append(f"### {qn}. MCQ (Bestantwort, A-E)")
            lines.append(q.get("stem", "").strip())
            lines.append("")
            opts = q.get("options", {})
            for k in ["A", "B", "C", "D", "E"]:
                lines.append(f"- **{k}**: {opts.get(k, '')}")
            lines.append("")
            lines.append(f"**Loesung:** {q.get('answer', '')}")
            lines.append(f"**Begruendung:** {q.get('rationale', '').strip()}")
            lines.append("")
        elif t == "short_answer":
            lines.append(f"### {qn}. Kurzantwort")
            lines.append(q.get("prompt", "").strip())
            lines.append("")
            lines.append("**Erwartete Punkte:**")
            for p in q.get("expected_points", []):
                lines.append(f"- {p}")
            lines.append("")
        elif t == "case_vignette":
            lines.append(f"### {qn}. Fallvignette")
            lines.append(q.get("vignette", "").strip())
            lines.append("")
            lines.append("**Aufgaben:**")
            for tsk in q.get("tasks", []):
                lines.append(f"- {tsk}")
            lines.append("")
            lines.append("**Loesungsschluessel:**")
            for ak in q.get("answer_key", []):
                lines.append(f"- {ak}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_after_episode_md(pkg: dict, material_files: dict) -> str:
    lines = []
    lines.append("# After-Episode Learning Package")
    lines.append("")
    lines.append("## 1) Learning Objectives (1-3)")
    for x in pkg["learning_objectives_1_3"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 2) Case Summary")
    cs = pkg["case_summary"]
    lines.append("**Key findings:**")
    for x in cs["key_findings"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append(f"**Radiologic problem:** {cs['radiologic_problem']}")
    lines.append("")
    lines.append(f"**Clinical relevance:** {cs['clinical_relevance']}")
    lines.append("")
    lines.append("**Interdisciplinary aspects:**")
    for x in cs["interdisciplinary_aspects"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 3) Static Radiologic Materials")
    for item in pkg["static_radiologic_materials"]:
        mod = item["modality"]
        cap = item["caption"]
        img_rel = material_files.get(mod)
        if img_rel:
            lines.append(f"### {mod}")
            lines.append(f"![{mod}]({img_rel})")
            lines.append(cap)
            lines.append("")
        else:
            lines.append(f"- {mod}: {cap}")
    lines.append("")
    lines.append("## 4) Key Reasoning Steps (5-step model)")
    rs = pkg["key_reasoning_steps_5"]
    lines.append("### Identify")
    for x in rs["identify"]:
        lines.append(f"- {x}")
    lines.append("### Interpret")
    for x in rs["interpret"]:
        lines.append(f"- {x}")
    lines.append("### Prioritize")
    for x in rs["prioritize"]:
        lines.append(f"- {x}")
    lines.append("### Integrate")
    for x in rs["integrate"]:
        lines.append(f"- {x}")
    lines.append("### Plan")
    for x in rs["plan"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 5) Take-Home Messages (3-5)")
    for x in pkg["take_home_messages_3_5"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## 6) Micro-Reflection Prompt (1)")
    lines.append(pkg["micro_reflection_prompt_1"])
    lines.append("")
    lines.append("## 7) Optional Self-Test Item (1 MCQ)")
    mcq = pkg["optional_self_test_mcq_1"]
    lines.append(mcq["stem"])
    lines.append("")
    for k in ["A", "B", "C", "D", "E"]:
        lines.append(f"- **{k}**: {mcq['options'][k]}")
    lines.append("")
    lines.append(f"**Loesung:** {mcq['answer']}")
    lines.append(f"**Begruendung:** {mcq['rationale']}")
    lines.append("")
    lines.append("## 8) Key Resources (max 3)")
    for r in pkg["key_resources_max_3"]:
        lines.append(f"- [{r['kind']}] {r['citation']} - {r['note']}")
    lines.append("")
    lines.append("## Traceability Notes")
    for t in pkg["traceability_notes"]:
        lines.append(f"- {t}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_z3_smp_md(z3: dict) -> str:
    lines = []
    lines.append("# Z3-SMP")
    lines.append("")
    lines.append(f"Fach: {z3['fach']}\tAufgabe Nr.: __________")
    lines.append("Datum: ________________")
    lines.append("Gruppe: ____________\tPruefling: ____________________")
    lines.append("Pruefer: ________________")
    lines.append("Beisitzer: ________________\t\t\tProtokoll")
    lines.append("")
    lines.append("## Pruefungsziele:")
    lines.append("")
    case0 = z3["cases"][0]
    for x in case0["pruefungsziele"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Pruefungsgrundlage: (aus NKLZ, IMPP-Gegenstandkatalog oder Leitlinie)")
    lines.append("")
    for x in case0["pruefungsgrundlage"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append(f"## Pruefungsaufgabe (klinisch-theoretisch): ungefaehre Zeitangabe: {z3['dauer_min']} Minuten")
    lines.append("")
    lines.append(case0["pruefer_intro"])
    lines.append("")
    lines.append("### Fall:")
    lines.append(case0["fall"])
    lines.append("")
    lines.append("### Aufgaben:")
    for x in case0["aufgaben"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("### Fragen (genau 2):")
    for i, q in enumerate(case0["fragen_genau_2"], start=1):
        lines.append(f"{i}. {q}")
    lines.append("")
    lines.append("## Erwartungshorizont (erwartete Loesungen):")
    lines.append("")
    eh = case0["erwartungshorizont"]
    lines.append("| Fuehrt zur Notenverbesserung (bis Note 1) | Erwartete Loesungen (Note 3) | Fuehrt zur Notenverschlechterung (bis Note 5) |")
    lines.append("|---|---|---|")
    rows = max(len(eh["note1_plus"]), len(eh["note3"]), len(eh["note5_minus"]))
    for i in range(rows):
        a = eh["note1_plus"][i] if i < len(eh["note1_plus"]) else ""
        b = eh["note3"][i] if i < len(eh["note3"]) else ""
        c = eh["note5_minus"][i] if i < len(eh["note5_minus"]) else ""
        lines.append(f"| {a} | {b} | {c} |")
    lines.append("")
    lines.append("## Prueferkommentar:")
    lines.append("")
    lines.append(case0["prueferkommentar"] if case0["prueferkommentar"] else "______________________________")
    lines.append("")
    lines.append("## Bewertung in Noten 1-5:")
    lines.append("")
    lines.append("_____")
    lines.append("")
    lines.append("Unterschrift Pruefer: ____________________")
    lines.append("")
    lines.append("Unterschrift Beisitzer: __________________")
    lines.append("")
    lines.append("## Anhaenge")
    for x in case0["anhaenge"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Traceability Notes")
    for x in case0["traceability_notes"]:
        lines.append(f"- {x}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# -----------------------------
# DOCX builder
# -----------------------------

def build_docx(course_title: str, sections: list, after_md: str, z3_md: str | None, material_paths: list) -> None:
    doc = Document()
    doc.add_heading(course_title, level=0)

    for section in sections:
        idx = int(section["index"])
        title = section["title"]
        slug = slugify(title)

        doc.add_heading(f"{idx:02d}. {title}", level=1)

        img_path = SECTIONS_DIR / f"{idx:02d}_{slug}_gemini.png"
        if img_path.exists():
            doc.add_picture(str(img_path), width=Inches(5.8))

        doc.add_heading("Lernziele", level=2)
        for lo in section.get("learning_objectives", []):
            doc.add_paragraph(lo, style="List Bullet")

        doc.add_heading("Inhalt", level=2)
        for b in section.get("bullets", []):
            doc.add_paragraph(b, style="List Bullet")

        doc.add_heading("Key Terms", level=2)
        kt = section.get("key_terms", [])
        doc.add_paragraph(", ".join(kt) if kt else "-")

        doc.add_heading("Testfragen (schriftlich orientiert)", level=2)
        qn = 0
        for q in section.get("questions", []):
            qn += 1
            t = q.get("type", "")
            if t == "mcq_sba":
                doc.add_paragraph(f"{qn}. MCQ (Bestantwort)")
                doc.add_paragraph(q.get("stem", "").strip())
                opts = q.get("options", {})
                for k in ["A", "B", "C", "D", "E"]:
                    doc.add_paragraph(f"{k}: {opts.get(k, '')}", style="List Bullet")
                doc.add_paragraph(f"Loesung: {q.get('answer','')}")
                doc.add_paragraph(f"Begruendung: {q.get('rationale','').strip()}")
            elif t == "short_answer":
                doc.add_paragraph(f"{qn}. Kurzantwort")
                doc.add_paragraph(q.get("prompt", "").strip())
                doc.add_paragraph("Erwartete Punkte:")
                for p in q.get("expected_points", []):
                    doc.add_paragraph(p, style="List Bullet")
            elif t == "case_vignette":
                doc.add_paragraph(f"{qn}. Fallvignette")
                doc.add_paragraph(q.get("vignette", "").strip())
                doc.add_paragraph("Aufgaben:")
                for x in q.get("tasks", []):
                    doc.add_paragraph(x, style="List Bullet")
                doc.add_paragraph("Loesungsschluessel:")
                for x in q.get("answer_key", []):
                    doc.add_paragraph(x, style="List Bullet")

        doc.add_page_break()

    doc.add_heading("After-Episode Learning Package", level=1)
    for line in after_md.splitlines():
        doc.add_paragraph(line)

    if material_paths:
        doc.add_heading("Static Radiologic Materials", level=2)
        for p in material_paths:
            if p.exists():
                doc.add_picture(str(p), width=Inches(5.8))
    doc.add_page_break()

    if z3_md:
        doc.add_heading("Z3-SMP", level=1)
        for line in z3_md.splitlines():
            doc.add_paragraph(line)
        doc.add_page_break()

    doc.save(str(OUT_DIR / "course.docx"))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ensure_dirs()
    transcript = read_text(IN_TRANSCRIPT)

    plan = gemini_plan_course(transcript)
    write_text(OUT_DIR / "course_plan.json", json.dumps(plan, indent=2, ensure_ascii=False))

    sections = plan.get("sections", [])

    for section in sections:
        idx = int(section["index"])
        title = section["title"]
        slug = slugify(title)

        md_path = SECTIONS_DIR / f"{idx:02d}_{slug}.md"
        img_path = SECTIONS_DIR / f"{idx:02d}_{slug}_gemini.png"

        image_rel = None
        if MAKE_SECTION_IMAGES:
            try:
                gemini_generate_png(section["image_prompt"], img_path)
                image_rel = img_path.name
            except Exception as e:
                print(f"[WARN] section image failed {idx}: {e}")

        md = render_section_md(section, image_rel)
        write_text(md_path, md)

    pkg = gemini_after_episode_package(transcript)
    write_text(OUT_DIR / "after_episode_package.json", json.dumps(pkg, indent=2, ensure_ascii=False))

    material_files_rel = {}
    material_paths = []
    if MAKE_STATIC_MATERIAL_IMAGES:
        for item in pkg["static_radiologic_materials"]:
            mod = item["modality"]
            prompt = item["image_prompt"]
            out_name = f"{mod.lower()}_{slugify(item['caption'])[:40]}_gemini.png"
            out_path = MATERIALS_DIR / out_name
            try:
                gemini_generate_png(prompt, out_path)
                material_files_rel[mod] = str((Path("materials") / out_name).as_posix())
                material_paths.append(out_path)
            except Exception as e:
                print(f"[WARN] material image failed {mod}: {e}")

    after_md = render_after_episode_md(pkg, material_files_rel)
    write_text(OUT_DIR / "after_episode_package.md", after_md)

    z3_md = None
    if MAKE_Z3_SMP:
        z3 = gemini_generate_z3_smp(transcript, SMP_N_CASES, SMP_FACH, SMP_DAUER_MIN)
        write_text(OUT_DIR / "z3_smp_prothetik.json", json.dumps(z3, indent=2, ensure_ascii=False))
        z3_md = render_z3_smp_md(z3)
        write_text(OUT_DIR / "z3_smp_prothetik.md", z3_md)

    build_docx(plan.get("course_title", "Course"), sections, after_md, z3_md, material_paths)
    print("DONE -> out_google/")


if __name__ == "__main__":
    # Needs: export GEMINI_API_KEY="..."
    main()
# requirements.txt
openai>=1.40.0
google-genai>=1.0.0
python-docx>=1.1.0
pillow>=10.0.0
# run OpenAI
export OPENAI_API_KEY="..."
python openai_course_builder_2026.py

# run Gemini
export GEMINI_API_KEY="..."
python google_course_builder_2026.py
Du:
Hight level report with added tech specs 
ChatGPT:

Nachgedacht für 19s
High-level Report: After-Episode Learning Package + Z3-SMP (OpenAI vs. Google) mit Tech Specs (Stand: 01/2026)
Zielbild
Aus einer Podcast-/Vortragsfolge entsteht ein komplettes, didaktisch strukturiertes Lernpaket inkl.:

Lesson Sections (strukturierte Kapitel)

Detaillierte Bulletpoints + Lernziele

Prüfungsfragen (schriftlich orientiert: MCQ A–E, Kurzantwort, Fallvignette)

After-Episode Learning Package nach deinem Blueprint (1–8)

Z3-SMP (strukturierte mündliche Prüfung, Zahnärztliche Prothetik, 20 min, exakt 2 Fragen)

Bilder (synthetische didaktische Abbildungen/Röntgen-schemata) als PNG

Gesamtausgabe als MD + DOCX (Bilder eingebettet)

Workflow Overview (Systemarchitektur)
Input

transcript.txt (UTF-8, reiner Text)

Core Pipeline

LLM Structuring (Reasoning, medium)
Transcript → Kapitel/Sections → Lernziele + Bulletpoints + Key Terms + Prüfungsfragen + Bildprompts

Bildgenerierung (optional)

Pro Section: 1 didaktisches Bild (Infografik/Schema)

Zusätzlich: “Static Radiologic Materials” (PA/BW/OPG/CBCT) als synthetische Abbildungen

Didactic Blueprint Generator

Erzeugt die 1–8 Komponenten als konsistentes Nachbereitungs-Paket

Z3-SMP Generator

Erzeugt n=1 (modifizierbar) mündlich strukturierte Prothetik-SMP inklusive Erwartungshorizont/Notenbändern + Anhänge

Render & Export

Markdown je Section

Episode-weit: after_episode_package.md, z3_smp_prothetik.md

DOCX: alles zusammengeführt, Bilder eingebettet

Output-Verzeichnisse

OpenAI: out_openai/

Google: out_google/

Didactic Blueprint Mapping (1–8) – Implementiert als strukturierte JSON → MD/DOCX
Learning Objectives (1–3): competency-based, knapp

Case Summary: key findings, radiologic problem, clinical relevance, interdisciplinary aspects

Static Radiologic Materials: PA/BW/OPG/CBCT + captions + image_prompt

Key Reasoning Steps: Identify → Interpret → Prioritize → Integrate → Plan

Take-Home Messages (3–5)

Micro-Reflection Prompt (1)

Optional Self-Test (1 MCQ) (A–E, Single Best Answer)

Key Resources (max 3) + Traceability Notes (was transcriptbasiert vs. “unsicher”)

Z3-SMP (Zahnärztliche Prothetik) – Format-Compliance (dein Template)
Erzeugt pro Folge (additiv) eine Z3-SMP-Struktur mit:

Fach (fix konfigurierbar), Dauer (fix/konfigurierbar), Aufgabe Nr. Platzhalter

Prüfungsziele

Prüfungsgrundlage (NKLZ/IMPP/Leitlinie: wenn nicht im Transcript → als Placeholder + “unsicher” in Traceability)

Prüfer-Intro exakt: “Ich werde Ihnen einen Fall beschreiben … 20 Min. Zeit.”

Fallbeschreibung

Aufgaben

Genau 2 Fragen

Erwartungshorizont als 3 Notenbänder (Note 1 / Note 3 / Note 5)

Prüferkommentar + Anhänge (als Liste: Materialien/Checklisten/Röntgentextbefunde)

Tech Specs
Gemeinsame technische Eckdaten
Language/Encoding

Inputs/Outputs: UTF-8, Umlaute in Outputs erlaubt

Code: ASCII-only (konsequent eingehalten)

Artefakt-Formate

MD: pro Section + episode-wide Zusatzdokumente

PNG: 1024×1024 (konfigurierbar über Model-Optionen)

DOCX: python-docx; Bilder per add_picture eingebettet

Abhängigkeiten

python-docx>=1.1.0

pillow>=10.0.0

OpenAI-Variante: openai>=1.40.0

Google-Variante: google-genai>=1.0.0

Konfigurierbare Parameter

SMP_N_CASES (default 1)

SMP_FACH, SMP_DAUER_MIN

MAKE_SECTION_IMAGES, MAKE_STATIC_MATERIAL_IMAGES

Modellnamen (Text/Image) zentral als Konstanten

Reliability / Guardrails

Strict JSON schemas bei OpenAI (Structured Outputs) → minimiert Parsing-Fehler

Gemini: JSON über response_mime_type="application/json" + robustes JSON-Extraction-Fallback

“Unsicher”-Markierung: wenn nicht vom Transcript gedeckt (Traceability Notes / Rationale)

OpenAI Pipeline (01/2026)
Text LLM

Modell: gpt-5.2-thinking

Reasoning: reasoning.effort = "medium"

API: client.responses.create(...) mit text.format = json_schema (strict)

Image Generation

Modell: gpt-image-1

API: client.images.generate(...)

Output: base64 → PNG

Artefakte

out_openai/course_plan.json

out_openai/sections/*.md

out_openai/sections/*_openai.png

out_openai/after_episode_package.json/.md

out_openai/materials/*.png

out_openai/z3_smp_prothetik.json/.md

out_openai/course.docx

Runtime/Secrets

Env: OPENAI_API_KEY

Google Gemini Pipeline (01/2026)
Text LLM

Modell: gemini-2.5-pro

API: client.models.generate_content(...)

Config: response_mime_type="application/json"

Image Generation

Modell: gemini-3-pro-image-preview (Nano Banana Pro)
(alternativ: gemini-2.5-flash-image für schneller)

API: client.models.generate_content(...) → inline_data → Bild speichern

Artefakte

analog zu OpenAI, aber unter out_google/ und *_gemini.png

Runtime/Secrets

Env: GEMINI_API_KEY

Performance- und Skalierungsannahmen (praktisch)
Laufzeit skaliert grob mit:

Transcript-Länge (LLM Token)

Anzahl Sections (mehr Sections → mehr Bildgenerierungen + längere DOCX)

Static Materials (bis zu 4 zusätzliche Bilder)

SMP-Fälle (n)

Für sehr lange Transkripte sind zwei robuste Strategien vorgesehen:

(A) Vorab-Chunking (Transcript in Segmente) + Plan-Merge

(B) Erst grobe Sectioning-Pass, dann pro Section Detailpass (reduziert Kontextdruck)

Daten- und Compliance-Sicht
Bilder sind synthetisch (Prompts erzwingen “schematisch / educational”), keine Realpatientenbilder.

Transcript sollte de-identifiziert werden, wenn es aus klinischen Quellen stammt.

Traceability Notes dienen als Audit-Layer: klare Trennung zwischen transcriptbasiert und “unsicher/placeholder”.

Run Commands (wie konfiguriert)
# run OpenAI
export OPENAI_API_KEY="..."
python openai_course_builder_2026.py

# run Gemini
export GEMINI_API_KEY="..."
python google_course_builder_2026.py
