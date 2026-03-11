#!/usr/bin/env python3
"""Minimal image prompt guardrails for the PodcastAI hybrid pipeline."""


NEGATIVE_RULES = (
    "Keine medizinischen Bilder. "
    "Keine medizinnahe Bildgebung wie Roentgen-, CT-, MRT-, Histologie- oder Endoskopie-Optik. "
    "Keine Patienten, keine klinischen Szenen, keine OP-Situationen, keine Geraete und keine anatomisch-realistischen Darstellungen."
)

def build_image_prompt(prompt: str) -> str:
    return f"{prompt.strip()}\n\n{NEGATIVE_RULES}"
