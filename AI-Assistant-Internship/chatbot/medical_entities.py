"""Rule-based medical entity recognition for MedQuAD workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MedicalEntity:
    """A detected medical entity."""

    text: str
    label: str


class MedicalEntityRecognizer:
    """Lightweight recognizer for symptoms, diseases, tests, and treatments."""

    TERMS: dict[str, set[str]] = {
        "Disease": {
            "diabetes",
            "hypertension",
            "asthma",
            "cancer",
            "stroke",
            "infection",
            "migraine",
            "arthritis",
            "depression",
            "anxiety",
        },
        "Symptom": {
            "fever",
            "cough",
            "pain",
            "fatigue",
            "headache",
            "nausea",
            "dizziness",
            "shortness of breath",
            "blurred vision",
            "swelling",
        },
        "Treatment": {
            "medicine",
            "medication",
            "insulin",
            "antibiotic",
            "therapy",
            "surgery",
            "exercise",
            "diet",
            "vaccination",
        },
        "Test": {
            "blood test",
            "x-ray",
            "mri",
            "ct scan",
            "biopsy",
            "screening",
            "glucose test",
            "blood pressure",
        },
    }

    def extract(self, text: str) -> list[MedicalEntity]:
        """Extract simple medical entities from text."""

        normalized = text.lower()
        entities: list[MedicalEntity] = []
        seen: set[tuple[str, str]] = set()
        for label, terms in self.TERMS.items():
            for term in terms:
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, normalized):
                    key = (term, label)
                    if key not in seen:
                        entities.append(MedicalEntity(text=term, label=label))
                        seen.add(key)
        return entities

    def format_entities(self, entities: list[MedicalEntity]) -> str:
        """Format entities for prompts and UI output."""

        if not entities:
            return "No explicit medical entities detected."
        return ", ".join(f"{entity.text} ({entity.label})" for entity in entities)
