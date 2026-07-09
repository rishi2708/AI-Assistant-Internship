"""Evidence-based multimodal reasoning across text, OCR, image observations, and memory."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MultimodalEvidence:
    """Evidence item extracted from text, OCR, image model output, or conversation memory."""

    source: str
    content: str
    confidence: float


@dataclass(frozen=True)
class MultimodalReasoningResult:
    """Validated multimodal answer with traceable evidence and ambiguity handling."""

    answer: str
    evidence: list[MultimodalEvidence]
    ambiguities: list[str] = field(default_factory=list)
    validation: str = "validated"
    decision: str = "answer"


class MultimodalReasoner:
    """Combine text and image signals before asking or returning a final answer."""

    UNCERTAINTY_TERMS = {"maybe", "possibly", "unclear", "unknown", "ambiguous", "not sure"}

    def build_evidence(
        self,
        user_text: str,
        image_observations: str = "",
        ocr_text: str = "",
        prior_context: list[str] | None = None,
    ) -> list[MultimodalEvidence]:
        """Extract evidence from all available modalities."""

        evidence: list[MultimodalEvidence] = []
        if user_text.strip():
            evidence.append(MultimodalEvidence("user_text", user_text.strip(), 0.95))
        if image_observations.strip():
            evidence.append(MultimodalEvidence("image_observation", image_observations.strip(), 0.82))
        if ocr_text.strip():
            evidence.append(MultimodalEvidence("ocr", ocr_text.strip(), 0.78))
        for item in prior_context or []:
            if item.strip():
                evidence.append(MultimodalEvidence("conversation_memory", item.strip(), 0.65))
        return evidence

    def find_ambiguities(self, evidence: list[MultimodalEvidence]) -> list[str]:
        """Identify ambiguity and weak-evidence conditions."""

        joined = " ".join(item.content.lower() for item in evidence)
        ambiguities: list[str] = []
        if len(evidence) < 2:
            ambiguities.append("Only one modality is available; answer should state the evidence limitation.")
        if any(term in joined for term in self.UNCERTAINTY_TERMS):
            ambiguities.append("Input or model observation contains uncertainty terms.")
        if self._has_conflict(evidence):
            ambiguities.append("Text and image/OCR evidence appear to conflict; ask a clarifying question.")
        return ambiguities

    def validate_answer(self, answer: str, evidence: list[MultimodalEvidence]) -> str:
        """Validate that the answer is grounded in available evidence."""

        answer_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", answer.lower()))
        evidence_terms = set(
            re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", " ".join(item.content.lower() for item in evidence))
        )
        overlap = answer_terms & evidence_terms
        if not answer.strip():
            return "invalid: empty answer"
        if len(overlap) < 2 and evidence_terms:
            return "needs_revision: answer has low overlap with extracted evidence"
        return "validated: answer is grounded in multimodal evidence"

    def reason(
        self,
        user_text: str,
        generated_answer: str,
        image_observations: str = "",
        ocr_text: str = "",
        prior_context: list[str] | None = None,
    ) -> MultimodalReasoningResult:
        """Return a final decision with evidence, ambiguity handling, and validation."""

        evidence = self.build_evidence(user_text, image_observations, ocr_text, prior_context)
        ambiguities = self.find_ambiguities(evidence)
        validation = self.validate_answer(generated_answer, evidence)
        decision = "clarify" if any("conflict" in item.lower() for item in ambiguities) else "answer"
        evidence_text = "\n".join(f"- {item.source}: {item.content}" for item in evidence)
        ambiguity_text = "\n".join(f"- {item}" for item in ambiguities) or "- None detected"
        final_answer = (
            f"{generated_answer}\n\n"
            "Evidence used:\n"
            f"{evidence_text}\n\n"
            "Ambiguity handling:\n"
            f"{ambiguity_text}\n\n"
            f"Response validation: {validation}\n"
            f"Decision: {decision}"
        )
        return MultimodalReasoningResult(final_answer, evidence, ambiguities, validation, decision)

    @staticmethod
    def _has_conflict(evidence: list[MultimodalEvidence]) -> bool:
        """Detect simple contradiction markers between text and visual evidence."""

        text = " ".join(item.content.lower() for item in evidence if item.source == "user_text")
        visual = " ".join(
            item.content.lower() for item in evidence if item.source in {"image_observation", "ocr"}
        )
        if not text or not visual:
            return False
        pairs = [("red", "blue"), ("cat", "dog"), ("yes", "no"), ("safe", "unsafe")]
        return any((left in text and right in visual) or (right in text and left in visual) for left, right in pairs)
