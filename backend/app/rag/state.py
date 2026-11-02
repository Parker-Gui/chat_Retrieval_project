"""Shared state for the hybrid graph retrieval and generation workflow."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HybridState:
    """Mutable state passed between workflow nodes.

    The field set mirrors what a LangGraph ``StateGraph`` would declare: every
    node reads this object and writes back the fields it owns.
    """

    question: str
    vector_context: list[dict] = field(default_factory=list)
    kg_answers: list[str] = field(default_factory=list)
    kg_cypher: str = ""
    retrieval_errors: list[str] = field(default_factory=list)
    final_answer: str = ""
    answer_set: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    used_kg: bool = False
    used_vector: bool = False

    def add_error(self, stage: str, error: Exception) -> None:
        self.retrieval_errors.append(f"{stage}: {type(error).__name__}: {error}")

    def as_record(self) -> dict:
        """Serializable view used by the evaluator and the API debug payload."""
        return {
            "question": self.question,
            "vector_context": self.vector_context,
            "kg_answers": self.kg_answers,
            "kg_cypher": self.kg_cypher,
            "retrieval_errors": self.retrieval_errors,
            "final_answer": self.final_answer,
            "answer_set": self.answer_set,
            "timings_ms": self.timings_ms,
            "used_kg": self.used_kg,
            "used_vector": self.used_vector,
        }
