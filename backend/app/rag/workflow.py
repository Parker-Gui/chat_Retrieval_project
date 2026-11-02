"""Answering pipelines for the midterm comparison (and the final hybrid one).

Every system shares the same generation step and the same answer-set policy, so the
only thing that differs between them is which retrieval evidence is supplied:

``pure_llm``      no evidence at all
``vector_rag``    top-k graph triples from the Neo4j vector index
``kg_retrieval``  the structured answer set from Text2Cypher
``hybrid``        both, for the final-stage experiment

The pipeline is a deterministic sequence of node functions over ``HybridState``.
LangGraph could not be installed on the lab machine (PyPI unreachable), so the
nodes are plain functions; moving this graph to LangGraph would be mechanical.
"""

from __future__ import annotations

import time

from app.rag.generator import generate_answer
from app.rag.state import HybridState


SUPPORTED_SYSTEMS = ("pure_llm", "vector_rag", "kg_retrieval", "hybrid")


class AnswerPipeline:
    """Run one answering system: optional retrievers, then grounded generation."""

    def __init__(
        self,
        llm,
        system: str = "pure_llm",
        kg_retriever=None,
        vector_retriever=None,
        top_k: int = 5,
    ) -> None:
        if system not in SUPPORTED_SYSTEMS:
            raise ValueError(f"Unknown system {system!r}; expected one of {SUPPORTED_SYSTEMS}")
        needs_kg = system in ("kg_retrieval", "hybrid")
        needs_vector = system in ("vector_rag", "hybrid")
        if needs_kg and kg_retriever is None:
            raise ValueError(f"System {system!r} requires a kg_retriever")
        if needs_vector and vector_retriever is None:
            raise ValueError(f"System {system!r} requires a vector_retriever")
        self.llm = llm
        self.system = system
        self.kg_retriever = kg_retriever
        self.vector_retriever = vector_retriever
        self.top_k = top_k

    # --- nodes -----------------------------------------------------------

    def kg_node(self, state: HybridState) -> HybridState:
        if self.kg_retriever is None:
            return state
        started = time.perf_counter()
        try:
            result = self.kg_retriever.search(query_text=state.question)
            answers: list[str] = []
            for item in result.items:
                answers.extend((item.metadata or {}).get("answers", []))
            state.kg_answers = list(
                dict.fromkeys(str(a) for a in answers if str(a).strip())
            )
            state.kg_cypher = str((result.metadata or {}).get("cypher", "") or "")
            state.used_kg = True
        except Exception as error:  # keep the pipeline alive, record provenance
            state.add_error("kg_retrieval", error)
        state.timings_ms["kg_ms"] = (time.perf_counter() - started) * 1_000
        return state

    def vector_node(self, state: HybridState) -> HybridState:
        if self.vector_retriever is None:
            return state
        started = time.perf_counter()
        try:
            result = self.vector_retriever.search(
                query_text=state.question, top_k=self.top_k
            )
            state.vector_context = [
                {"text": item.content, **(item.metadata or {})} for item in result.items
            ]
            state.used_vector = True
        except Exception as error:
            state.add_error("vector_retrieval", error)
        state.timings_ms["vector_ms"] = (time.perf_counter() - started) * 1_000
        return state

    def merge_node(self, state: HybridState) -> HybridState:
        started = time.perf_counter()
        if self.system != "pure_llm" and not state.kg_answers and not state.vector_context:
            state.add_error("merge", RuntimeError("no retrieval evidence available"))
        state.timings_ms["merge_ms"] = (time.perf_counter() - started) * 1_000
        return state

    def generate_node(self, state: HybridState) -> HybridState:
        started = time.perf_counter()
        try:
            generated = generate_answer(
                self.llm,
                state.question,
                state.kg_answers,
                state.vector_context,
                grounded=self.system != "pure_llm",
            )
            state.final_answer = generated.answer
            if self.system in ("kg_retrieval", "hybrid") and state.kg_answers:
                # Structured KG answers stay authoritative so this score remains
                # comparable with the Step 6 Text2Cypher measurement.
                state.answer_set = list(state.kg_answers)
            else:
                state.answer_set = list(
                    dict.fromkeys(a.strip() for a in generated.answers if a.strip())
                )
        except Exception as error:
            state.add_error("generation", error)
            if not state.final_answer:
                state.final_answer = (
                    "I could not answer this question from the movie knowledge base."
                )
            state.answer_set = list(state.kg_answers)
        state.timings_ms["generation_ms"] = (time.perf_counter() - started) * 1_000
        return state

    # --- graph -----------------------------------------------------------

    def run(self, question: str) -> HybridState:
        state = HybridState(question=question)
        for node in (self.kg_node, self.vector_node, self.merge_node, self.generate_node):
            state = node(state)
        state.timings_ms["total_ms"] = sum(state.timings_ms.values())
        return state


# Final-stage name kept for the hybrid experiment.
HybridGraphRAG = AnswerPipeline
