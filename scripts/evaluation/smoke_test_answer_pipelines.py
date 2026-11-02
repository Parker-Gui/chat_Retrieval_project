"""No-cost smoke test for the answering pipelines.

Uses stub retrievers and a stub LLM, so it verifies node wiring, the answer-set
policy of every system, and error capture without calling DeepSeek or Neo4j.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem  # noqa: E402

from app.rag.generator import (  # noqa: E402
    AnswerParsingError,
    GroundedAnswer,
    generate_answer,
)
from app.rag.workflow import AnswerPipeline  # noqa: E402


class StubKGRetriever:
    def __init__(self, answers, cypher="MATCH ... RETURN answer.name AS answer", fail=False):
        self.answers = answers
        self.cypher = cypher
        self.fail = fail

    def search(self, query_text):
        if self.fail:
            raise RuntimeError("simulated Cypher failure")
        return RetrieverResult(
            items=[
                RetrieverResultItem(
                    content=" | ".join(self.answers), metadata={"answers": list(self.answers)}
                )
            ],
            metadata={"cypher": self.cypher},
        )


class StubVectorRetriever:
    def __init__(self, triples, fail=False):
        self.triples = triples
        self.fail = fail

    def search(self, query_text, top_k=5):
        if self.fail:
            raise RuntimeError("simulated vector failure")
        return RetrieverResult(
            items=[
                RetrieverResultItem(
                    content=f"{s} | {r} | {o}",
                    metadata={"subject": s, "relation": r, "object": o, "score": 0.9 - 0.1 * i},
                )
                for i, (s, r, o) in enumerate(self.triples)
            ]
        )


class StubLLM:
    def __init__(self, answer="ok", answers=None, fail=False):
        self.answer = answer
        self.answers = answers or []
        self.fail = fail
        self.calls = 0

    def invoke(self, prompt, system_instruction=None, response_format=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated generation failure")
        return SimpleNamespace(content=GroundedAnswer(answer=self.answer, answers=self.answers))


class JsonStubLLM:
    """Exercise the real string-JSON parsing path and capture invocation shape."""

    def __init__(self, content):
        self.content = content
        self.input = None
        self.response_format = None

    def invoke(self, input, response_format=None, **_kwargs):
        self.input = input
        self.response_format = response_format
        return SimpleNamespace(content=self.content)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"{'PASS' if condition else 'FAIL'} {label}{(' :: ' + detail) if detail else ''}")
    return condition


def main() -> int:
    failures = 0

    # 1. Pure LLM sees no evidence and answers from the model's own list.
    llm = StubLLM(answer="Forrest Gump.", answers=["Forrest Gump"])
    state = AnswerPipeline(llm=llm, system="pure_llm").run("which movie won Best Picture")
    failures += not check("pure_llm extracts its own answers", state.answer_set == ["Forrest Gump"], str(state.answer_set))
    failures += not check("pure_llm uses exactly one model call", llm.calls == 1, str(llm.calls))
    failures += not check("pure_llm records no evidence", not state.used_kg and not state.used_vector)

    # 2. Vector RAG answers only from retrieved triples.
    state = AnswerPipeline(
        llm=StubLLM(answer="Drama.", answers=["Drama"]),
        system="vector_rag",
        vector_retriever=StubVectorRetriever([("Forrest Gump", "HAS_GENRE", "Drama")]),
    ).run("what genre is Forrest Gump")
    failures += not check("vector_rag answers from the generator", state.answer_set == ["Drama"], str(state.answer_set))
    failures += not check("vector_rag records its context", len(state.vector_context) == 1)
    failures += not check("vector_rag skips the KG", state.kg_answers == [])

    # 3. KG retrieval keeps the structured answer set authoritative.
    state = AnswerPipeline(
        llm=StubLLM(answer="Forrest Gump.", answers=["wrong"]),
        system="kg_retrieval",
        kg_retriever=StubKGRetriever(["Forrest Gump"]),
    ).run("what did Tom Hanks star in")
    failures += not check("kg_retrieval keeps KG answers over model text", state.answer_set == ["Forrest Gump"], str(state.answer_set))
    failures += not check("kg_retrieval keeps the Cypher", "RETURN answer.name AS answer" in state.kg_cypher)
    failures += not check("kg_retrieval still produces a sentence", state.final_answer == "Forrest Gump.")

    # 4. Hybrid merges both retrievers.
    state = AnswerPipeline(
        llm=StubLLM(answer="Forrest Gump.", answers=["Forrest Gump"]),
        system="hybrid",
        kg_retriever=StubKGRetriever(["Forrest Gump"]),
        vector_retriever=StubVectorRetriever([("Forrest Gump", "HAS_GENRE", "Drama")]),
    ).run("which movie did Tom Hanks win an Oscar for")
    failures += not check("hybrid merges both sources", state.used_kg and state.used_vector)
    failures += not check("hybrid prefers KG answers", state.answer_set == ["Forrest Gump"])
    failures += not check("hybrid records per-node timings", {"kg_ms", "vector_ms", "generation_ms"} <= set(state.timings_ms))

    # 5. KG failure in a KG system falls back to the generator's list.
    state = AnswerPipeline(
        llm=StubLLM(answer="Cornelia Funke.", answers=["Cornelia Funke"]),
        system="kg_retrieval",
        kg_retriever=StubKGRetriever([], fail=True),
    ).run("who wrote Inkheart")
    failures += not check("kg failure is recorded", any("kg_retrieval" in e for e in state.retrieval_errors), str(state.retrieval_errors))
    failures += not check("generator supplies the fallback", state.answer_set == ["Cornelia Funke"], str(state.answer_set))

    # 6. Generation failure never loses retrieved answers.
    state = AnswerPipeline(
        llm=StubLLM(fail=True),
        system="kg_retrieval",
        kg_retriever=StubKGRetriever(["Big"]),
    ).run("what did Tom Hanks star in")
    failures += not check("generation failure is recorded", any("generation" in e for e in state.retrieval_errors))
    failures += not check("KG answers survive generation failure", state.answer_set == ["Big"], str(state.answer_set))
    failures += not check("safe reply text is provided", "could not answer" in state.final_answer.lower(), state.final_answer)

    # 7. Misconfiguration is rejected early.
    try:
        AnswerPipeline(llm=StubLLM(), system="vector_rag")
        failures += not check("vector_rag without a retriever is rejected", False)
    except ValueError:
        failures += not check("vector_rag without a retriever is rejected", True)

    # 8. DeepSeek's supported JSON-object path is used and parsed strictly.
    json_llm = JsonStubLLM(
        '{"answer":"Forrest Gump.","answers":["Forrest Gump"]}'
    )
    parsed = generate_answer(json_llm, "what movie", [], [], grounded=False)
    failures += not check(
        "generator parses JSON content", parsed.answers == ["Forrest Gump"]
    )
    failures += not check(
        "generator invokes the V2 message-list path",
        isinstance(json_llm.input, list)
        and [message.get("role") for message in json_llm.input] == ["system", "user"],
        str(json_llm.input),
    )
    failures += not check(
        "generator requests DeepSeek JSON-object mode",
        json_llm.response_format == {"type": "json_object"},
        str(json_llm.response_format),
    )

    # 9. Invalid or silently empty closed-book outputs must not become fake zeros.
    try:
        generate_answer(JsonStubLLM("not json"), "what movie", [], [], grounded=False)
        failures += not check("invalid JSON is rejected", False)
    except AnswerParsingError:
        failures += not check("invalid JSON is rejected", True)
    try:
        generate_answer(
            JsonStubLLM('{"answer":"I do not know.","answers":[]}'),
            "what movie",
            [],
            [],
            grounded=False,
        )
        failures += not check("empty pure-LLM answer set is rejected", False)
    except AnswerParsingError:
        failures += not check("empty pure-LLM answer set is rejected", True)

    print("SMOKE_RESULT", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
