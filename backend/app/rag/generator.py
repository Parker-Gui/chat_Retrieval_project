"""Grounded answer generation for the hybrid GraphRAG workflow."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field

GROUNDED_SYSTEM_INSTRUCTION = (
    "You answer questions about movies using only the evidence provided. "
    "Never add facts that are absent from the evidence. Return one valid JSON "
    "object with exactly two fields: answer (string) and answers (array of strings)."
)

CLOSED_BOOK_SYSTEM_INSTRUCTION = (
    "You answer questions about movies from your own knowledge, without any "
    "retrieval. If you are unsure, give your single best guess instead of refusing. "
    "Return one valid JSON object with exactly two fields: answer (string) and "
    "answers (non-empty array of strings)."
)

CLOSED_BOOK_PROMPT = """
Answer the user's movie question from your own knowledge. No external evidence or
retrieval context is available for this request.

Rules:
1. Always give a concrete answer, even when you are not fully certain.
2. "answer" is one short sentence for a chat user.
3. "answers" is the complete list of answer entities you can name, without
   duplicates. Provide at least one entry whenever you can name an answer.
4. Return JSON exactly in this shape:
   {{"answer": "One short sentence.", "answers": ["Entity 1", "Entity 2"]}}

User question:
{question}
""".strip()

GROUNDED_PROMPT = """
Answer the user's movie question using only the evidence below.

Structured knowledge-graph answers:
{kg_answers}

Retrieved graph triples (subject | relation | object), most similar first:
{vector_context}

Rules:
1. If structured answers are present, they are the authoritative answer set.
2. If they are absent, derive the answer set only from the retrieved triples.
3. Never invent a title, person, year, genre, or language that is not in the evidence.
4. "answer" is one short sentence for a chat user; "answers" is the complete list.
5. If the evidence does not contain the answer, return an empty list and say the
   question could not be answered from the knowledge base.
6. Return JSON exactly in this shape:
   {{"answer": "One short sentence.", "answers": ["Entity 1", "Entity 2"]}}

User question:
{question}
""".strip()


class GroundedAnswer(BaseModel):
    """Structured output returned by the generator model."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1, description="One short sentence answering the question."
    )
    answers: list[str] = Field(
        description="Complete list of answer entities, without duplicates.",
    )


class AnswerParsingError(RuntimeError):
    """Raised when the model did not return a valid answer JSON object."""


def _parse_answer_content(content) -> GroundedAnswer:
    """Validate a GraphRAG LLM response without silently inventing an empty list."""
    if isinstance(content, GroundedAnswer):
        return content
    if isinstance(content, dict):
        return GroundedAnswer.model_validate(content)
    if not isinstance(content, str) or not content.strip():
        raise AnswerParsingError("DeepSeek returned blank structured-answer content")

    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as first_error:
        # Defensive fallback for providers that wrap an otherwise valid object in text.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise AnswerParsingError(
                f"DeepSeek did not return JSON: {text[:200]!r}"
            ) from first_error
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise AnswerParsingError(
                f"DeepSeek returned invalid JSON: {text[:200]!r}"
            ) from error
    try:
        return GroundedAnswer.model_validate(payload)
    except Exception as error:
        raise AnswerParsingError(
            f"DeepSeek JSON does not match GroundedAnswer: {payload!r}"
        ) from error


def build_generation_prompt(
    question: str,
    kg_answers: list[str],
    vector_context: list[dict],
) -> str:
    """Render the grounded prompt from the merged retrieval state."""
    answers_block = (
        "\n".join(f"- {value}" for value in kg_answers)
        if kg_answers
        else "(none returned)"
    )
    if vector_context:
        context_block = "\n".join(
            "- {subject} | {relation} | {object}".format(
                subject=item.get("subject", "?"),
                relation=item.get("relation", "?"),
                object=item.get("object", "?"),
            )
            for item in vector_context
        )
    else:
        context_block = "(none returned)"
    return GROUNDED_PROMPT.format(
        kg_answers=answers_block,
        vector_context=context_block,
        question=question,
    )


def generate_answer(
    llm,
    question: str,
    kg_answers: list[str],
    vector_context: list[dict],
    grounded: bool = True,
):
    """Call the model once and return a validated GroundedAnswer.

    ``grounded=False`` implements the closed-book baseline: no evidence is shown
    and the model must answer from its own knowledge, so the no-retrieval system is
    not accidentally scored on a prompt that instructs it to refuse.
    """
    if grounded:
        prompt = build_generation_prompt(question, kg_answers, vector_context)
        instruction = GROUNDED_SYSTEM_INSTRUCTION
    else:
        prompt = CLOSED_BOOK_PROMPT.format(question=question)
        instruction = CLOSED_BOOK_SYSTEM_INSTRUCTION
    response = llm.invoke(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    result = _parse_answer_content(getattr(response, "content", response))
    if not grounded and not any(value.strip() for value in result.answers):
        raise AnswerParsingError("Pure LLM returned an empty answers list")
    return result
