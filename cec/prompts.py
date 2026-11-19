"""Prompt template for grounded CEC answers."""

from __future__ import annotations

from collections.abc import Iterable

from cec.retrieval import Match

ANSWER_TEMPLATE = """\
You are a Canadian Electrical Code assistant.

Answer the user's question using ONLY the CEC material provided below.

IMPORTANT RULES:

1. Do not use outside electrical knowledge.
2. Do not invent requirements, numbers, exceptions, definitions, or rule references.
3. Do not infer that a retrieved rule contains a requirement unless that requirement is explicitly present in the supplied text.
4. Prefer the rule that directly answers the question over rules that are merely related.
5. When describing an exception, state only what the supplied CEC text actually says.
6. Identify the specific CEC rule number that supports each important requirement.
7. If the supplied material does not contain enough information to answer confidently, say so.
8. Do not cite a rule simply because it was retrieved. Cite it only if its supplied text supports the statement.

Format the answer in Markdown.

CEC MATERIAL:

{context}

USER QUESTION:

{question}
"""


def format_context(matches: Iterable[Match]) -> str:
    return "\n\n".join(
        f"Rule {m.rule.rule} — {m.rule.title}\n\n{m.rule.text}" for m in matches
    )


def build_prompt(question: str, matches: Iterable[Match]) -> str:
    return ANSWER_TEMPLATE.format(context=format_context(matches), question=question.strip())
