"""
Turns the raw natural-language challenge question into a structured spec
the rest of the pipeline can execute against, using an LLM for the
open-ended parsing (attribute extraction, relation extraction, path
constraint decomposition) that a hand-written grammar would be too brittle
to cover.

This module makes ONE outbound HTTPS call per question. If your evaluation
sandbox has no internet egress, replace `call_anthropic` with a local model
call (e.g. a small fine-tuned classifier) -- the rest of the pipeline only
depends on the QuerySpec shape below, not on how it was produced.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from config import ANTHROPIC_API_URL, ANTHROPIC_MODEL

SYSTEM_PROMPT = """You convert a single navigation-challenge question into strict JSON.

Question types:
- "numerical": asks "how many X" -- answer is an integer count.
- "object_reference": asks to "find" a single unique object -- answer is one object's bounding box.
- "instruction_following": asks the robot to take/follow a path described via landmarks/constraints -- answer is a sequence of waypoints.

Output ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{
  "question_type": "numerical" | "object_reference" | "instruction_following",
  "target": {
    "category": "<object noun, singular, e.g. 'chair'>",
    "attributes": ["<adjectives, e.g. 'blue'>"],
  },
  "relations": [
    {"type": "near"|"between"|"on"|"closest_to"|"far_from", "anchors": ["<object phrase>", ...]}
  ],
  "path_constraints": [
    {"type": "near"|"avoid"|"between", "anchors": ["<object phrase>", ...], "order": <int, 1-based order this constraint applies in the path, or null if unordered>}
  ]
}

Leave "path_constraints" empty for non-instruction questions, and "relations"/"target" minimal for instruction questions.
For numerical questions, "target" is the thing being counted and "relations" captures any qualifying spatial constraint.
"""


@dataclass
class Relation:
    type: str
    anchors: List[str] = field(default_factory=list)


@dataclass
class PathConstraint:
    type: str
    anchors: List[str] = field(default_factory=list)
    order: Optional[int] = None


@dataclass
class QuerySpec:
    question_type: str
    target_category: Optional[str]
    target_attributes: List[str]
    relations: List[Relation]
    path_constraints: List[PathConstraint]
    raw_question: str


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def call_anthropic(question: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in environment. Export it before "
            "launching the ai_module, or swap call_anthropic() for a local model."
        )
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 500,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": question}],
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    raw = "".join(text_blocks)
    return json.loads(_strip_code_fence(raw))


def _fallback_parse(question: str) -> dict:
    """Rough regex-based fallback if the LLM call fails -- never leaves the
    pipeline with nothing to act on, just degrades to a much dumber parse."""
    q = question.lower()
    if q.strip().startswith("how many"):
        qtype = "numerical"
    elif q.strip().startswith(("take the path", "avoid", "go near", "follow")):
        qtype = "instruction_following"
    else:
        qtype = "object_reference"
    words = re.findall(r"[a-z]+", q)
    return {
        "question_type": qtype,
        "target": {"category": words[-1] if words else "object", "attributes": []},
        "relations": [],
        "path_constraints": [],
    }


def parse_question(question: str) -> QuerySpec:
    try:
        parsed = call_anthropic(question)
    except Exception as e:  # noqa: BLE001 -- pipeline must never crash on a parse failure
        print(f"[question_parser] LLM parse failed ({e}), using regex fallback")
        parsed = _fallback_parse(question)

    target = parsed.get("target") or {}
    relations = [Relation(**r) for r in parsed.get("relations", [])]
    path_constraints = [PathConstraint(**c) for c in parsed.get("path_constraints", [])]

    return QuerySpec(
        question_type=parsed.get("question_type", "object_reference"),
        target_category=target.get("category"),
        target_attributes=target.get("attributes", []) or [],
        relations=relations,
        path_constraints=path_constraints,
        raw_question=question,
    )
