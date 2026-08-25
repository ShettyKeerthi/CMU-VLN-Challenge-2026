"""
Turns the raw natural-language challenge question into a structured spec
the rest of the pipeline can execute against, using an LLM for the
open-ended parsing (attribute extraction, relation extraction, path
constraint decomposition) that a hand-written grammar would be too brittle
to cover.

This module makes ONE outbound HTTPS call per question. If your evaluation
sandbox has no internet egress, replace `call_anthropic` with a local model
call -- the rest of the pipeline only depends on the QuerySpec shape below.

SCHEMA NOTE -- why every PathConstraint carries its own target.
An instruction statement is a SEQUENCE of legs and each leg has its own
referring expression. "Go to the potted plant closest to the pyramid candle
holder and stop at the vase between the TV and the door" has two destinations:

    leg 1  goto  potted plant  (closest_to: pyramid candle holder)
    leg 2  goto  vase          (between: tv, door)        <- final

A single top-level `target` cannot hold both, so a flat schema silently keeps
only the anchors and discards both destinations.

GOTO vs VIA -- the distinction is whether the robot STOPS.
  "go near the lamp"            -> goto. A destination; the robot arrives.
  "take the path near the lamp" -> via.  A route hint; the robot passes by.
Only phrasings about the PATH are via. This matters because via uses a looser
standoff and does not count as the final destination.

Backend: Groq's OpenAI-compatible chat/completions endpoint. Config names are
kept as ANTHROPIC_* for continuity with the rest of the pipeline:

    ANTHROPIC_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    ANTHROPIC_MODEL   = "openai/gpt-oss-120b"    # full id, org prefix included

and export ANTHROPIC_API_KEY with your gsk_... key.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from config import ANTHROPIC_API_URL, ANTHROPIC_MODEL, ANTHROPIC_API_KEY

# Ops the executor understands. Anything else is a parse error caught here,
# not a surprise during waypoint generation.
VALID_OPS = {"goto", "via", "between", "avoid_between", "avoid_near"}

# Relations the scene graph can score. Keep in sync with scene_graph._SIMPLE
# and scene_graph._SUPERLATIVE.
VALID_RELATIONS = {
    "near", "between", "on", "above", "below", "supports",
    "closest_to", "farthest_from",
}

# Relation words the model sometimes puts in the `type` slot where an op
# belongs. Seeing one there means the destination object was dropped.
_RELATION_AS_OP = VALID_RELATIONS - {"between"}


SYSTEM_PROMPT = """You convert a single navigation-challenge question into strict JSON.

Question types:
- "numerical": asks "how many X" -- answer is an integer count.
- "object_reference": asks to "find" a single unique object -- answer is one object's bounding box.
- "instruction_following": asks the robot to follow a path described via landmarks -- answer is a sequence of waypoints.

Output ONLY valid JSON (no markdown fences, no commentary):

{
  "question_type": "numerical" | "object_reference" | "instruction_following",
  "target": {
    "category": "<object noun, singular>",
    "attributes": ["<adjectives, e.g. 'blue'>"],
    "relation": "<relation name or null>",
    "anchors": ["<object phrase>", ...]
  },
  "relations": [
    {"type": "<relation name>", "anchors": ["<object phrase>", ...]}
  ],
  "path_constraints": [
    {"type": "goto" | "via" | "between" | "avoid_between" | "avoid_near",
     "target": {"category": "<noun>", "attributes": [...],
                "relation": "<relation name or null>", "anchors": [...]},
     "anchors": ["<object phrase>", ...],
     "order": <int, 1-based>}
  ]
}

Relation names: near, between, on, above, below, supports, closest_to, farthest_from.
  "the vase ON the table"                -> target vase,    relation "on",       anchors ["table"]
  "the table WITH the vase on it"        -> target table,   relation "supports", anchors ["vase"]
  "the cabinet WITH a picture ABOVE it"  -> target cabinet, relation "below",    anchors ["picture"]

RULES FOR instruction_following -- read these carefully:
- Each place the robot must go is a SEPARATE path_constraints entry, in order.

- "goto" is a DESTINATION: the robot travels there and stops.
  Use it for "go to X", "go near X", "stop at X", "stop by X", "then to X",
  "head to X", "move to X". Note "go near X" is a destination, NOT a via.

- "via" is a ROUTE HINT: the robot passes X without stopping. Use it ONLY when
  the phrasing is about the PATH itself -- "take the path near X",
  "pass by X", "passing X", "going past X". If the verb is go/head/move/stop
  rather than a phrase about the path, it is "goto".

- "take the path between A and B" -> type "between", A and B in the top-level
  "anchors", target null. The robot passes through a gap, it does not stop.
- "avoiding the path between A and B" -> type "avoid_between", anchors A and B.
- "avoid the path near X" -> type "avoid_near", X in "target".

- "type" is ALWAYS one of the five ops. NEVER put a relation name
  (near, closest_to, on, below) in "type" -- relations belong in
  "target.relation".
- NEVER omit "target" for goto/via/avoid_near. The destination object is the
  most important field in the whole output; without it the robot has nowhere
  to drive and the leg scores zero.
- The modifier belongs to the destination, not to the leg. In "go to the
  potted plant closest to the candle holder", the leg is goto, the target is
  the potted plant, and closest_to/candle holder go inside that target.
- Categories are SINGULAR nouns. "the round tables" -> category "round table".

Leave "path_constraints" empty for non-instruction questions.
For numerical questions "target" is the thing being counted and "relations"
holds any qualifying spatial constraint.
"""

# Worked examples as prior turns. These override the prose rules when they
# conflict, so they must be exactly right -- an earlier version mapped
# "go near the lamp" to via here and the model copied it every time.
_FEWSHOT = [
    ("Go to the potted plant closest to the pyramid candle holder and stop at "
     "the vase between the TV and the door.",
     {
         "question_type": "instruction_following",
         "target": {"category": None, "attributes": [], "relation": None, "anchors": []},
         "relations": [],
         "path_constraints": [
             {"type": "goto", "order": 1, "anchors": [],
              "target": {"category": "potted plant", "attributes": [],
                         "relation": "closest_to", "anchors": ["pyramid candle holder"]}},
             {"type": "goto", "order": 2, "anchors": [],
              "target": {"category": "vase", "attributes": [],
                         "relation": "between", "anchors": ["tv", "door"]}},
         ],
     }),
    # "go near the lamp" is a DESTINATION -> goto.
    # "take the path between ..." is a gap traversal -> between.
    ("First, go near the lamp closest to the black chair, then take the path "
     "between the sofa and the round tables, and stop at the cabinet with a "
     "picture above it.",
     {
         "question_type": "instruction_following",
         "target": {"category": None, "attributes": [], "relation": None, "anchors": []},
         "relations": [],
         "path_constraints": [
             {"type": "goto", "order": 1, "anchors": [],
              "target": {"category": "lamp", "attributes": [],
                         "relation": "closest_to", "anchors": ["black chair"]}},
             {"type": "between", "order": 2, "anchors": ["sofa", "round table"],
              "target": None},
             {"type": "goto", "order": 3, "anchors": [],
              "target": {"category": "cabinet", "attributes": [],
                         "relation": "below", "anchors": ["picture"]}},
         ],
     }),
    # The only shape that produces a via: a phrase about the PATH.
    ("Take the path near the bookshelf, avoiding the path between the bed and "
     "the nightstand, and stop at the desk.",
     {
         "question_type": "instruction_following",
         "target": {"category": None, "attributes": [], "relation": None, "anchors": []},
         "relations": [],
         "path_constraints": [
             {"type": "via", "order": 1, "anchors": [],
              "target": {"category": "bookshelf", "attributes": [],
                         "relation": None, "anchors": []}},
             {"type": "avoid_between", "order": 2, "anchors": ["bed", "nightstand"],
              "target": None},
             {"type": "goto", "order": 3, "anchors": [],
              "target": {"category": "desk", "attributes": [],
                         "relation": None, "anchors": []}},
         ],
     }),
]

# Phrase-level override applied after the LLM replies. The goto/via boundary is
# lexical and deterministic, so there is no reason to leave it to the model.
_VIA_PATTERNS = re.compile(
    r"\b(take the path (near|by|past)|pass(ing)? (by|near|past)|go(ing)? past)\b")
_GOTO_PATTERNS = re.compile(
    r"\b(go (to|near)|stop (at|by)|head (to|towards)|move to|then to|end (at|near))\b")


@dataclass
class ObjectRef:
    """A referring expression: what to find, and how it relates to anchors."""
    category: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    relation: Optional[str] = None
    anchors: List[str] = field(default_factory=list)

    def __repr__(self):
        s = " ".join(filter(None, [" ".join(self.attributes), self.category or "?"]))
        if self.relation and self.anchors:
            s += f" [{self.relation}: {', '.join(self.anchors)}]"
        return s


@dataclass
class Relation:
    type: str
    anchors: List[str] = field(default_factory=list)


@dataclass
class PathConstraint:
    type: str                                           # one of VALID_OPS
    target: Optional[ObjectRef] = None                  # goto / via / avoid_near
    anchors: List[str] = field(default_factory=list)    # between / avoid_between
    order: Optional[int] = None

    def __repr__(self):
        body = str(self.target) if self.target is not None else " | ".join(self.anchors)
        return f"{self.order}. {self.type}({body})"


@dataclass
class QuerySpec:
    question_type: str
    target_category: Optional[str]
    target_attributes: List[str]
    relations: List[Relation]
    path_constraints: List[PathConstraint]
    raw_question: str

    def targets(self) -> List[ObjectRef]:
        """Every destination, in travel order. The last is the final stop."""
        return [c.target for c in self.path_constraints
                if c.type in ("goto", "via") and c.target is not None]

    def final_target(self) -> Optional[ObjectRef]:
        for c in reversed(self.path_constraints):
            if c.type == "goto" and c.target is not None:
                return c.target
        t = self.targets()
        return t[-1] if t else None

    def required_categories(self) -> List[str]:
        """Every category this question depends on -- targets, relation anchors,
        AND anchors nested inside each leg's target. Miss the last group and
        the detector is never prompted for them, so the relation silently
        scores zero on every candidate."""
        cats = set()
        if self.target_category:
            cats.add(self.target_category)
        for rel in self.relations:
            cats.update(rel.anchors)
        for c in self.path_constraints:
            cats.update(c.anchors)
            if c.target is not None:
                if c.target.category:
                    cats.add(c.target.category)
                cats.update(c.target.anchors)
        return sorted(c for c in cats if c)


# ------------------------------------------------------------- normalisation

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _clean(s) -> Optional[str]:
    """Lowercase, strip leading articles, singularise a trailing plural."""
    if not s:
        return None
    s = re.sub(r"^(a|an|the)\s+", "", str(s).strip().lower())
    words = s.split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") \
            and not words[-1].endswith(("ss", "us", "is")):
        words[-1] = words[-1][:-1]          # "round tables" -> "round table"
    return " ".join(words) or None


_REL_ALIASES = {
    "on_the": "on", "in": "near", "at": "near", "next_to": "near",
    "beside": "near", "adjacent_to": "near", "close_to": "closest_to",
    "nearest": "closest_to", "nearest_to": "closest_to",
    "far_from": "farthest_from", "furthest_from": "farthest_from",
    "under": "below", "underneath": "below", "beneath": "below",
    "with": "supports", "has": "supports", "holding": "supports",
    "on_top_of": "on", "over": "above",
}


def _norm_relation(r) -> Optional[str]:
    if not r:
        return None
    r = str(r).strip().lower().replace("-", "_").replace(" ", "_")
    r = _REL_ALIASES.get(r, r)
    return r if r in VALID_RELATIONS else None


# ---------------------------------------------------------------------- api

def call_anthropic(question: str) -> dict:
    """Name kept for continuity; the backend is Groq's OpenAI-compatible API."""
    #api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GROQ_API_KEY")
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in environment. Export it before "
            "launching the ai_module, or swap call_anthropic() for a local model."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in _FEWSHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": json.dumps(a)})
    messages.append({"role": "user", "content": question})

    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Default python-requests UA trips Cloudflare bot protection (1010).
            "User-Agent": "curl/8.5.0",
        },
        json={
            "model": ANTHROPIC_MODEL,
            # Generous: reasoning models spend tokens before emitting any JSON,
            # and a tight cap truncates mid-object, which looks like a parse bug.
            "max_tokens": 2000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": messages,
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        # Groq names the problem in the body (bad model id, unsupported
        # response_format); raise_for_status alone hides it.
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")

    raw = _strip_code_fence(resp.json()["choices"][0]["message"]["content"] or "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e > s:
            return json.loads(raw[s:e + 1])
        raise


# ------------------------------------------------------------------ parsing

def _parse_ref(d) -> Optional[ObjectRef]:
    if not isinstance(d, dict):
        return None
    cat = _clean(d.get("category"))
    anchors = [a for a in (_clean(x) for x in (d.get("anchors") or [])) if a]
    rel = _norm_relation(d.get("relation"))
    if not cat and not anchors:
        return None
    return ObjectRef(
        category=cat,
        attributes=[str(a).strip().lower() for a in (d.get("attributes") or []) if a],
        relation=rel,
        anchors=anchors,
    )


def _fix_goto_via(constraints: List[PathConstraint], question: str) -> None:
    """Correct the goto/via choice from the question text.

    The boundary is lexical: "go near X" is a destination, "take the path near
    X" is a route hint. Deterministic, so there is no reason to depend on the
    model getting it right -- and getting it wrong changes the standoff and
    which leg counts as final.
    """
    q = question.lower()
    has_via_phrase = bool(_VIA_PATTERNS.search(q))

    for c in constraints:
        if c.type not in ("goto", "via") or c.target is None:
            continue
        cat = c.target.category
        if not cat:
            continue

        # Find the clause mentioning this target and inspect the verb before it.
        m = re.search(r"([^,.;]*\b" + re.escape(cat.split()[-1]) + r"\b[^,.;]*)", q)
        clause = m.group(1) if m else ""

        if _VIA_PATTERNS.search(clause):
            want = "via"
        elif _GOTO_PATTERNS.search(clause):
            want = "goto"
        else:
            want = "via" if (has_via_phrase and c.type == "via") else "goto"

        if want != c.type:
            print(f"[question_parser] leg {c.order}: {c.type} -> {want} "
                  f"(phrasing in {clause.strip()!r})")
            c.type = want


def _parse_constraint(c) -> Optional[PathConstraint]:
    """One path_constraints entry -> PathConstraint, repairing what it can."""
    if not isinstance(c, dict):
        return None

    op = str(c.get("type") or "goto").strip().lower().replace(" ", "_").replace("-", "_")
    op = {"avoid": "avoid_near", "go_to": "goto", "stop_at": "goto",
          "go_near": "goto", "head_to": "goto", "move_to": "goto",
          "pass_by": "via", "near_path": "via", "path_near": "via"}.get(op, op)

    target = _parse_ref(c.get("target"))
    anchors = [a for a in (_clean(x) for x in (c.get("anchors") or [])) if a]

    # REPAIR: a relation in the op slot means the destination was dropped and
    # its anchors surfaced at the top level. Rebuild a goto so ordering
    # survives even though the category is missing.
    if op in _RELATION_AS_OP:
        print(f"[question_parser] repairing leg: type={op!r} is a relation, not an op "
              f"(anchors={anchors}) -- destination object was lost")
        if target is None and anchors:
            target = ObjectRef(category=None, relation=op, anchors=anchors)
        elif target is not None and target.relation is None:
            target.relation, target.anchors = op, (target.anchors or anchors)
        op, anchors = "goto", []

    if op not in VALID_OPS:
        print(f"[question_parser] unknown op {op!r}, treating as goto")
        op = "goto"

    if op in ("between", "avoid_between"):
        if len(anchors) < 2 and target is not None and len(target.anchors) >= 2:
            anchors, target = target.anchors[:], None
        if len(anchors) < 2:
            print(f"[question_parser] {op!r} needs 2 anchors, got {anchors} -- dropping leg")
            return None

    if op in ("goto", "via", "avoid_near"):
        if target is None:
            print(f"[question_parser] {op!r} has no target object -- dropping leg")
            return None
        if not target.category:
            print(f"[question_parser] {op!r} target has no category ({target}) -- "
                  f"it will not resolve against the scene graph")

    try:
        order = int(c.get("order")) if c.get("order") is not None else None
    except (TypeError, ValueError):
        order = None

    return PathConstraint(type=op, target=target, anchors=anchors, order=order)


def _fallback_parse(question: str) -> dict:
    """Regex fallback when the LLM call fails. Crude, but it still produces
    ordered legs WITH targets, which is what the executor needs."""
    q = question.strip().lower().rstrip(".")

    if q.startswith("how many"):
        words = re.findall(r"[a-z]+", q)
        return {"question_type": "numerical",
                "target": {"category": words[-1] if words else "object", "attributes": []},
                "relations": [], "path_constraints": []}

    if not re.search(r"\b(go to|go near|stop at|stop by|take the path|then to|pass by|avoid)\b", q):
        words = re.findall(r"[a-z]+", q)
        return {"question_type": "object_reference",
                "target": {"category": words[-1] if words else "object", "attributes": []},
                "relations": [], "path_constraints": []}

    legs, order = [], 0
    for seg in re.split(r",|\band then\b|\bthen\b|\band\b", q):
        seg = seg.strip()
        if not seg:
            continue

        m = re.search(r"path between (?:the )?(.+?) and (?:the )?(.+)$", seg)
        if m:
            order += 1
            legs.append({"type": "between", "order": order, "target": None,
                         "anchors": [m.group(1), m.group(2)]})
            continue

        m = re.search(r"(?:go to|go near|stop at|stop by|pass by|take the path near|to)\s+(?:the )?(.+)$", seg)
        if m:
            phrase = m.group(1)
            op = "via" if _VIA_PATTERNS.search(seg) else "goto"
            rel, anchors = None, []
            m2 = re.search(r"^(.+?)\s+(closest to|nearest to|next to|near|on|under|below|above)\s+(?:the )?(.+)$", phrase)
            if m2:
                phrase, rel, anchors = m2.group(1), m2.group(2), [m2.group(3)]
            m3 = re.search(r"^(.+?)\s+between\s+(?:the )?(.+?)\s+and\s+(?:the )?(.+)$", phrase)
            if m3:
                phrase, rel, anchors = m3.group(1), "between", [m3.group(2), m3.group(3)]
            order += 1
            legs.append({"type": op, "order": order, "anchors": [],
                         "target": {"category": phrase, "attributes": [],
                                    "relation": rel, "anchors": anchors}})

    return {"question_type": "instruction_following",
            "target": {"category": None, "attributes": []},
            "relations": [], "path_constraints": legs}


def parse_question(question: str) -> QuerySpec:
    try:
        parsed = call_anthropic(question)
    except Exception as e:  # noqa: BLE001 -- must never crash the pipeline
        print(f"[question_parser] LLM parse failed ({e}), using regex fallback")
        if os.environ.get("STRICT_LLM"):
            raise
        parsed = _fallback_parse(question)

    target = parsed.get("target") or {}

    relations = []
    for r in parsed.get("relations") or []:
        if not isinstance(r, dict):
            continue
        rtype = _norm_relation(r.get("type"))
        anchors = [a for a in (_clean(x) for x in (r.get("anchors") or [])) if a]
        if rtype and anchors:
            relations.append(Relation(type=rtype, anchors=anchors))

    trel = _norm_relation(target.get("relation"))
    tanch = [a for a in (_clean(x) for x in (target.get("anchors") or [])) if a]
    if trel and tanch:
        relations.append(Relation(type=trel, anchors=tanch))

    constraints = []
    for c in parsed.get("path_constraints") or []:
        pc = _parse_constraint(c)
        if pc is not None:
            constraints.append(pc)
    constraints.sort(key=lambda c: (c.order is None, c.order or 0))
    for i, c in enumerate(constraints, 1):
        c.order = i

    _fix_goto_via(constraints, question)

    qtype = parsed.get("question_type", "object_reference")
    spec = QuerySpec(
        question_type=qtype,
        target_category=_clean(target.get("category")),
        target_attributes=[str(a).strip().lower() for a in (target.get("attributes") or []) if a],
        relations=relations,
        path_constraints=constraints,
        raw_question=question,
    )

    if qtype == "instruction_following":
        if not constraints:
            print("[question_parser] WARNING: instruction question parsed to zero legs "
                  "-- the robot has nowhere to drive")
        else:
            final = spec.final_target()
            print(f"[question_parser] {len(constraints)} legs:")
            for c in constraints:
                mark = "  <- FINAL" if (c.target is not None and c.target is final) else ""
                print(f"    {c}{mark}")
    else:
        print(f"[question_parser] {qtype}: target={spec.target_category!r} "
              f"attrs={spec.target_attributes} relations={relations}")
    print(f"[question_parser] categories needed: {spec.required_categories()}")
    return spec


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or _FEWSHOT[1][0]
    spec = parse_question(q)
    print()
    print("question_type :", spec.question_type)
    print("targets       :", spec.targets())
    print("final target  :", spec.final_target())
