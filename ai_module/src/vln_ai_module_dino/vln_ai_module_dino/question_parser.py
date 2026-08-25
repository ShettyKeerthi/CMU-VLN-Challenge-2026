"""
Turns the raw natural-language challenge question into a structured spec
the rest of the pipeline can execute against, using an LLM for the
open-ended parsing (attribute extraction, relation extraction, path
constraint decomposition) that a hand-written grammar would be too brittle
to cover.

Uses a local Ollama server (no API key, no internet egress needed at grading
time -- see config.py for why). If Ollama isn't already running, this module
starts it itself so nothing manual is required before launching the node.
"""

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from .config import NAMED_COLORS_RGB, OLLAMA_API_URL, OLLAMA_MODEL

SYSTEM_PROMPT = """You convert a single navigation-challenge question into strict JSON.

Question types:
- "numerical": asks "how many X" -- answer is an integer count.
- "object_reference": asks to "find" a single unique object -- answer is one object's bounding box.
- "instruction_following": asks the robot to take/follow a path described via landmarks/constraints -- answer is a sequence of waypoints.

Output ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{
  "question_type": "numerical" | "object_reference" | "instruction_following",
  "target": {
    "category": "<COMPLETE object noun phrase, singular, e.g. 'chair', 'wall lamp', 'door frame'>",
    "attributes": ["<adjectives, e.g. 'blue'>"],
  },
  "relations": [
    {"type": "near"|"between"|"on"|"closest_to"|"far_from"|"below"|"above", "anchors": [<anchor>, ...]}
  ],
  "path_constraints": [
    {"type": "near"|"avoid"|"between", "anchors": [<anchor>, ...], "order": <int, 1-based order this constraint applies in the path, or null if unordered>}
  ]
}

Each <anchor> in "anchors" (for BOTH "relations" and "path_constraints") is EITHER:
  (a) a plain string category, SINGULAR, e.g. "window" -- use this for the common case where the anchor is just named directly, with no further description.
  (b) an OBJECT, for when the anchor noun is ITSELF further qualified by its own relation to another object -- e.g. "the book ON THE STOOL" as an anchor means the anchor isn't just any book, it's specifically the book that is on a stool. Use this shape:
      {"category": "<anchor noun, SINGULAR>", "qualifier_type": "<relation word: near|between|on|closest_to|far_from|below|above>", "qualifier_category": "<the OTHER noun this anchor relates to, SINGULAR>"}
  Only ONE level of qualification is supported -- if an anchor's own qualifier is itself further qualified, just use the qualifier's own head noun as qualifier_category and drop any further nesting.
  "far_from"/"furthest_from"/"furthest" as a qualifier_type means: among every instance of that category, pick the ONE instance FARTHEST from the qualifier_category. "closest_to"/"nearest" means pick the ONE instance NEAREST to the qualifier_category. Use these whenever the question says "furthest"/"farthest" or "closest"/"nearest" to single out ONE specific instance by comparison to another object.

WORKED EXAMPLE 1 -- "Find the pillow closest to the book on the stool":
{
  "question_type": "object_reference",
  "target": {"category": "pillow", "attributes": []},
  "relations": [
    {"type": "closest_to", "anchors": [
      {"category": "book", "qualifier_type": "on", "qualifier_category": "stool"}
    ]}
  ],
  "path_constraints": []
}
Note "book" is an OBJECT (qualified anchor) here, not a plain string, because the question specifies WHICH book (the one on the stool) rather than just any book.

WORKED EXAMPLE 2 -- "First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table":
{
  "question_type": "instruction_following",
  "target": {"category": null, "attributes": []},
  "relations": [],
  "path_constraints": [
    {"type": "near", "anchors": [
      {"category": "potted plant", "qualifier_type": "far_from", "qualifier_category": "hookah"}
    ], "order": 1},
    {"type": "between", "anchors": ["column", "column"], "order": 2},
    {"type": "near", "anchors": [
      {"category": "tray", "qualifier_type": "on", "qualifier_category": "table"}
    ], "order": 3}
  ]
}
CRITICAL for instruction_following: EVERY step of the instruction -- including a step that selects ONE specific object by superlative comparison ("furthest", "closest") -- becomes its OWN path_constraint with the correct "order", using type "near" (meaning: go to / stop at this object). Do NOT put any part of a multi-step instruction into "target" or "relations" -- those fields must stay empty/minimal for instruction_following. A step like "go to X" or "stop at X" is type "near" with X as the anchor (qualified if X is further described, as in Example 2's steps 1 and 3).

Leave "path_constraints" empty for non-instruction questions, and "relations"/"target" minimal for instruction questions.
For numerical questions, "target" is the thing being counted and "relations" captures any qualifying spatial constraint.
IMPORTANT: every object noun anywhere in the output (target.category AND every relations/path_constraints anchor, including qualifier_category) must be SINGULAR, even if the question uses a plural noun (e.g. question says "windows" -> output "window"). Downstream matching is exact-word based, not plural-aware.
IMPORTANT: "attributes" must contain ONLY color words from this exact list: red, orange, yellow, green, blue, purple, pink, brown, black, white, gray. Never put prepositions, articles ("a", "the"), or relation words ("below", "near", "between", etc.) in "attributes" -- those belong in "relations"/"path_constraints" instead, or nowhere at all. If the question has no color mentioned, "attributes" must be an empty list [].
IMPORTANT: object categories are often TWO OR MORE WORDS ("wall lamp", "door frame", "trash can", "coffee table"). Always copy the COMPLETE noun phrase from the question -- do NOT drop modifier words and reduce it to just the head noun (e.g. the question says "wall lamp" -> output "wall lamp", NOT "lamp"; "door frame" -> "door frame", NOT "frame" or "door"). This applies to target.category and every anchor in relations/path_constraints.
"""


@dataclass
class Anchor:
    """An anchor category, optionally qualified by its OWN relation to a
    further category -- e.g. "the book ON THE STOOL" as an anchor for
    "closest to" is Anchor(category="book", qualifier_type="on",
    qualifier_category="stool"), not just a plain "book" string. Without
    this, a compound reference like this gets silently flattened into an
    OR-pool ("closest to any book OR any stool"), which is a different and
    usually wrong meaning than "closest to the specific book that's on a
    stool" -- confirmed live against "find the pillow closest to the book
    on the stool" before this was added.
    """
    category: str
    qualifier_type: Optional[str] = None
    qualifier_category: Optional[str] = None


@dataclass
class Relation:
    type: str
    anchors: List[Anchor] = field(default_factory=list)


@dataclass
class PathConstraint:
    type: str
    anchors: List[Anchor] = field(default_factory=list)
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


def _ollama_is_up() -> bool:
    try:
        requests.get(OLLAMA_API_URL.replace("/api/generate", "/api/tags"), timeout=1)
        return True
    except requests.exceptions.RequestException:
        return False


def _ensure_ollama_running(startup_timeout_s: float = 30.0) -> None:
    """Self-healing start: if the server isn't already up (e.g. someone
    launched the node without pre-starting `ollama serve`), start it as a
    background process and poll until it responds. No-op if it's already
    running -- safe to call on every question."""
    if _ollama_is_up():
        return
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if _ollama_is_up():
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"Ollama server did not come up within {startup_timeout_s}s. "
        "Check it's installed and the model was pulled at image build time "
        "(see ai_module/docker/Dockerfile)."
    )


def call_ollama(question: str) -> dict:
    _ensure_ollama_running()
    resp = requests.post(
        OLLAMA_API_URL,
        json={
            "model": OLLAMA_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": question,
            "format": "json",  # Ollama enforces valid JSON output server-side
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "")
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


def _expand_to_full_phrase(category: Optional[str], question: str) -> Optional[str]:
    """SAFETY NET 2026-08-07: confirmed live that the parser can truncate a
    compound noun to just its head word -- "Find the wall lamp that is
    between a door frame and a window" parsed target.category as "lamp",
    not "wall lamp", even with an explicit prompt instruction not to do
    this. Since the OWL-ViT vocabulary is built as f"a {category}" (see
    _build_vocabulary in main_node.py), "lamp" alone makes it search for
    ANY lamp (floor, ceiling, table...), not specifically wall lamps --
    confirmed this produced a scene graph full of generic "a lamp" nodes
    and picked the wrong object entirely, a category error, not just an
    accuracy one.

    If `category` is a single word that appears in the question immediately
    preceded by another word (e.g. "lamp" preceded by "wall"), prefer the
    longer two-word phrase actually present in the question over trusting
    the parser's (possibly truncated) category.
    """
    if not category or " " in category:
        return category  # already multi-word, or nothing to expand
    match = re.search(r"(\w+)\s+" + re.escape(category) + r"\b", question, re.IGNORECASE)
    if match:
        preceding_word = match.group(1)
        # DETERMINER FIX 2026-08-13: confirmed live that this regex has NO
        # exclusion for determiners, so "the pillow" in "Find THE PILLOW
        # closest to..." matched exactly like a genuine compound noun ("wall
        # lamp") -- producing target_category='the pillow' instead of
        # 'pillow'. Since scene_graph._matches() does a substring check
        # between category and each node's label (e.g. "a pillow"), "the
        # pillow" is NOT a substring of "a pillow" -- this silently broke
        # ALL matching for the target and every anchor, returning zero
        # candidates even with real pillows correctly in the scene graph.
        # Determiners are never the first half of a real compound noun, so
        # exclude them explicitly rather than trusting "any preceding word"
        # blindly.
        if preceding_word.lower() in ("a", "an", "the"):
            return category
        expanded = f"{preceding_word} {category}".lower()
        if expanded != category.lower():
            print(f"[question_parser] expanded truncated category {category!r} -> {expanded!r} (found in question text)")
        return expanded
    return category


# Non-color descriptive adjectives the model sometimes leaves attached to a
# category noun (e.g. "small table" instead of just "table") since there's
# no supported attribute slot for size/material/etc -- only color is
# supported (see NAMED_COLORS_RGB whitelist). Stripping these prevents a
# category string like "small table" from silently matching ZERO real
# scene-graph nodes (labels are always plain "a table", never "a small
# table"), the same class of bug the DETERMINER FIX addressed for "the".
_NON_COLOR_DESCRIPTORS = {
    "small", "large", "big", "tiny", "huge", "little", "medium", "tall", "short",
}


def _strip_leading_descriptor(category: Optional[str]) -> Optional[str]:
    if not category:
        return category
    words = category.split()
    if len(words) > 1 and words[0].lower() in _NON_COLOR_DESCRIPTORS:
        stripped = " ".join(words[1:])
        print(f"[question_parser] stripped unsupported descriptor from category {category!r} -> {stripped!r}")
        return stripped
    return category


# VALID qualifier_type values -- must be an actual relation word, never a
# descriptive adjective (size, material, etc). See _parse_anchor below for
# why this whitelist exists.
_VALID_QUALIFIER_TYPES = {
    "near", "between", "on", "closest_to", "far_from", "furthest_from",
    "furthest", "nearest", "below", "above", "under", "underneath",
    "beneath", "over", "atop",
}


def _parse_anchor(raw) -> Anchor:
    """Anchors from the LLM parse come back either as a plain string
    category (the common, unqualified case) or, for a QUALIFIED anchor
    like "the book ON THE STOOL", as an object with category/qualifier_type/
    qualifier_category (see the Anchor dataclass and the WORKED EXAMPLE in
    SYSTEM_PROMPT). Accept both shapes -- if raw isn't the expected object
    shape (model didn't follow instructions, or this came from the crude
    regex fallback which never produces qualified anchors), degrade
    gracefully to a plain unqualified Anchor rather than crashing. This
    matches the project's existing philosophy (see the Relation(**r) crash
    history) of never letting a single parse quirk take down the whole
    pipeline.

    QUALIFIER_TYPE VALIDATION 2026-08-17: confirmed live -- "the small
    table farthest from the columns" produced qualifier_type='small', a
    size adjective, not a relation word. The Anchor schema has no
    attributes field (only target.attributes does, and even that's
    color-only), so the model had nowhere correct to put "small" and
    improvised badly, corrupting both the category string ("small table")
    AND the qualifier_type slot. Rather than trust qualifier_type blindly,
    validate it against the actual relation vocabulary -- an invalid value
    is dropped entirely (falls back to an unqualified anchor) rather than
    reaching scene_graph._relation_score with nonsense input.
    """
    if isinstance(raw, dict):
        category = str(raw.get("category", "")).strip()
        qualifier_type = raw.get("qualifier_type")
        qualifier_category = raw.get("qualifier_category")
        if qualifier_type and qualifier_type.lower() not in _VALID_QUALIFIER_TYPES:
            print(f"[question_parser] dropped invalid qualifier_type {qualifier_type!r} "
                  f"(not a recognized relation word) for anchor category {category!r}")
            qualifier_type = None
        # Only treat as genuinely qualified if BOTH qualifier fields are
        # present and non-empty -- a dict with just "category" (model
        # inconsistently wrapped a plain anchor in an object) should still
        # behave as an unqualified anchor, not a broken qualified one.
        if not qualifier_type or not qualifier_category:
            qualifier_type, qualifier_category = None, None
        return Anchor(category=category, qualifier_type=qualifier_type, qualifier_category=qualifier_category)
    return Anchor(category=str(raw))


# MISATTACHED SUPERLATIVE REPAIR 2026-08-17: confirmed live, 6 CONSECUTIVE
# times, that llama3.2:3b mis-parses "the [target] [superlative] from the
# [plural anchor category]" (e.g. "the table FARTHEST FROM THE COLUMNS")
# into TWO separate, broken path_constraints instead of one correctly-
# qualified one -- even with an explicit worked example in the prompt
# already covering the singular case ("furthest from the hookah"). The
# model appears to pattern-match the plural comparison object ("the
# columns") to the UNRELATED "between the two X" example instead. Since
# prompting alone has not fixed this across 6 tries, repair the known
# failure signature in code rather than keep relying on instructions the
# model has repeatedly not followed for this specific pattern.
_SUPERLATIVE_STANDALONE_TYPES = {"far_from", "furthest_from", "furthest", "closest_to", "nearest"}


def _repair_misattached_superlative(path_constraints: List[PathConstraint]) -> List[PathConstraint]:
    """Detects: a standalone constraint whose TYPE is itself a superlative
    word, with exactly 2 unqualified anchors of the SAME category (echoing
    the "between the two X" signature) -- immediately followed by a plain,
    unqualified "near" constraint for the real target. Repairs this into
    the single correctly-qualified constraint it should have been:
    type="near" with ONE anchor whose category is the real target,
    qualified by the superlative type and the comparison category.
    """
    repaired = []
    i = 0
    while i < len(path_constraints):
        c = path_constraints[i]
        if (
            c.type in _SUPERLATIVE_STANDALONE_TYPES
            and len(c.anchors) == 2
            and c.anchors[0].category == c.anchors[1].category
            and not c.anchors[0].qualifier_type
            and i + 1 < len(path_constraints)
        ):
            nxt = path_constraints[i + 1]
            if nxt.type == "near" and len(nxt.anchors) == 1 and not nxt.anchors[0].qualifier_type:
                merged_anchor = Anchor(
                    category=nxt.anchors[0].category,
                    qualifier_type=c.type,
                    qualifier_category=c.anchors[0].category,
                )
                print(f"[question_parser] repaired misattached superlative: merged {c.type!r} "
                      f"constraint (anchors={[a.category for a in c.anchors]}) into the next step "
                      f"-> Anchor(category={merged_anchor.category!r}, qualifier_type={merged_anchor.qualifier_type!r}, "
                      f"qualifier_category={merged_anchor.qualifier_category!r})")
                repaired.append(PathConstraint(type="near", anchors=[merged_anchor], order=nxt.order))
                i += 2
                continue
        repaired.append(c)
        i += 1
    return repaired


def parse_question(question: str) -> QuerySpec:
    try:
        parsed = call_ollama(question)
    except Exception as e:  # noqa: BLE001 -- pipeline must never crash on a parse failure
        print(f"[question_parser] LLM parse failed ({e}), using regex fallback")
        parsed = _fallback_parse(question)

    target = parsed.get("target") or {}

    # SAFETY NET 2026-08-07: confirmed live that a 3B local model can stuff
    # stray non-color words into "attributes" (e.g. saw
    # target_attributes=['below', 'a'] for "how many sofas are below a
    # window" -- "below" and "a" leaked in from the relation clause instead
    # of staying out of this field). Since every entry in target_attributes
    # is used as a REQUIRED color match in SceneGraph._matches(), even one
    # garbage word makes every real object fail to match, silently
    # producing 0/None regardless of what's actually in the scene graph.
    # Whitelist against the same color vocabulary perception uses (see
    # NAMED_COLORS_RGB in config.py) so a parsing slip can't corrupt the
    # count -- anything that isn't a real known color is dropped rather
    # than trusted.
    raw_attrs = target.get("attributes", []) or []
    valid_attrs = [a for a in raw_attrs if a.lower().strip() in NAMED_COLORS_RGB]
    # SAFETY NET #2 2026-08-07: the color-vocabulary check above stops
    # garbage words, but not a fabricated-yet-valid color -- confirmed live
    # against "how many sofas are below a window?" (no color mentioned at
    # all) that the LLM hallucinated target_attributes=['blue'] anyway,
    # despite the prompt explicitly saying attributes must be [] when no
    # color is mentioned, and despite temperature=0.0. Since "blue" passes
    # the vocabulary check (it IS a real color), that check alone can't
    # catch fabrication. Cross-check against the raw question text as a
    # second, stronger gate: only trust a color if that word actually
    # appears in what was asked. This can't be fooled by hallucination the
    # way a vocabulary whitelist can.
    question_lower = question.lower()
    valid_attrs = [a for a in valid_attrs if a.lower().strip() in question_lower]
    dropped = set(raw_attrs) - set(valid_attrs)
    if dropped:
        print(f"[question_parser] dropped attribute(s) {dropped} not present in question text: {question!r}")

    # ANCHOR PARSING 2026-08-12: no longer a naive Relation(**r) unpack --
    # "anchors" can now contain a mix of plain strings and qualified-anchor
    # objects (see Anchor dataclass), so each one is parsed individually via
    # _parse_anchor rather than trusted to unpack directly.
    relations = []
    for r in parsed.get("relations", []):
        raw_anchors = r.get("anchors", []) or []
        anchors = [_parse_anchor(a) for a in raw_anchors]
        relations.append(Relation(type=r.get("type", ""), anchors=anchors))
    path_constraints = []
    for c in parsed.get("path_constraints", []):
        raw_anchors = c.get("anchors", []) or []
        anchors = [_parse_anchor(a) for a in raw_anchors]
        path_constraints.append(PathConstraint(type=c.get("type", ""), anchors=anchors, order=c.get("order")))

    target_category = _strip_leading_descriptor(_expand_to_full_phrase(target.get("category"), question))
    for r in relations:
        for a in r.anchors:
            a.category = _strip_leading_descriptor(_expand_to_full_phrase(a.category, question))
            if a.qualifier_category:
                a.qualifier_category = _strip_leading_descriptor(_expand_to_full_phrase(a.qualifier_category, question))
    for c in path_constraints:
        for a in c.anchors:
            a.category = _strip_leading_descriptor(_expand_to_full_phrase(a.category, question))
            if a.qualifier_category:
                a.qualifier_category = _strip_leading_descriptor(_expand_to_full_phrase(a.qualifier_category, question))

    path_constraints = _repair_misattached_superlative(path_constraints)

    return QuerySpec(
        question_type=parsed.get("question_type", "object_reference"),
        target_category=target_category,
        target_attributes=valid_attrs,
        relations=relations,
        path_constraints=path_constraints,
        raw_question=question,
    )
