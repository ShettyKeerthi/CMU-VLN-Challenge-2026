"""
Incrementally-built open-vocabulary 3D scene graph: a flat set of object
nodes (label, attributes, 3D position, rough size) plus on-demand spatial
relation queries. Deliberately simple (ConceptGraphs-style, not a full
hierarchical region graph like HOV-SG) -- start here, add a region layer
only if flat-object queries prove insufficient on your training scenes.

Referent resolution evaluates the RELATION TYPE, not just proximity to
anchors. Summing distances to every anchor scores "far from the sofa"
identically to "near the sofa", and scores "between A and B" the same as
"near A and B" -- which are different objects whenever A and B are not
adjacent. Each relation now has its own scoring function and they are
combined multiplicatively, so a candidate must satisfy every stated
constraint rather than averaging a failure away.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import DETECTION_DEDUP_DIST_M

# Distance at which "near" has decayed to ~0.37. Roughly furniture scale.
NEAR_SCALE_M = 1.5

# Vertical gap tolerated for on/above/below before the support reads as broken.
SUPPORT_GAP_M = 0.30

# Minimum score for a candidate to be considered a valid answer at all.
MIN_RELATION_SCORE = 0.05


@dataclass
class ObjectNode:
    label: str
    color: str
    position: np.ndarray       # (3,) map frame
    size: np.ndarray           # (3,) rough bbox extent, meters
    observation_count: int = 1
    confidence: float = 0.0

    @property
    def top(self) -> float:
        return float(self.position[2] + 0.5 * self.size[2])

    @property
    def bottom(self) -> float:
        return float(self.position[2] - 0.5 * self.size[2])


def _xy(n: ObjectNode) -> np.ndarray:
    return n.position[:2]


def _planar_dist(a: ObjectNode, b: ObjectNode) -> float:
    return float(np.linalg.norm(_xy(a) - _xy(b)))


def _footprint_overlap(a: ObjectNode, b: ObjectNode, pad: float = 0.15) -> float:
    """How much of a's centre sits within b's footprint, as 0..1.

    Axis-aligned. ObjectNode carries no heading, so a rotated anchor is
    approximated by its axis-aligned extent -- fine for the on/under queries
    in these scenes, but it is the first thing to revisit if 'the vase on the
    tv cabinet' picks the wrong object.
    """
    d = np.abs(_xy(a) - _xy(b))
    half = 0.5 * b.size[:2] + pad
    if np.any(half <= 1e-6):
        return 0.0
    frac = np.clip(1.0 - d / half, 0.0, 1.0)
    return float(frac[0] * frac[1])


# --------------------------------------------------------------- relations
# Each returns 0.0 (constraint violated) .. 1.0 (perfectly satisfied).

def _score_near(cand: ObjectNode, anchors: List[ObjectNode]) -> float:
    if not anchors:
        return 0.0
    d = min(_planar_dist(cand, a) for a in anchors)
    return math.exp(-d / NEAR_SCALE_M)


def _score_closest_to(cand: ObjectNode, anchors: List[ObjectNode],
                      siblings: List[ObjectNode]) -> float:
    """Superlative: rank against the other candidates, don't just measure.

    'the lamp closest to the black chair' is a comparison among lamps. An
    absolute distance score would happily pick a lamp 3 m away if it were the
    only one scored.
    """
    if not anchors or not siblings:
        return 0.0
    d = min(_planar_dist(cand, a) for a in anchors)
    dists = sorted(min(_planar_dist(s, a) for a in anchors) for s in siblings)
    if len(dists) == 1:
        return 1.0
    if d <= dists[0] + 1e-6:
        # Reward a clear win: a decisive margin is more trustworthy than a tie.
        margin = dists[1] - dists[0]
        return 0.7 + 0.3 * min(margin / NEAR_SCALE_M, 1.0)
    rank = dists.index(d) if d in dists else len(dists) - 1
    return max(0.0, 0.5 * (1.0 - rank / max(1, len(dists) - 1)))


def _score_far_from(cand: ObjectNode, anchors: List[ObjectNode],
                    siblings: List[ObjectNode]) -> float:
    if not anchors or not siblings:
        return 0.0
    d = min(_planar_dist(cand, a) for a in anchors)
    dists = sorted((min(_planar_dist(s, a) for a in anchors) for s in siblings),
                   reverse=True)
    if len(dists) == 1:
        return 1.0
    if d >= dists[0] - 1e-6:
        margin = dists[0] - dists[1]
        return 0.7 + 0.3 * min(margin / NEAR_SCALE_M, 1.0)
    rank = dists.index(d) if d in dists else len(dists) - 1
    return max(0.0, 0.5 * (1.0 - rank / max(1, len(dists) - 1)))


def _score_between(cand: ObjectNode, anchors: List[ObjectNode]) -> float:
    """Perpendicular offset from the A-B segment, plus a check that the
    candidate projects INSIDE the segment rather than beyond either end.

    Detour ratio alone is not enough: a point just past B has a small detour
    but is not between anything.
    """
    if len(anchors) < 2:
        return 0.0
    best = 0.0
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            a, b = _xy(anchors[i]), _xy(anchors[j])
            ab = b - a
            L = float(np.linalg.norm(ab))
            if L < 1e-6:
                continue
            t = float(np.dot(_xy(cand) - a, ab) / (L * L))
            if not (0.0 <= t <= 1.0):
                continue                       # projects outside the segment
            perp = float(np.linalg.norm(_xy(cand) - (a + t * ab)))
            # Allow half the separation as lateral slack; closer to the line is better.
            lateral = math.exp(-perp / max(0.5 * L, 0.5))
            # Prefer the middle of the span over hugging an endpoint.
            centred = 1.0 - abs(t - 0.5) * 2.0 * 0.4
            best = max(best, lateral * centred)
    return best


def _score_on(cand: ObjectNode, anchors: List[ObjectNode]) -> float:
    """Candidate rests on top of an anchor: footprint overlap AND the
    candidate's underside sitting near the anchor's top surface."""
    best = 0.0
    for a in anchors:
        overlap = _footprint_overlap(cand, a)
        if overlap <= 0.0:
            continue
        gap = abs(cand.bottom - a.top)
        vertical = math.exp(-gap / SUPPORT_GAP_M) if cand.position[2] > a.position[2] else 0.0
        best = max(best, overlap * vertical)
    return best


def _score_above(cand: ObjectNode, anchors: List[ObjectNode]) -> float:
    """Higher than the anchor and overlapping in plan, but not necessarily
    touching -- 'the cabinet with a picture above it' spans a wall gap."""
    best = 0.0
    for a in anchors:
        overlap = _footprint_overlap(cand, a, pad=0.4)
        if overlap <= 0.0 or cand.position[2] <= a.position[2]:
            continue
        best = max(best, overlap)
    return best


def _score_below(cand: ObjectNode, anchors: List[ObjectNode]) -> float:
    best = 0.0
    for a in anchors:
        overlap = _footprint_overlap(cand, a, pad=0.4)
        if overlap <= 0.0 or cand.position[2] >= a.position[2]:
            continue
        best = max(best, overlap)
    return best


def _score_supports(cand: ObjectNode, anchors: List[ObjectNode]) -> float:
    """Inverse of 'on': 'the table WITH the vase ON it' -- the table is the
    target and the vase is the anchor."""
    best = 0.0
    for a in anchors:
        overlap = _footprint_overlap(a, cand)
        if overlap <= 0.0:
            continue
        gap = abs(a.bottom - cand.top)
        vertical = math.exp(-gap / SUPPORT_GAP_M) if a.position[2] > cand.position[2] else 0.0
        best = max(best, overlap * vertical)
    return best


# Relations needing the sibling set (superlatives) vs those that don't.
_SIMPLE: Dict[str, Callable] = {
    "near": _score_near,
    "next_to": _score_near,
    "beside": _score_near,
    "between": _score_between,
    "on": _score_on,
    "above": _score_above,
    "below": _score_below,
    "under": _score_below,
    "supports": _score_supports,
    "with": _score_supports,
}
_SUPERLATIVE: Dict[str, Callable] = {
    "closest_to": _score_closest_to,
    "nearest": _score_closest_to,
    "far_from": _score_far_from,
    "farthest_from": _score_far_from,
    "furthest_from": _score_far_from,
}


class SceneGraph:
    def __init__(self):
        self.nodes: List[ObjectNode] = []

    # ------------------------------------------------------------- ingest

    def add_or_merge(self, label: str, color: str, position: np.ndarray, score: float,
                     size=None, approx_size=(0.4, 0.4, 0.4)):
        """Adds a new node, or merges into a nearby matching one.

        Position is averaged across observations, but size is taken from the
        single best-scoring view rather than averaged: a partial view yields a
        truncated extent, and averaging it in shrinks a box that was correct.
        """
        size = np.asarray(size if size is not None else approx_size, dtype=np.float32)
        position = np.asarray(position, dtype=np.float32)

        for node in self.nodes:
            if node.label == label and np.linalg.norm(node.position[:2] - position[:2]) < DETECTION_DEDUP_DIST_M:
                n = node.observation_count
                node.position = (node.position * n + position) / (n + 1)
                node.observation_count += 1
                if score > node.confidence:
                    node.confidence = score
                    node.size = size
                    if color and color != "unknown":
                        node.color = color
                return node

        node = ObjectNode(label=label, color=color, position=position,
                          size=size, confidence=score)
        self.nodes.append(node)
        return node

    # ------------------------------------------------------------ matching

    @staticmethod
    def _norm(s: str) -> str:
        s = (s or "").strip().lower()
        for art in ("a ", "an ", "the "):
            if s.startswith(art):
                s = s[len(art):]
                break
        return s

    def _matches(self, node: ObjectNode, category: Optional[str], attributes: List[str]) -> bool:
        if category and self._norm(category) not in self._norm(node.label):
            return False
        for attr in attributes or []:
            if attr.lower() != (node.color or "").lower():
                return False
        return True

    def find_matching(self, category: Optional[str], attributes: List[str]) -> List[ObjectNode]:
        return [n for n in self.nodes if self._matches(n, category, attributes)]

    def _resolve_anchors(self, anchor_specs: Sequence[str]) -> List[ObjectNode]:
        out = []
        for spec in anchor_specs or []:
            out.extend(self.find_matching(spec, []))
        return out

    # -------------------------------------------------------------- counts

    def count(self, category: Optional[str], attributes: List[str],
              near_anchor_category: Optional[str] = None, near_radius_m: float = 2.0) -> int:
        candidates = self.find_matching(category, attributes)
        if near_anchor_category:
            anchors = self.find_matching(near_anchor_category, [])
            candidates = [c for c in candidates
                          if any(_planar_dist(c, a) < near_radius_m for a in anchors)]
        return len(candidates)

    # ------------------------------------------------------------ referent

    def score_candidate(self, cand: ObjectNode, relations, siblings: List[ObjectNode]
                        ) -> Tuple[float, List[str]]:
        """Combined score for one candidate against all stated relations.

        Multiplicative: every relation must hold. Averaging would let a
        candidate that nails 'near the tv' but sits nowhere near 'between the
        sofa and the door' still win on the strength of the first.
        """
        if not relations:
            return 1.0, []

        total, notes = 1.0, []
        for rel in relations:
            rtype = (getattr(rel, "type", None) or "").strip().lower().replace(" ", "_")
            rel_anchors = list(getattr(rel, "anchors", []) or [])

            # `between` is the one relation whose anchors are POSITIONAL: A and
            # B are distinct roles, not a pooled candidate set. Flattening them
            # through _resolve_anchors turns ["tv","door"] into [tv,tv,tv] when
            # there are three tv nodes and no door, and _score_between then
            # happily measures "between two TVs" and returns 0.99.
            if rtype == "between":
                if len(rel_anchors) < 2:
                    notes.append("between: needs 2 anchors, skipped")
                    continue
                a = self.find_unique_referent(rel_anchors[0], [], verbose=False)
                b = self.find_unique_referent(rel_anchors[1], [], verbose=False)
                if a is None or b is None:
                    missing = [n for n, o in zip(rel_anchors[:2], (a, b)) if o is None]
                    notes.append(f"between: {missing} not in graph, SKIPPED")
                    continue
                s = _score_between(cand, [a, b])
                notes.append(f"between({self._norm(a.label)},{self._norm(b.label)})={s:.2f}")
                total *= s
                continue
            anchors = self._resolve_anchors(getattr(rel, "anchors", []) or [])
            if not anchors:
                notes.append(f"{rtype}: anchors {list(getattr(rel, 'anchors', []))} not in graph, skipped")
                continue

            if rtype in _SUPERLATIVE:
                s = _SUPERLATIVE[rtype](cand, anchors, siblings)
            elif rtype in _SIMPLE:
                s = _SIMPLE[rtype](cand, anchors)
            else:
                notes.append(f"{rtype}: unknown relation, skipped")
                continue

            notes.append(f"{rtype}({','.join(self._norm(a.label) for a in anchors[:3])})={s:.2f}")
            total *= s

        return total, notes

    def find_unique_referent(self, category: Optional[str], attributes: List[str],
                             anchor_categories: List[str] = None,
                             relations=None, verbose: bool = True) -> Optional[ObjectNode]:
        """Resolve a referring expression to one object.

        Pass `relations` (the QuerySpec.relations list) to get relation-aware
        scoring. `anchor_categories` is kept for backward compatibility and
        degrades to proximity, which is what the old implementation did for
        every relation type.
        """
        candidates = self.find_matching(category, attributes)

        # Attributes come from mean-RGB colour estimation and are unreliable;
        # do not let a bad colour guess return nothing at all.
        if not candidates and attributes:
            candidates = self.find_matching(category, [])
            if verbose and candidates:
                print(f"[graph] no {category!r} with attributes {attributes}, "
                      f"falling back to {len(candidates)} by category alone")
        if not candidates:
            return None

        if not relations:
            if not anchor_categories or len(candidates) == 1:
                return max(candidates, key=lambda n: (n.confidence, n.observation_count))
            anchors = self._resolve_anchors(anchor_categories)
            if not anchors:
                return max(candidates, key=lambda n: (n.confidence, n.observation_count))
            return min(candidates,
                       key=lambda n: sum(_planar_dist(n, a) for a in anchors))

        scored = []
        for c in candidates:
            s, notes = self.score_candidate(c, relations, candidates)
            # Detection confidence as a mild tiebreaker only; geometry dominates.
            s *= (0.85 + 0.15 * min(c.confidence, 1.0))
            scored.append((s, c, notes))
        scored.sort(key=lambda t: t[0], reverse=True)

        if verbose:
            for s, c, notes in scored[:5]:
                print(f"[graph] {s:.3f}  {c.label} at "
                      f"({c.position[0]:.2f},{c.position[1]:.2f},{c.position[2]:.2f})  "
                      f"{' '.join(notes)}")

        best_score, best, _ = scored[0]
        if best_score < MIN_RELATION_SCORE:
            # Nothing satisfies the relations. Returning the most-confident
            # detection beats returning nothing, since an unanswered
            # object_reference scores zero either way.
            if verbose:
                print(f"[graph] no candidate satisfies the relations "
                      f"(best {best_score:.3f}); falling back to confidence")
            return max(candidates, key=lambda n: (n.confidence, n.observation_count))

        if len(scored) > 1 and verbose:
            runner = scored[1][0]
            if runner > 0 and best_score / max(runner, 1e-6) < 1.25:
                print(f"[graph] WARNING ambiguous: {best_score:.3f} vs {runner:.3f}")

        return best

    # --------------------------------------------------------------- misc

    def nodes_by_category(self, category: str) -> List[ObjectNode]:
        return self.find_matching(category, [])

    def has_all(self, categories: Sequence[str]) -> bool:
        """True if every category has at least one node. Useful as an
        answer-time gate: keep exploring rather than answering on a graph
        that is missing an anchor the question depends on."""
        return all(self.find_matching(c, []) for c in categories)

    def summary(self) -> str:
        if not self.nodes:
            return "  (empty)"
        return "\n".join(
            f"  {n.label} [{n.color}] at ({n.position[0]:.2f}, {n.position[1]:.2f}, {n.position[2]:.2f}) "
            f"size ({n.size[0]:.2f},{n.size[1]:.2f},{n.size[2]:.2f}), "
            f"seen {n.observation_count}x, conf={n.confidence:.2f}"
            for n in sorted(self.nodes, key=lambda n: n.label)
        )
