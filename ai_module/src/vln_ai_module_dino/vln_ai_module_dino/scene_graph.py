"""
Incrementally-built open-vocabulary 3D scene graph: a flat set of object
nodes (label, attributes, 3D position, rough size) plus on-demand spatial
relation queries. Deliberately simple (ConceptGraphs-style, not a full
hierarchical region graph like HOV-SG) -- start here, add a region layer
only if flat-object queries prove insufficient on your training scenes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import DETECTION_DEDUP_DIST_M

# ---------------------------------------------------------------------------
# Geometric category sanity-check (2026-08-12)
# ---------------------------------------------------------------------------
# ADDED to directly fix the confirmed live bug (see DISTRACTOR CATEGORIES
# note in config.py): a real column at (-0.99, 1.08) was being confidently
# (0.57-0.94) misclassified as "sofa" by Grounding DINO's phrase-grounding.
# That mislabeled node corrupts BOTH numerical counts (inflates the sofa
# count by however many columns get mislabeled) and object-reference
# answers (a mislabeled column can get selected as "the sofa below the
# window" if it happens to sit in a plausible position). The distractor-
# vocabulary fix in config.py reduces how often this happens at detection
# time; this is a second, independent check applied to whatever label
# actually comes back, using real 3D shape from the lidar cluster
# (ros_utils.localize_detection_3d now returns cluster size, not just
# position) -- catches misclassifications the vocabulary trick doesn't.
#
# TODO: these thresholds are reasonable starting estimates (columns ~0.2-
# 0.3m square cross-section and much taller than wide; sofas low, wide,
# elongated), NOT yet empirically tuned against this scene's real cluster
# sizes. Before trusting this in a submission run, log the actual (w, d, h)
# computed for the known column at (-0.99, 1.08) and for a known real sofa,
# and adjust these constants to match what the lidar cluster actually
# measures -- don't leave them as guesses.
GEOMETRIC_CHECK_COLUMN_SLENDERNESS_MIN = 2.5   # height / min(width,depth)
GEOMETRIC_CHECK_COLUMN_FOOTPRINT_MAX_M2 = 0.35  # width * depth
GEOMETRIC_CHECK_COLUMN_ASPECT_MAX = 1.8         # max(w,d)/min(w,d), roughly-square cross-section
GEOMETRIC_CHECK_SOFA_HEIGHT_MAX_M = 1.1
GEOMETRIC_CHECK_SOFA_FOOTPRINT_MIN_M2 = 0.6
GEOMETRIC_CHECK_SOFA_ASPECT_MIN = 1.4           # elongated footprint (length >> depth)

# ---------------------------------------------------------------------------
# General degenerate-cluster filter (2026-08-12) -- SCENE-AGNOSTIC, not tied
# to any specific category.
# ---------------------------------------------------------------------------
# Confirmed live (arabic_room) that several false-positive detections across
# MULTIPLE categories share a common signature: a lidar cluster that's a
# paper-thin sliver in one horizontal axis (e.g. w=0.050, d=0.652,
# aspect=13.04) -- a shape no real piece of furniture or fixture produces in
# any scene. This is very likely a partial/edge cluster (an occluded corner,
# a stray point cluster on a wall edge) rather than any genuine object, so
# it's rejected outright regardless of what label DINO assigned -- unlike
# geometric_category_check above, this isn't correcting a label, it's
# discarding noise before it ever becomes a node. Kept intentionally
# conservative (only the most extreme, physically-implausible shapes) so it
# doesn't risk discarding genuinely thin real objects (e.g. a picture frame,
# a slim floor lamp) -- 0.050m is also literally the size-flooring value
# used in ros_utils.localize_detection_3d, so an unclipped 0.050 here is a
# strong signal the true cluster was even thinner/more degenerate than that.
DEGENERATE_ASPECT_MIN = 6.0
DEGENERATE_MIN_DIM_MAX_M = 0.10

# FLAT-OBJECT EXEMPTION 2026-08-12: confirmed live (arabic_room) that
# is_degenerate_cluster() was wrongly rejecting REAL windows -- a window is
# structurally a flat opening in a wall, so its depth dimension is
# naturally tiny (~0.05m) compared to width/height, producing the exact
# same "thin sliver" shape signature as genuine sofa-detection noise. The
# filter can't tell "real flat wall fixture" apart from "degenerate/noisy
# cluster" by shape alone -- they're geometrically identical. Exempting
# known flat/wall-mounted category keywords is a generalizable compromise:
# it keeps the noise filter working for furniture-like categories (sofas,
# chairs, tables -- anything expected to have real volume) while not
# penalizing categories that are supposed to be thin. This list is a
# reasonable starting set based on common indoor-scene vocabulary, not
# exhaustive -- extend it if a new scene's question vocabulary surfaces
# another flat category (e.g. "shelf", "vent", "switch plate") getting
# wrongly rejected.
FLAT_OBJECT_CATEGORY_KEYWORDS = [
    "window", "door", "frame", "mirror", "painting", "picture",
    "poster", "curtain", "lamp", "sign", "vent", "switch", "outlet",
    "clock", "shelf",
]


def is_degenerate_cluster(size: np.ndarray, label: str = "") -> bool:
    """True if this detection's 3D cluster shape is implausibly thin/sliver-
    like for a non-flat object -- a general noise filter, not specific to
    any one furniture category, so it generalizes across scenes and
    vocabularies you haven't tested yet. Exempts known flat/wall-mounted
    categories (see FLAT_OBJECT_CATEGORY_KEYWORDS) since their natural
    shape is indistinguishable from noise by geometry alone."""
    label_lower = label.lower()
    if any(kw in label_lower for kw in FLAT_OBJECT_CATEGORY_KEYWORDS):
        return False
    w, d, _h = float(size[0]), float(size[1]), float(size[2])
    min_dim = min(w, d)
    aspect_xy = max(w, d) / max(min_dim, 1e-3)
    return aspect_xy > DEGENERATE_ASPECT_MIN and min_dim < DEGENERATE_MIN_DIM_MAX_M


def geometric_category_check(label: str, size: np.ndarray):
    """Sanity-checks a DINO-proposed label against the detection's real 3D
    shape (from the lidar cluster bounding box, see
    ros_utils.localize_detection_3d). Only special-cases the sofa/column
    confusion that's been directly confirmed live; every other label passes
    through unchanged with no penalty.

    size: (3,) array-like of (width, depth, height) in meters.
    Returns: (corrected_label: str, confidence_multiplier: float in [0,1])
    """
    w, d, h = float(size[0]), float(size[1]), float(size[2])
    footprint = w * d
    aspect_xy = max(w, d) / max(min(w, d), 1e-3)
    slenderness = h / max(min(w, d), 1e-3)

    is_column_shaped = (
        slenderness > GEOMETRIC_CHECK_COLUMN_SLENDERNESS_MIN
        and footprint < GEOMETRIC_CHECK_COLUMN_FOOTPRINT_MAX_M2
        and aspect_xy < GEOMETRIC_CHECK_COLUMN_ASPECT_MAX
    )
    is_sofa_shaped = (
        h < GEOMETRIC_CHECK_SOFA_HEIGHT_MAX_M
        and footprint > GEOMETRIC_CHECK_SOFA_FOOTPRINT_MIN_M2
        and aspect_xy > GEOMETRIC_CHECK_SOFA_ASPECT_MIN
    )

    label_lower = label.lower()
    if "sofa" in label_lower and is_column_shaped and not is_sofa_shaped:
        return "column", 0.0   # geometry flatly contradicts "sofa" -- hard override
    if "sofa" in label_lower and not is_sofa_shaped and not is_column_shaped:
        return label, 0.5      # ambiguous -- demote confidence, don't relabel blind
    if "column" in label_lower and is_sofa_shaped:
        return "sofa", 0.3     # symmetric case, demoted (less common failure direction)

    return label, 1.0


# ---------------------------------------------------------------------------
# Minimum-observation filter for counting (2026-08-12) -- catches noise the
# shape-based checks above CAN'T see.
# ---------------------------------------------------------------------------
# Confirmed live (arabic_room) that some false-positive detections localize
# via localize_detection_3d's SPARSE-LIDAR FALLBACK PATH (single nearest
# point, not a real cluster) rather than the real-cluster path -- when that
# happens, the returned size is the generic FALLBACK_SIZE (0.4,0.4,0.4),
# which looks perfectly plausible to both is_degenerate_cluster() and
# geometric_category_check() (there's no real shape data to contradict a
# label with). This lets exactly the kind of detection those checks exist to
# catch slip through undetected when lidar coverage happens to be sparse at
# that moment.
#
# Independent signal that doesn't depend on shape at all: across repeated
# live runs, genuinely real objects consistently accumulate MANY
# observations as the robot explores near them (4x, 6x, 8x, 9x, 10x seen),
# while noise/false-positive detections are overwhelmingly seen only once or
# twice. Applied only to COUNTING (not find_unique_referent) -- an
# object-reference query can legitimately have just one genuine sighting of
# the correct object, so filtering there risks discarding real answers; a
# numerical count benefits from the extra robustness since it's summing
# across many candidates, where a couple of single-observation false
# positives directly corrupts the total.
MIN_OBSERVATIONS_FOR_COUNT = 2

# REFERENT RELIABILITY PENALTY 2026-08-24: confirmed live -- find_unique_referent()
# was picking a single-sighting, lower-confidence candidate over a well-
# validated one (seen 6x, conf=0.81, matching documented ground truth)
# purely because it had a marginally better raw geometric _relation_score.
# These two constants make a candidate with fewer than
# REFERENT_RELIABLE_OBS observations need a meaningfully BETTER geometric
# fit to win a tie, rather than winning on a few coincidental centimeters.
# Soft penalty, not a hard filter (unlike MIN_OBSERVATIONS_FOR_COUNT) --
# a genuine single sighting can still win if it's clearly the best fit.
REFERENT_RELIABLE_OBS = 3
REFERENT_LOW_OBS_PENALTY_M = 0.4

# ---------------------------------------------------------------------------
# Shared relation geometry (2026-08-12) -- used by BOTH count() (numerical)
# and find_unique_referent() (object-reference), so all relation types get
# consistent, correct geometry across both question types rather than each
# reimplementing its own (and drifting apart, or one getting fixed while the
# other doesn't).
# ---------------------------------------------------------------------------
# DIRECTIONAL_RADIUS_M: horizontal tolerance for "below"/"above" -- see the
# RADIUS FIX 2026-08-07 history: a loose generic radius applied to a
# directional relation like "below" massively overcounts (confirmed live,
# 21 vs ground truth 2), since "below X" implies close physical alignment
# (same wall, same spot), unlike generic "near X" which reasonably
# tolerates more slack.
DIRECTIONAL_RADIUS_M = 0.8
NEAR_RADIUS_M = 2.0
# "on" implies direct physical contact/resting atop something -- much
# tighter than generic "near". Added 2026-08-12 after a live test caught
# that the generic relation branch only enforced a distance cutoff for
# "near", so "on" silently accepted ANY distance as satisfying the
# relation (just with a worse score) -- meaning a qualified anchor like
# "the book ON THE STOOL" would incorrectly accept a book on the far side
# of the room as "on" the stool, defeating the whole point of the
# qualifier filter in _resolve_anchor_nodes.
ON_RADIUS_M = 1.0
# TUNING NOTE 2026-08-12: originally set to 0.6m as a tight "direct
# contact" estimate. Real detected data from arabic_room (single-
# observation "book" detection, nearest stool 0.73m away) showed this was
# already too tight for real single-observation localization noise before
# even deploying -- loosened to 1.0m. Still much tighter than generic
# NEAR_RADIUS_M (2.0m), appropriately strict for "on" while tolerant of
# the localization noise repeatedly measured this session (single-
# observation detections routinely drift 0.1-0.3m+ from true position).

# BETWEEN geometry: confirmed live ("find the wall lamp between a door
# frame and a window") that scoring "between" as mere proximity to both
# anchor categories (whether summed over all instances or just the nearest
# instance of each) is NOT the same as genuine betweenness -- a candidate
# can be close to both an anchor door and an anchor window without being
# positioned between them at all. Real betweenness requires the candidate
# to project onto the segment connecting a specific door instance and a
# specific window instance, close to that segment (not just close to its
# endpoints). BETWEEN_T_SLACK allows a little room beyond the two anchors'
# own positions (real objects have physical extent, not point positions).
# BETWEEN_PERP_TOLERANCE_FRAC is relative to the anchor-pair separation
# distance, not a fixed meters value, so this scales correctly whether the
# door and window are 1m apart or 6m apart -- a fixed-meters tolerance
# would be too loose for close anchor pairs and too strict for far ones.
BETWEEN_T_SLACK = 0.15
BETWEEN_PERP_TOLERANCE_FRAC = 0.35

# ---------------------------------------------------------------------------
# Qualified anchors (2026-08-12) -- e.g. "the book ON THE STOOL" as an
# anchor for "closest to", not just any book.
# ---------------------------------------------------------------------------
# Confirmed live ("find the pillow closest to the book on the stool") that
# treating "book" and "stool" as two interchangeable anchor categories in
# one OR-pool (the previous behavior) answers a DIFFERENT question --
# "closest to whichever is nearer, any book or any stool" -- not "closest
# to the specific book that is on a stool". AnchorSpec mirrors
# question_parser.Anchor's shape but is defined locally rather than
# imported, preserving this module's existing design of staying free of a
# question_parser dependency (keeps it testable without the LLM/Ollama
# stack -- see the module docstring). main_node.py bridges between the two
# by converting question_parser.Anchor -> AnchorSpec when building the
# relations list passed into count()/find_unique_referent().
@dataclass
class AnchorSpec:
    category: str
    qualifier_type: Optional[str] = None
    qualifier_category: Optional[str] = None


@dataclass
class ObjectNode:
    label: str
    color: str
    position: np.ndarray       # (3,) map frame
    size: np.ndarray           # (3,) rough bbox extent, meters
    observation_count: int = 1
    confidence: float = 0.0


class SceneGraph:
    def __init__(self):
        self.nodes: List[ObjectNode] = []

    def add_or_merge(self, label: str, color: str, position: np.ndarray, score: float, size=(0.4, 0.4, 0.4)):
        """Adds a new node, or if a matching node already exists nearby,
        merges into it (running position average + observation count bump).
        This is what keeps repeated sightings of the same physical object
        across frames from being double-counted -- critical for the
        numerical question type.

        GEOMETRIC CHECK 2026-08-12: `size` now comes from the real lidar
        cluster (see ros_utils.localize_detection_3d), not always the old
        fixed (0.4,0.4,0.4) placeholder. Before trusting DINO's label,
        cross-check it against this real 3D shape -- catches the confirmed
        column-labeled-as-sofa case (and the symmetric case) before a bad
        label ever enters the graph, rather than trying to clean it up
        after the fact. A demoted (not relabeled) detection still enters
        the graph, just with lower confidence, since it may legitimately be
        a partially-observed/ambiguous real object rather than a true
        misclassification.
        """
        size = np.array(size, dtype=np.float32)

        # TEMP DEBUG 2026-08-17: targeted diagnostic for "column" detections
        # specifically -- confirmed live that "column" (explicitly searched
        # for via path_constraints vocabulary, not just a sofa-relabel
        # target) has now failed to appear in the scene graph across 3
        # consecutive runs despite the official ground-truth image
        # confirming 2 real, prominent columns exist in this scene. This
        # print shows every "column"-labeled detection BEFORE any filtering,
        # so we can tell whether DINO simply never proposes it (a detection/
        # vocabulary-recall problem) vs proposes it but our filters reject
        # it (a filter-tuning problem) -- REMOVE once root cause is found.
        if "column" in label.lower():
            print(f"[DEBUG column] raw label={label!r} pos=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) "
                  f"size=(w={size[0]:.3f}, d={size[1]:.3f}, h={size[2]:.3f}) score={score:.3f} "
                  f"is_degenerate={is_degenerate_cluster(size, label)}")

        # GENERAL FILTER 2026-08-12: reject implausibly thin/sliver clusters
        # for ANY category before any label-specific logic runs -- see
        # is_degenerate_cluster() above. This is what generalizes across
        # scenes/categories; the sofa/column check below is a secondary,
        # category-specific layer on top.
        if is_degenerate_cluster(size, label):
            return None

        label, confidence_mult = geometric_category_check(label, size)
        score = score * confidence_mult
        for node in self.nodes:
            if node.label == label and np.linalg.norm(node.position - position) < DETECTION_DEDUP_DIST_M:
                n = node.observation_count
                node.position = (node.position * n + position) / (n + 1)
                node.observation_count += 1
                node.confidence = max(node.confidence, score)
                return node
        node = ObjectNode(label=label, color=color, position=position, size=size, confidence=score)
        self.nodes.append(node)
        return node

    def consolidate(self):
        """Re-merge nodes that ended up as separate despite being close
        together, independent of insertion order.

        BUG FOUND 2026-08-07: add_or_merge() only ever compares a new
        detection against the CURRENT running-average position of existing
        nodes at the moment it arrives -- greedy, insertion-order-dependent
        merging. Confirmed live (arabic_room): two "black sofa" detections
        only 0.57m apart (well under DETECTION_DEDUP_DIST_M=1.2m) ended up
        as separate nodes, because by the time the second one arrived, the
        first had already merged into a different node whose running
        average had drifted elsewhere -- the two never got compared
        directly against each other. With many same-label detections
        scattered across a room, this ordering effect can silently leave
        genuinely-mergeable nodes split, inflating counts.

        This does one full pass, order-independent: repeatedly merge any
        pair of same-label nodes within DETECTION_DEDUP_DIST_M until no
        more merges are possible. Call this once before querying the scene
        graph for an answer (see main_node.py's _answer), not on every
        detection -- it's O(n^2) per pass and the graph is small enough
        per-question that this is cheap at answer time but wasteful to run
        on every incoming detection.
        """
        merged_any = True
        while merged_any:
            merged_any = False
            for i in range(len(self.nodes)):
                for j in range(i + 1, len(self.nodes)):
                    a, b = self.nodes[i], self.nodes[j]
                    if a.label == b.label and np.linalg.norm(a.position - b.position) < DETECTION_DEDUP_DIST_M:
                        n_a, n_b = a.observation_count, b.observation_count
                        a.position = (a.position * n_a + b.position * n_b) / (n_a + n_b)
                        a.observation_count = n_a + n_b
                        a.confidence = max(a.confidence, b.confidence)
                        del self.nodes[j]
                        merged_any = True
                        break
                if merged_any:
                    break

    def _matches(self, node: ObjectNode, category: Optional[str], attributes: List[str]) -> bool:
        # SAFETY NET 2026-08-07: matching was a raw substring check, which
        # silently fails on a plain singular/plural mismatch -- "sofas" is
        # NOT a substring of "a sofa" (missing trailing 's'), so a category
        # or anchor coming back plural from the parser (regex fallback, or
        # an LLM not perfectly following the singular-only instruction)
        # would make every candidate and every anchor match nothing, with
        # no error, just a silently wrong 0. Confirmed live: a fully
        # populated scene graph with obvious sofas/windows still returned
        # count=0 for "how many sofas are below a window?". Strip a
        # trailing 's' before comparing as a cheap, dependency-free safety
        # net on top of the prompt fix (see SYSTEM_PROMPT in
        # question_parser.py) -- doesn't handle irregular plurals, but
        # covers the common case without pulling in a real NLP lib.
        def _singularize(word: str) -> str:
            word = word.lower().strip()
            return word[:-1] if word.endswith("s") and not word.endswith("ss") else word

        if category and _singularize(category) not in _singularize(node.label):
            return False
        for attr in attributes:
            if _singularize(attr) != _singularize(node.color):
                return False
        return True

    def find_matching(self, category: Optional[str], attributes: List[str]) -> List[ObjectNode]:
        return [n for n in self.nodes if self._matches(n, category, attributes)]

    def _resolve_anchor_nodes(self, anchor: AnchorSpec) -> List[ObjectNode]:
        """Returns every node matching anchor.category, additionally
        filtered/selected down based on anchor's own qualifying relation,
        if it has one.

        Two qualifier behaviors, matching how the two relation FAMILIES
        actually work:

        1. THRESHOLD qualifiers ("on", "below", "above", "near", "between"):
           filter candidates down to those that satisfy the relation at
           all (e.g. anchor=book qualified by "on the stool" -> only books
           actually on some stool). Multiple candidates can survive.

        2. SUPERLATIVE qualifiers ("far_from"/"furthest_from"/"furthest",
           "closest_to"/"nearest"): confirmed live ("go to the potted
           plant FURTHEST from the hookah") that these describe picking
           the SINGLE best-matching instance by comparison, not a
           pass/fail threshold -- "far_from" with no selection logic
           previously let EVERY instance silently pass (no threshold
           existed for it at all), so the qualifier did nothing. Select
           the one candidate that maximizes (far_from family) or
           minimizes (closest_to family) distance to the nearest
           qualifier-category instance, and return just that one node.
        """
        candidates = self.find_matching(anchor.category, [])
        if not candidates:
            return []
        if anchor.qualifier_type and anchor.qualifier_category:
            qualifier_nodes = self.find_matching(anchor.qualifier_category, [])
            if not qualifier_nodes:
                return []
            qt = anchor.qualifier_type
            if qt in ("far_from", "furthest_from", "furthest"):
                def dist(c):
                    return min(np.linalg.norm(c.position - q.position) for q in qualifier_nodes)
                return [max(candidates, key=dist)]
            if qt in ("closest_to", "nearest"):
                def dist(c):
                    return min(np.linalg.norm(c.position - q.position) for q in qualifier_nodes)
                return [min(candidates, key=dist)]
            qualifier_anchor = AnchorSpec(category=anchor.qualifier_category)
            candidates = [
                c for c in candidates
                if self._relation_score(c, qt, [qualifier_anchor]) is not None
            ]
        return candidates

    def _relation_score(self, candidate: ObjectNode, relation_type: str, anchor_categories: List[AnchorSpec]) -> Optional[float]:
        """Returns a non-negative score for how well `candidate` satisfies
        this single relation (lower = better match), or None if the
        relation is NOT satisfied at all (no valid anchor, or -- for
        "between" -- no anchor pair the candidate genuinely sits between).
        This is the single shared implementation of relation semantics used
        by both count() (a candidate counts iff every relation returns a
        non-None score) and find_unique_referent() (candidates are ranked
        by their summed score across all relations) -- fixing or improving
        relation geometry here automatically improves both question types
        at once, instead of the two drifting apart.

        anchor_categories: List[AnchorSpec] -- each may itself be qualified
        by its own relation (see AnchorSpec/_resolve_anchor_nodes above);
        resolution of qualified anchors happens transparently via
        _resolve_anchor_nodes, so the rest of this method doesn't need to
        know or care whether a given anchor is qualified.
        """
        # SYNONYM NORMALIZATION 2026-08-17: confirmed live risk before it
        # became a bug -- "the stool UNDER the picture" would otherwise
        # fall through to the generic proximity branch below (no height
        # check at all) rather than the directional below/above logic,
        # since only the literal words "below"/"above" were recognized.
        # Normalizing common synonyms here, in code, is more reliable than
        # depending on the LLM to always emit the exact canonical word --
        # this session repeatedly found llama3.2:3b doesn't reliably
        # follow such instructions even when the prompt states them.
        if relation_type in ("under", "underneath", "beneath"):
            relation_type = "below"
        elif relation_type in ("over", "atop", "on top of"):
            relation_type = "above"

        if relation_type == "between" and len(anchor_categories) >= 2:
            nodes_a = self._resolve_anchor_nodes(anchor_categories[0])
            nodes_b = self._resolve_anchor_nodes(anchor_categories[1])
            if not nodes_a or not nodes_b:
                return None
            best_perp = None
            for a in nodes_a:
                for b in nodes_b:
                    ab = b.position - a.position
                    ab_len = np.linalg.norm(ab)
                    if ab_len < 1e-6:
                        continue  # degenerate pair (a and b at the same spot) -- not a meaningful segment
                    t = float(np.dot(candidate.position - a.position, ab) / (ab_len ** 2))
                    if t < -BETWEEN_T_SLACK or t > 1.0 + BETWEEN_T_SLACK:
                        continue  # projects outside the segment (even with slack) -- not between this pair
                    closest_on_segment = a.position + np.clip(t, 0.0, 1.0) * ab
                    perp = float(np.linalg.norm(candidate.position - closest_on_segment))
                    if perp > BETWEEN_PERP_TOLERANCE_FRAC * ab_len:
                        continue  # too far off the line connecting this specific pair
                    if best_perp is None or perp < best_perp:
                        best_perp = perp
            return best_perp  # None if no anchor pair satisfies genuine betweenness

        if relation_type in ("below", "above") and anchor_categories:
            anchor_nodes = self._resolve_anchor_nodes(anchor_categories[0])
            if not anchor_nodes:
                return None
            best = None
            for a in anchor_nodes:
                horiz = float(np.hypot(candidate.position[0] - a.position[0], candidate.position[1] - a.position[1]))
                if horiz >= DIRECTIONAL_RADIUS_M:
                    continue
                if relation_type == "below" and not (candidate.position[2] < a.position[2] - 0.1):
                    continue
                if relation_type == "above" and not (candidate.position[2] > a.position[2] + 0.1):
                    continue
                if best is None or horiz < best:
                    best = horiz
            return best

        if anchor_categories:
            # Generic proximity relations ("near", "closest_to", "on",
            # "far_from") -- use the nearest instance across all named
            # anchor categories (each possibly qualified -- resolved via
            # _resolve_anchor_nodes).
            anchor_nodes = []
            for a in anchor_categories:
                anchor_nodes.extend(self._resolve_anchor_nodes(a))
            if not anchor_nodes:
                return None
            best = min(float(np.linalg.norm(candidate.position - a.position)) for a in anchor_nodes)
            if relation_type == "near" and best >= NEAR_RADIUS_M:
                return None
            if relation_type == "on" and best >= ON_RADIUS_M:
                return None
            return best

        return None

    def count(self, category: Optional[str], attributes: List[str],
              relations: Optional[List[Tuple[str, List[AnchorSpec]]]] = None) -> int:
        """relations: list of (relation_type, anchor_categories) tuples --
        e.g. [("below", ["window"])] or [("between", ["door frame", "window"])].
        A candidate counts iff it satisfies EVERY relation (matches the
        semantics of the original question, e.g. "sofas below a window").
        """
        candidates = self.find_matching(category, attributes)
        # MIN-OBSERVATION FILTER 2026-08-12: see MIN_OBSERVATIONS_FOR_COUNT
        # comment above -- catches noise (e.g. sparse-lidar-fallback
        # detections with generic placeholder size) that the shape-based
        # checks in add_or_merge can't see, since they have no real shape
        # data to check against. Only applied here (counting), not in
        # find_unique_referent, since a numerical count is summing across
        # many candidates where a couple of single-observation false
        # positives directly corrupts the total -- an object-reference
        # query can legitimately have just one genuine sighting of the
        # right answer.
        candidates = [c for c in candidates if c.observation_count >= MIN_OBSERVATIONS_FOR_COUNT]
        if not relations:
            return len(candidates)
        matched = []
        for c in candidates:
            if all(self._relation_score(c, rtype, anchors) is not None for rtype, anchors in relations):
                matched.append(c)
        return len(matched)

    def find_unique_referent(self, category: Optional[str], attributes: List[str],
                              relations: Optional[List[Tuple[str, List[AnchorSpec]]]] = None) -> Optional[ObjectNode]:
        """Object-reference queries guarantee a single correct answer in the
        scene -- if multiple candidates match, use the parsed relations
        (e.g. "between a door frame and a window", "closest to the fridge")
        to break the tie. relations: list of (relation_type,
        anchor_categories) tuples, same format as count() -- uses the same
        shared _relation_score so both question types get identical,
        correct relation geometry.
        """
        candidates = self.find_matching(category, attributes)
        if not candidates:
            return None
        if len(candidates) == 1 or not relations:
            return candidates[0]

        scored = []
        for c in candidates:
            scores = [self._relation_score(c, rtype, anchors) for rtype, anchors in relations]
            if any(s is None for s in scores):
                continue  # doesn't satisfy every relation -- not a valid candidate
            geo_score = sum(scores)
            # REFERENT RELIABILITY PENALTY 2026-08-24: see constant comment
            # above -- a candidate short of REFERENT_RELIABLE_OBS sightings
            # gets an added penalty so it needs a clearly better geometric
            # fit to win against a more-observed candidate, not just a
            # coincidentally closer one.
            reliability_penalty = max(0, REFERENT_RELIABLE_OBS - c.observation_count) * REFERENT_LOW_OBS_PENALTY_M
            scored.append((c, geo_score + reliability_penalty))

        if not scored:
            # SAFETY NET: nothing satisfies every relation strictly (e.g. a
            # relevant anchor was never detected this run -- an exploration-
            # coverage gap, not a logic bug). Fall back to the first
            # category-only match rather than returning nothing, since a
            # best-effort answer is better than silence.
            return candidates[0]
        return min(scored, key=lambda item: item[1])[0]

    def nodes_by_category(self, category: str) -> List[ObjectNode]:
        return self.find_matching(category, [])

    def summary(self) -> str:
        return "\n".join(
            f"  {n.label} [{n.color}] at ({n.position[0]:.2f}, {n.position[1]:.2f}, {n.position[2]:.2f}), "
            f"seen {n.observation_count}x, conf={n.confidence:.2f}"
            for n in self.nodes
        )
