"""
Converts a parsed instruction-following QuerySpec (near / avoid / between
constraints over named landmarks) into an ordered sequence of Pose2D
waypoints, using whatever landmark nodes currently exist in the scene graph.

This is intentionally simple: it does NOT do full motion planning against
the terrain map (the base autonomy stack already handles local obstacle
avoidance and moving out-of-bounds waypoints into the traversable area --
see README "System Inputs"). It only decides *where* the waypoints should
be, in what order, respecting near/avoid/between semantics.
"""

from typing import List, Optional, Tuple

import numpy as np

from .question_parser import Anchor, PathConstraint
from .scene_graph import AnchorSpec, SceneGraph


def _anchor_positions(scene_graph: SceneGraph, anchor: Anchor) -> List[np.ndarray]:
    """Resolves a path-constraint anchor to real scene-graph positions.

    QUALIFIED-ANCHOR REWRITE 2026-08-13: previously did its own crude
    word-stripping ("the blue trash can" -> guess the category is the last
    word) and a bare category match -- confirmed live that this silently
    dropped qualifiers entirely (e.g. "the tray ON THE TABLE" would just
    become "tray", matching ANY tray in the room) and had no way to
    express superlative selection ("the potted plant FURTHEST from the
    hookah"). Anchor is now the same structured type used by relations
    (see question_parser.Anchor / SYSTEM_PROMPT's WORKED EXAMPLE 2), so
    this just bridges to scene_graph.AnchorSpec and reuses
    _resolve_anchor_nodes -- the SAME qualifier/superlative resolution
    logic validated for object-reference and numerical questions, instead
    of a separate, weaker implementation living only in this module.
    """
    spec = AnchorSpec(category=anchor.category, qualifier_type=anchor.qualifier_type, qualifier_category=anchor.qualifier_category)
    nodes = scene_graph._resolve_anchor_nodes(spec)
    return [n.position for n in nodes]


def build_waypoint_sequence(
    constraints: List[PathConstraint],
    scene_graph: SceneGraph,
    start_xy: Tuple[float, float],
    near_offset_m: float = 1.0,
    avoid_margin_m: float = 1.5,
) -> List[Tuple[float, float]]:
    """Returns an ordered list of (x, y) waypoints in the map frame.

    - "near" anchors -> waypoint placed near_offset_m from the anchor, on the
      side closest to the previous waypoint (so the path actually passes it).
      Also used for "go to X" / "stop at X" steps (see SYSTEM_PROMPT WORKED
      EXAMPLE 2) -- X may itself be a qualified anchor (superlative
      selection or a nested relation), resolved via _anchor_positions.
    - "between" anchors (exactly two) -> waypoint at their midpoint.
    - "avoid" anchors don't produce their own waypoint; they're used as a
      keep-out check against every other waypoint, nudging any waypoint that
      falls inside avoid_margin_m radially outward.
    """
    ordered = sorted(
        [c for c in constraints if c.type != "avoid"],
        key=lambda c: (c.order if c.order is not None else 999),
    )
    avoid_positions: List[np.ndarray] = []
    for c in constraints:
        if c.type == "avoid":
            for anchor in c.anchors:
                avoid_positions.extend(_anchor_positions(scene_graph, anchor))

    waypoints: List[Tuple[float, float]] = []
    prev = np.array(start_xy, dtype=np.float32)

    for constraint in ordered:
        target_pos: Optional[np.ndarray] = None

        if constraint.type == "between" and len(constraint.anchors) >= 2:
            pts_a = _anchor_positions(scene_graph, constraint.anchors[0])
            pts_b = _anchor_positions(scene_graph, constraint.anchors[1])
            if pts_a and pts_b:
                # SELF-PAIR FIX 2026-08-13: confirmed a real bug -- "between
                # the two columns" gives BOTH anchors the SAME category
                # ("column"), so pts_a and pts_b end up as the exact same
                # list of positions. The naive closest-pair search below
                # would then happily pair a column with ITSELF (distance
                # 0), which always wins as "closest", collapsing target_pos
                # to a single column's own position instead of a real
                # midpoint between two distinct columns. Exclude pairs
                # whose two points are (nearly) identical before picking
                # the closest pair -- this only matters when the two
                # anchors share a category, and is a no-op otherwise.
                candidate_pairs = [
                    (a, b) for a in pts_a for b in pts_b
                    if not np.allclose(a[:2], b[:2], atol=1e-3)
                ]
                if not candidate_pairs:
                    # Degenerate case: only one real instance exists total
                    # (e.g. a single detected column so far) -- nothing
                    # better to do than fall back to self-pairing rather
                    # than produce no waypoint at all; this is expected to
                    # self-correct as exploration continues and a second
                    # instance is detected.
                    candidate_pairs = [(a, b) for a in pts_a for b in pts_b]
                best_pair = min(
                    candidate_pairs,
                    key=lambda ab: np.linalg.norm(ab[0][:2] - ab[1][:2]),
                )
                target_pos = (best_pair[0] + best_pair[1]) / 2.0

        elif constraint.type == "near" and constraint.anchors:
            pts = _anchor_positions(scene_graph, constraint.anchors[0])
            if pts:
                # NEAREST-INSTANCE FIX 2026-08-12: same first-match issue as
                # "between" above -- pick the instance CLOSEST to the
                # current path position (the anchor actually on the way),
                # not an arbitrary first match. Note: if constraint.anchors[0]
                # is a QUALIFIED anchor (e.g. superlative "furthest from the
                # hookah"), _anchor_positions has already narrowed `pts`
                # down to that ONE specific instance -- this min() over a
                # single-element list is then a no-op, which is correct.
                anchor_pos = min(pts, key=lambda p: np.linalg.norm(p[:2] - prev[:2]))
                direction = prev[:2] - anchor_pos[:2]
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
                target_pos = anchor_pos.copy()
                target_pos[:2] = anchor_pos[:2] + direction * near_offset_m

        if target_pos is None:
            continue  # anchor not yet in scene graph -- caller should keep exploring, not fabricate a waypoint

        xy = target_pos[:2].copy()
        for avoid_pos in avoid_positions:
            delta = xy - avoid_pos[:2]
            dist = np.linalg.norm(delta)
            if dist < avoid_margin_m and dist > 1e-6:
                xy = avoid_pos[:2] + (delta / dist) * avoid_margin_m

        waypoints.append((float(xy[0]), float(xy[1])))
        prev = np.array([xy[0], xy[1], 0.0], dtype=np.float32)

    return waypoints
