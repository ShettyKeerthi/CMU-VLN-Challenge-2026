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

from question_parser import PathConstraint
from scene_graph import SceneGraph


def _anchor_positions(scene_graph: SceneGraph, anchor_phrase: str) -> List[np.ndarray]:
    # Anchor phrases come out of the LLM parse as noun phrases like "the window"
    # or "the blue trash can" -- strip determiners/color words crudely and
    # match against scene graph labels. TODO: reuse the same attribute
    # matching QuerySpec uses for targets, rather than this bare substring
    # match, once you see real anchor phrases from your training questions.
    words = [w for w in anchor_phrase.lower().split() if w not in ("the", "a", "an")]
    category_guess = words[-1] if words else anchor_phrase
    nodes = scene_graph.nodes_by_category(category_guess)
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
                target_pos = (pts_a[0] + pts_b[0]) / 2.0

        elif constraint.type == "near" and constraint.anchors:
            pts = _anchor_positions(scene_graph, constraint.anchors[0])
            if pts:
                anchor_pos = pts[0]
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
