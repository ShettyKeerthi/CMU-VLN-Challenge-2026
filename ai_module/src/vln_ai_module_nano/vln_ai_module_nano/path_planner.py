"""
Converts a parsed instruction-following QuerySpec into an ordered sequence of
Pose2D waypoints, resolving each leg's OWN destination against the scene graph.

The previous version resolved only `constraint.anchors`, which meant a leg like
"go to the potted plant closest to the candle holder" produced a waypoint at
the candle holder -- the anchor -- rather than at the potted plant. With the
parser now putting the destination in `constraint.target`, each leg is resolved
independently and in order:

    leg 1  goto  potted plant (closest_to: candle holder)  -> waypoint 1
    leg 2  goto  vase         (between: tv, door)          -> waypoint 2 (final)

This is intentionally simple: it does NOT do full motion planning against the
terrain map (the base autonomy stack already handles local obstacle avoidance
and nudges out-of-bounds waypoints into the traversable area -- see README
"System Inputs"). It only decides *where* the waypoints go and in what order.

Waypoints carry a heading, since the output topic is /way_point_with_heading.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from question_parser import ObjectRef, PathConstraint, Relation
from scene_graph import SceneGraph

# Distance to stop clear of a destination's footprint.
STANDOFF_M = 0.7          # was 1.2

# Looser for "via" legs: pass near, do not stop.
VIA_STANDOFF_M = 1.0      # was 1.6
# Approach and exit distance either side of a between-gap midpoint.
GAP_LEAD_M = 1.5
# How far to push a waypoint out of a keep-out region.
AVOID_MARGIN_M = 1.8

Waypoint = Tuple[float, float, float]     # x, y, theta


@dataclass
class Leg:
    """One resolved step, kept so the caller can see what was planned and,
    more importantly, what failed to resolve."""
    index: int
    op: str
    description: str
    waypoints: List[Waypoint] = field(default_factory=list)
    resolved: bool = False
    is_final: bool = False
    node: object = None


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    return (1.0, 0.0) if n < 1e-6 else (dx / n, dy / n)


def _radius(node) -> float:
    """Planar half-diagonal of a node's footprint."""
    try:
        return 0.5 * math.hypot(float(node.size[0]), float(node.size[1]))
    except Exception:
        return 0.3


def resolve_target(scene_graph: SceneGraph, ref: Optional[ObjectRef],
                   verbose: bool = True):
    """ObjectRef -> ObjectNode, using the relation-aware scene graph query.

    The relation travels WITH the target ("the potted plant closest to the
    candle holder"), so it is scored here rather than being flattened into a
    proximity sum over anchors.
    """
    if ref is None or not ref.category:
        return None
    rels = []
    if ref.relation and ref.anchors:
        rels.append(Relation(type=ref.relation, anchors=ref.anchors))
    node = scene_graph.find_unique_referent(ref.category, ref.attributes,
                                            relations=rels, verbose=verbose)
    if node is None and verbose:
        print(f"[planner] unresolved: {ref}")
    return node


def _resolve_anchor(scene_graph: SceneGraph, phrase: str, verbose: bool = False):
    """Single anchor phrase -> best matching node, or None."""
    if not phrase:
        return None
    return scene_graph.find_unique_referent(phrase, [], verbose=verbose)


def approach_point(node, frm: Tuple[float, float], standoff: float = STANDOFF_M,
                   free_check: Optional[Callable[[float, float], bool]] = None
                   ) -> Waypoint:
    """A point clear of the object, on the side we are arriving from.

    `free_check(x, y) -> bool` is optional; pass the explorer's drivability
    test and the point is pushed outward, then swept around other bearings,
    until it lands somewhere the robot can actually stand. Without it the
    standoff is a guess that can place a waypoint inside a wall.
    """
    ox, oy = float(node.position[0]), float(node.position[1])
    ux, uy = _unit(frm[0] - ox, frm[1] - oy)
    base = _radius(node)

    def make(x, y):
        return (x, y, math.atan2(oy - y, ox - x))       # face the object

    for extra in (0.0, 0.25, 0.5):     # was (0.0, 0.4, 0.8, 1.2)
        d = base + standoff + extra
        x, y = ox + ux * d, oy + uy * d
        if free_check is None or free_check(x, y):
            return make(x, y)

    if free_check is not None:
        base_ang = math.atan2(uy, ux)
        for k in range(1, 12):
            for sign in (1, -1):
                ang = base_ang + sign * k * (math.pi / 12)
                d = base + standoff
                x, y = ox + math.cos(ang) * d, oy + math.sin(ang) * d
                if free_check(x, y):
                    return make(x, y)
        print(f"[planner] no free approach to {node.label}; using raw standoff")

    d = base + standoff
    return make(ox + ux * d, oy + uy * d)


def between_points(a, b, frm: Tuple[float, float]) -> List[Waypoint]:
    """Approach -> midpoint -> exit, so the robot transits the gap rather than
    clipping its edge. Scoring is on the driven trajectory, not the waypoints,
    so a single midpoint waypoint is not enough."""
    mx = 0.5 * (float(a.position[0]) + float(b.position[0]))
    my = 0.5 * (float(a.position[1]) + float(b.position[1]))
    ax, ay = _unit(float(b.position[0]) - float(a.position[0]),
                   float(b.position[1]) - float(a.position[1]))
    px, py = -ay, ax                                   # direction of travel

    # Aim the traversal away from where we are, so we pass through the gap.
    if (frm[0] - mx) * px + (frm[1] - my) * py < 0:
        px, py = -px, -py

    ex, ey = mx + px * GAP_LEAD_M, my + py * GAP_LEAD_M
    xx, xy_ = mx - px * GAP_LEAD_M, my - py * GAP_LEAD_M
    heading = math.atan2(xy_ - ey, xx - ex)
    return [(ex, ey, heading), (mx, my, heading), (xx, xy_, heading)]


def _push_out(xy: Tuple[float, float], keepouts: Sequence[np.ndarray],
              margin: float = AVOID_MARGIN_M) -> Tuple[float, float]:
    """Nudge a waypoint radially clear of any keep-out centre."""
    x, y = xy
    for k in keepouts:
        dx, dy = x - float(k[0]), y - float(k[1])
        d = math.hypot(dx, dy)
        if 1e-6 < d < margin:
            ux, uy = dx / d, dy / d
            x, y = float(k[0]) + ux * margin, float(k[1]) + uy * margin
    return x, y


def build_plan(constraints: Sequence[PathConstraint], scene_graph: SceneGraph,
               start_xy: Tuple[float, float],
               free_check: Optional[Callable[[float, float], bool]] = None,
               verbose: bool = True) -> List[Leg]:
    """Ordered legs -> resolved Legs. Unresolved legs are RETAINED with
    resolved=False rather than dropped, so a missing object is visible instead
    of silently shortening the path."""
    legs: List[Leg] = []
    cursor = (float(start_xy[0]), float(start_xy[1]))

    # Keep-outs are collected up front: they constrain every waypoint, not
    # just the leg they appear in.
    keepouts: List[np.ndarray] = []
    for c in constraints:
        if c.type == "avoid_near" and c.target is not None:
            n = resolve_target(scene_graph, c.target, verbose=False)
            if n is not None:
                keepouts.append(n.position)
        elif c.type == "avoid_between":
            a = _resolve_anchor(scene_graph, c.anchors[0]) if c.anchors else None
            b = _resolve_anchor(scene_graph, c.anchors[1]) if len(c.anchors) > 1 else None
            if a is not None and b is not None:
                keepouts.append((np.asarray(a.position) + np.asarray(b.position)) / 2.0)

    travel = [c for c in constraints if c.type in ("goto", "via", "between")]
    final_idx = len(travel) - 1

    for i, c in enumerate(travel):
        is_final = (i == final_idx)

        if c.type in ("goto", "via"):
            node = resolve_target(scene_graph, c.target, verbose)
            if node is None:
                legs.append(Leg(i, c.type, f"{c.type} {c.target} [UNRESOLVED]",
                                [], False, is_final))
                continue

            standoff = STANDOFF_M if c.type == "goto" else VIA_STANDOFF_M
            x, y, th = approach_point(node, cursor, standoff, free_check)
            x, y = _push_out((x, y), keepouts)

            legs.append(Leg(i, c.type,
                            f"{c.type} {node.label} "
                            f"({node.position[0]:.2f}, {node.position[1]:.2f})",
                            [(x, y, th)], True, is_final, node))
            cursor = (x, y)

        else:  # between
            a = _resolve_anchor(scene_graph, c.anchors[0]) if c.anchors else None
            b = _resolve_anchor(scene_graph, c.anchors[1]) if len(c.anchors) > 1 else None
            if a is None or b is None:
                legs.append(Leg(i, c.type, f"between {c.anchors} [UNRESOLVED]",
                                [], False, is_final))
                continue

            wps = between_points(a, b, cursor)
            wps = [(*_push_out((w[0], w[1]), keepouts), w[2]) for w in wps]
            legs.append(Leg(i, c.type, f"between {a.label} and {b.label}",
                            wps, True, is_final))
            cursor = (wps[-1][0], wps[-1][1])

    if verbose:
        print("[planner] plan:")
        if not legs:
            print("  (nothing to execute)")
        for leg in legs:
            state = "ok  " if leg.resolved else "MISS"
            mark = "   <- FINAL" if leg.is_final else ""
            print(f"  {state} {leg.index + 1}. {leg.description} "
                  f"[{len(leg.waypoints)} wp]{mark}")
        if keepouts:
            print(f"  {len(keepouts)} keep-out region(s) applied")
    return legs


def build_waypoint_sequence(
    constraints: Sequence[PathConstraint],
    scene_graph: SceneGraph,
    start_xy: Tuple[float, float],
    free_check: Optional[Callable[[float, float], bool]] = None,
    verbose: bool = True,
) -> List[Waypoint]:
    """Flat (x, y, theta) list in execution order.

    Publish these ONE AT A TIME, waiting for arrival before sending the next.
    Publishing the whole sequence means the base system only ever acts on the
    last one, and reaching constraints out of order is penalised.
    """
    legs = build_plan(constraints, scene_graph, start_xy, free_check, verbose)
    return [wp for leg in legs for wp in leg.waypoints]


def final_target_node(constraints: Sequence[PathConstraint], scene_graph: SceneGraph):
    """The object the robot must finish at, or None. Useful for publishing a
    marker alongside navigation, and for checking the destination resolved
    before driving anywhere."""
    for c in reversed(list(constraints)):
        if c.type == "goto" and c.target is not None:
            n = resolve_target(scene_graph, c.target, verbose=False)
            if n is not None:
                return n
    return None


def missing_categories(constraints: Sequence[PathConstraint],
                       scene_graph: SceneGraph) -> List[str]:
    """Categories the plan needs that are not yet in the scene graph.

    Use as an answer-time gate: keep exploring rather than committing to a path
    with a leg that cannot resolve. Instruction-following is worth 6 points a
    statement, so an extra minute of exploration is cheap by comparison.
    """
    need = set()
    for c in constraints:
        need.update(c.anchors)
        if c.target is not None:
            if c.target.category:
                need.add(c.target.category)
            need.update(c.target.anchors)
    return sorted(cat for cat in need
                  if cat and not scene_graph.find_matching(cat, []))
