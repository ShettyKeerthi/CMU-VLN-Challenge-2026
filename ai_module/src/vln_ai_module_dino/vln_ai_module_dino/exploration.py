"""
Geometric frontier exploration: builds a rolling 2D occupancy grid from the
terrain map point cloud and picks the nearest unexplored frontier as the
next waypoint. Deliberately NOT semantically-scored (no VLFM-style
language-conditioned frontier ranking) -- get this loop working and
measured first, then decide whether semantic frontier scoring is worth the
added complexity for your scenes. Blind frontier coverage may well be
enough given scenes are single rooms/small buildings.
"""

from typing import List, Optional, Tuple

import numpy as np

from .config import (
    MAX_TERRAIN_POINT_DIST_FROM_ROBOT_M,
    MIN_FRONTIER_CLUSTER_SIZE,
    OCC_GRID_CELL_SIZE_M,
    OCC_GRID_RADIUS_M,
)


class FrontierExplorer:
    def __init__(self):
        self.visited_cells = set()  # set of (ix, iy) grid cells we've had terrain data for
        # STUCK-DETECTION 2026-08-07: confirmed live (arabic_room) that the
        # robot can get physically wedged (columns, tight furniture layout)
        # while this explorer keeps re-picking the SAME nearest frontier
        # every tick, since "nearest to current position" doesn't change
        # if the robot never actually moves. The result: the whole
        # exploration budget burns sitting in one spot, and the scene graph
        # ends up dominated by whatever's nearby (often false positives)
        # instead of genuine room coverage. main_node.py detects the stall
        # (position not changing) and calls blacklist_current_target() so
        # the next call to next_waypoint() is forced to pick a DIFFERENT
        # frontier instead of retrying the unreachable one forever.
        self._blacklisted_targets: List[Tuple[float, float]] = []
        self._blacklist_radius_m = 1.0

    def blacklist_current_target(self, target: Tuple[float, float]):
        self._blacklisted_targets.append(target)

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round(x / OCC_GRID_CELL_SIZE_M)), int(round(y / OCC_GRID_CELL_SIZE_M)))

    def update_known_cells(self, terrain_xyz: np.ndarray, robot_xy: Optional[Tuple[float, float]] = None):
        # GHOST-FRONTIER FIX 2026-08-07: confirmed live (arabic_room) that
        # frontier targets were being picked way outside the physical room
        # (e.g. (-11.2, 1.79), (-7.87, 12.07) against a real room spanning
        # roughly -4..4 in x, -2.5..7 in y) -- almost certainly noisy/
        # spurious terrain points (sensor multipath, etc.) polluting
        # visited_cells and seeding frontier clusters in places the robot
        # can never actually reach. This wasted real exploration budget
        # (each one burns a full STUCK_CHECK_INTERVAL_SEC before being
        # blacklisted) and is a plausible cause of the "moving in and out"
        # behavior reported live -- a frontier near a real doorway,
        # partially contaminated by a bad far point, can send the robot
        # toward/through it, fail partway, and retreat repeatedly.
        # /terrain_map_ext is documented as ~20m around the vehicle, so any
        # point much farther than that from the robot's CURRENT position at
        # the time of this specific scan is very likely spurious -- reject
        # it before it ever becomes a frontier candidate. This is a scene-
        # agnostic sanity bound (based on the sensor's own documented
        # coverage radius, not this scene's specific layout), so it should
        # generalize rather than being tuned to arabic_room.
        if robot_xy is not None and terrain_xyz.shape[0] > 0:
            dists = np.hypot(terrain_xyz[:, 0] - robot_xy[0], terrain_xyz[:, 1] - robot_xy[1])
            terrain_xyz = terrain_xyz[dists < MAX_TERRAIN_POINT_DIST_FROM_ROBOT_M]
        for x, y, _ in terrain_xyz:
            self.visited_cells.add(self._world_to_cell(x, y))

    def _frontier_cells(self, robot_xy: Tuple[float, float]) -> List[Tuple[int, int]]:
        """A frontier cell is a known (visited) cell with at least one
        unknown neighbor within the search radius around the robot."""
        rx, ry = self._world_to_cell(*robot_xy)
        radius_cells = int(OCC_GRID_RADIUS_M / OCC_GRID_CELL_SIZE_M)
        frontiers = []
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for cell in self.visited_cells:
            if abs(cell[0] - rx) > radius_cells or abs(cell[1] - ry) > radius_cells:
                continue
            for dx, dy in neighbors:
                if (cell[0] + dx, cell[1] + dy) not in self.visited_cells:
                    frontiers.append(cell)
                    break
        return frontiers

    def _cluster_frontiers(self, frontier_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Cheap grid-based clustering (no scipy dependency): union cells
        that are within 1 cell of each other."""
        remaining = set(frontier_cells)
        clusters = []
        while remaining:
            seed = remaining.pop()
            cluster = [seed]
            frontier_stack = [seed]
            while frontier_stack:
                cx, cy = frontier_stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbor = (cx + dx, cy + dy)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            cluster.append(neighbor)
                            frontier_stack.append(neighbor)
            clusters.append(cluster)
        return clusters

    def next_waypoint(self, robot_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        frontier_cells = self._frontier_cells(robot_xy)
        if not frontier_cells:
            return None  # fully explored within radius, or no terrain data yet

        clusters = [c for c in self._cluster_frontiers(frontier_cells) if len(c) >= MIN_FRONTIER_CLUSTER_SIZE]
        if not clusters:
            clusters = self._cluster_frontiers(frontier_cells)  # fall back to any cluster if none big enough
        if not clusters:
            return None

        def cluster_centroid(cluster):
            arr = np.array(cluster, dtype=np.float32) * OCC_GRID_CELL_SIZE_M
            return arr.mean(axis=0)

        rx, ry = robot_xy
        # Rank all clusters by distance, then pick the nearest one that
        # ISN'T blacklisted -- not just the single nearest overall. Without
        # this, a blacklisted-but-still-nearest frontier would just get
        # re-picked immediately after being blacklisted, defeating the
        # point.
        candidates = []
        for cluster in clusters:
            cx, cy = cluster_centroid(cluster)
            dist = np.hypot(cx - rx, cy - ry)
            candidates.append((dist, (cx, cy)))
        candidates.sort(key=lambda item: item[0])

        # DIVERSITY FIX 2026-08-10: confirmed live (arabic_room, two
        # separate runs) that pure nearest-unblacklisted selection keeps the
        # robot stuck exploring one confined local pocket for the ENTIRE
        # budget -- the exploration TARGETS themselves never crossed x=1.6
        # in either run, while the real objects being searched for sat at
        # x=2.1 to 3.6, on the far side of the room. When several nearby
        # frontiers are all genuinely unreachable (behind the same
        # obstacle, say), always picking the next-nearest survivor just
        # keeps retrying variations of the same blocked local area instead
        # of trying a meaningfully different direction. Sample uniformly at
        # random among the K nearest unblacklisted candidates instead of
        # deterministically taking the single nearest -- still biased
        # toward efficiency (only considers genuinely close options), but
        # no longer guaranteed to retry the same local pocket every time,
        # giving the far side of the room a real chance within the budget.
        #
        # BLACKLIST DECAY REVERTED 2026-08-12: a time-based decay (retry a
        # blacklisted frontier after 90s) was tried here, on the hypothesis
        # that permanent blacklisting was needlessly wasting budget on
        # transiently-stuck frontiers. Confirmed live it made numerical
        # accuracy WORSE (1,0,2,1,4 vs the permanent-blacklist baseline's
        # 2,1,1,2,3,2 across comparable trial counts, ground truth 2) --
        # most likely because a GENUINELY unreachable frontier re-entering
        # the candidate pool every 90s costs a full STUCK_CHECK_INTERVAL_SEC
        # (30s) before it gets re-blacklisted, and this can repeat multiple
        # times per run, burning more real exploration budget than the
        # decay recovers. Reverted to permanent blacklisting. If revisited
        # later, a "strike count" approach (allow exactly one retry, then
        # blacklist permanently) would likely be safer than pure time decay
        # -- but that needs its own live validation before trusting it,
        # same as this attempt did.
        unblacklisted = [
            (cx, cy) for dist, (cx, cy) in candidates
            if not any(np.hypot(cx - bx, cy - by) < self._blacklist_radius_m for bx, by in self._blacklisted_targets)
        ]
        if not unblacklisted:
            return None  # every reachable-looking frontier is blacklisted -- genuinely stuck, let the exploration timeout handle it
        pool = unblacklisted[:min(5, len(unblacklisted))]
        idx = np.random.randint(len(pool))
        return pool[idx]
