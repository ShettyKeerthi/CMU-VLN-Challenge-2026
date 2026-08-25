"""
Geometric frontier exploration over /terrain_map_ext.

Measured on livingroom_1 (8793 points):
    intensity = height above estimated ground plane, in metres
    range 0.000 .. 1.500 (1.5 is a hard clamp in the terrain analysis)
    percentiles [5,25,50,75,95,99] = 0.00, 0.00, 0.145, 0.453, 0.88, 1.293
    fraction below 0.2 = 0.547
So intensity < 0.2 separates floor from obstacles cleanly.

Three cell sets are tracked, and the distinction is the whole point:
    known     terrain data seen here
    free      known AND height below threshold  -> drivable
    obstacle  known AND NOT free                -> wall, furniture, step

A frontier is a FREE cell adjacent to UNKNOWN. Treating any known cell as
explorable makes the robot chase wall cells forever, since the space behind a
wall is permanently unknown.

Deliberately NOT semantically-scored (no VLFM-style language-conditioned
frontier ranking) -- get this loop working and measured first.
"""

import os
import time
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from config import (
    MIN_FRONTIER_CLUSTER_SIZE,
    OCC_GRID_CELL_SIZE_M,
    OCC_GRID_RADIUS_M,
)

# Height above ground below which a cell is drivable. Sits just above the p25
# floor mass and below the climb into obstacles.
FREE_HEIGHT_THRESH_M = 0.20
MIN_GOAL_DIST_M = 1.5
# The terrain analysis clamps here; anything taller reports as this value.
TERRAIN_CLAMP_M = 1.50

# Reject frontier cells that touch an obstacle. A free cell hard against a wall
# is a wall margin, not an opening. Doorway centres are not obstacle-adjacent,
# so real gaps survive this filter.
REJECT_WALL_ADJACENT = True

GOAL_TIMEOUT_S = 20.0     # pursue one frontier this long before blacklisting it
GOAL_REACHED_M = 0.8

Cell = Tuple[int, int]
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))


class FrontierExplorer:
    def __init__(self, debug_dir: Optional[str] = None):
        self.known_cells: Set[Cell] = set()
        self.free_cells: Set[Cell] = set()
        self.obstacle_cells: Set[Cell] = set()
        self.blacklist: Set[Cell] = set()

        self.goal: Optional[Tuple[float, float]] = None
        self.goal_cell: Optional[Cell] = None
        self.goal_started: float = 0.0
        self.complete = False

        self.debug_dir = debug_dir
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        self._debug_seq = 0
        self._last_frontiers: List[Cell] = []

    # ---------------------------------------------------------------- grid

    def _world_to_cell(self, x: float, y: float) -> Cell:
        return (int(round(x / OCC_GRID_CELL_SIZE_M)),
                int(round(y / OCC_GRID_CELL_SIZE_M)))

    def _cell_to_world(self, cell: Cell) -> Tuple[float, float]:
        return (cell[0] * OCC_GRID_CELL_SIZE_M, cell[1] * OCC_GRID_CELL_SIZE_M)

    def update_known_cells(self, terrain: np.ndarray,
                           height_thresh: float = FREE_HEIGHT_THRESH_M):
        """terrain: (N, 4) array of x, y, z, height_above_ground.

        Read it with the field names spelled out -- /terrain_map_ext has a
        4-byte pad after z (intensity is at offset 16, not 12):

            pc2.read_points_numpy(msg, field_names=('x','y','z','intensity'))
        """
        if terrain is None or len(terrain) == 0:
            return
        if terrain.shape[1] < 4:
            raise ValueError(
                "terrain needs 4 columns (x, y, z, height). Without the height "
                "channel every cell looks drivable and the explorer will chase walls.")

        xy = np.round(terrain[:, :2] / OCC_GRID_CELL_SIZE_M).astype(np.int32)
        drivable = terrain[:, 3] < height_thresh

        all_cells = set(map(tuple, xy))
        free_now = set(map(tuple, xy[drivable]))

        self.known_cells |= all_cells
        self.free_cells |= free_now
        # A cell seen free once stays free; obstacles are what was never free.
        self.obstacle_cells = self.known_cells - self.free_cells

    # ------------------------------------------------------------ frontier

    def _frontier_cells(self, robot_xy: Tuple[float, float]) -> List[Cell]:
        rx, ry = self._world_to_cell(*robot_xy)
        radius = int(OCC_GRID_RADIUS_M / OCC_GRID_CELL_SIZE_M)

        frontiers = []
        for cell in self.free_cells:
            if cell in self.blacklist:
                continue
            if abs(cell[0] - rx) > radius or abs(cell[1] - ry) > radius:
                continue
            if not any((cell[0] + dx, cell[1] + dy) not in self.known_cells
                       for dx, dy in _N4):
                continue
            if REJECT_WALL_ADJACENT and any(
                    (cell[0] + dx, cell[1] + dy) in self.obstacle_cells
                    for dx, dy in _N4):
                continue
            frontiers.append(cell)

        self._last_frontiers = frontiers
        return frontiers

    def _cluster(self, cells: Iterable[Cell]) -> List[List[Cell]]:
        """8-connected flood fill. No scipy dependency."""
        remaining = set(cells)
        clusters = []
        while remaining:
            seed = remaining.pop()
            cluster, stack = [seed], [seed]
            while stack:
                cx, cy = stack.pop()
                for dx, dy in _N8:
                    n = (cx + dx, cy + dy)
                    if n in remaining:
                        remaining.remove(n)
                        cluster.append(n)
                        stack.append(n)
            clusters.append(cluster)
        return clusters

    # -------------------------------------------------------------- policy

    def report_failed(self, radius_cells: int = 2):
        """Blacklist the current goal and its surroundings, then drop it."""
        if self.goal_cell is None:
            return
        gx, gy = self.goal_cell
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                self.blacklist.add((gx + dx, gy + dy))
        self.goal = self.goal_cell = None

    def next_waypoint(self, robot_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Returns a goal, or None once exploration is finished.

        Sticky: keeps the current goal until reached, timed out, or
        blacklisted. Re-picking every call makes the robot oscillate between
        two nearly-equidistant clusters and cover no ground.
        """
        now = time.time()

        if self.goal is not None:
            if np.hypot(self.goal[0] - robot_xy[0],
                        self.goal[1] - robot_xy[1]) < GOAL_REACHED_M:
                self.goal = self.goal_cell = None
            elif now - self.goal_started > GOAL_TIMEOUT_S:
                self.report_failed()
            else:
                return self.goal

        frontier_cells = self._frontier_cells(robot_xy)
        if not frontier_cells:
            self.complete = True
            return None

        clusters = [c for c in self._cluster(frontier_cells)
                    if len(c) >= MIN_FRONTIER_CLUSTER_SIZE]
        if not clusters:
            # No fallback to single-cell clusters on purpose: chasing one-cell
            # noise is indistinguishable from being finished, and it means this
            # function can never report completion.
            self.complete = True
            return None

        rx, ry = robot_xy
        best, best_dist = None, float("inf")
        skipped_near = 0
        for cluster in clusters:
            arr = np.array(cluster, dtype=np.float32) * OCC_GRID_CELL_SIZE_M
            cx, cy = arr.mean(axis=0)
            d = np.hypot(cx - rx, cy - ry)
            if d < MIN_GOAL_DIST_M:
                skipped_near += 1
                continue
            if d < best_dist:
                best_dist, best = d, (float(cx), float(cy))

        if best is None:
            self.complete = True
            return None

        self.goal = best
        self.goal_cell = self._world_to_cell(*best)
        self.goal_started = now
        return self.goal

    # --------------------------------------------------------------- debug

    def stats(self) -> str:
        known = len(self.known_cells) or 1
        return (f"known={len(self.known_cells)} "
                f"free={len(self.free_cells)} ({len(self.free_cells)/known:.0%}) "
                f"obst={len(self.obstacle_cells)} "
                f"frontier={len(self._last_frontiers)} "
                f"blacklist={len(self.blacklist)} "
                f"goal={self.goal} complete={self.complete}")

    def save_debug_image(self, robot_xy: Tuple[float, float],
                         scale: int = 4, tag: str = "") -> Optional[str]:
        """Render the grid to a PNG.

            black   unknown
            white   free / drivable
            grey    obstacle (known, not free)
            red     frontier
            orange  blacklisted
            blue    robot
            green   current goal

        Read it like this:
          - grey forming continuous lines around the room -> walls are in the
            cloud, everything is behaving
          - grey only as furniture blobs with black past the floor edge -> the
            terrain analysis drops tall points, so wall lines read as unknown;
            the blacklist is then your only defence and GOAL_TIMEOUT_S matters
          - red tracing the room outline -> FREE_HEIGHT_THRESH_M is too high
          - red only in doorways and gaps -> correct
        """
        if not self.debug_dir or not self.known_cells:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None

        xs = [c[0] for c in self.known_cells]
        ys = [c[1] for c in self.known_cells]
        x0, x1 = min(xs) - 2, max(xs) + 2
        y0, y1 = min(ys) - 2, max(ys) + 2
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w <= 0 or h <= 0 or w * h > 4_000_000:
            return None

        img = np.zeros((h, w, 3), dtype=np.uint8)

        def put(cell, rgb):
            ix, iy = cell[0] - x0, cell[1] - y0
            if 0 <= ix < w and 0 <= iy < h:
                img[iy, ix] = rgb

        for c in self.obstacle_cells:
            put(c, (90, 90, 90))
        for c in self.free_cells:
            put(c, (245, 245, 245))
        for c in self._frontier_cells(robot_xy):
            put(c, (220, 40, 40))
        for c in self.blacklist:
            if c in self.known_cells:
                put(c, (235, 140, 30))
        if self.goal_cell:
            for dx, dy in _N8 + ((0, 0),):
                put((self.goal_cell[0] + dx, self.goal_cell[1] + dy), (40, 200, 80))
        rc = self._world_to_cell(*robot_xy)
        for dx, dy in _N8 + ((0, 0),):
            put((rc[0] + dx, rc[1] + dy), (60, 120, 255))

        img = np.flipud(img)   # +y up, image convention
        pil = Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST)

        self._debug_seq += 1
        name = f"grid_{self._debug_seq:04d}{('_' + tag) if tag else ''}.png"
        path = os.path.join(self.debug_dir, name)
        pil.save(path)
        return path
