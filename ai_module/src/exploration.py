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

from config import (
    MIN_FRONTIER_CLUSTER_SIZE,
    OCC_GRID_CELL_SIZE_M,
    OCC_GRID_RADIUS_M,
)


class FrontierExplorer:
    def __init__(self):
        self.visited_cells = set()  # set of (ix, iy) grid cells we've had terrain data for

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round(x / OCC_GRID_CELL_SIZE_M)), int(round(y / OCC_GRID_CELL_SIZE_M)))

    def update_known_cells(self, terrain_xyz: np.ndarray):
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
        best_centroid, best_dist = None, float("inf")
        for cluster in clusters:
            cx, cy = cluster_centroid(cluster)
            dist = np.hypot(cx - rx, cy - ry)
            if dist < best_dist:
                best_dist, best_centroid = dist, (cx, cy)
        return best_centroid
