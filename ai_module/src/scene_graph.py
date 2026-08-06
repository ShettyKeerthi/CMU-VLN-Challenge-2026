"""
Incrementally-built open-vocabulary 3D scene graph: a flat set of object
nodes (label, attributes, 3D position, rough size) plus on-demand spatial
relation queries. Deliberately simple (ConceptGraphs-style, not a full
hierarchical region graph like HOV-SG) -- start here, add a region layer
only if flat-object queries prove insufficient on your training scenes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import DETECTION_DEDUP_DIST_M


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

    def add_or_merge(self, label: str, color: str, position: np.ndarray, score: float, approx_size=(0.4, 0.4, 0.4)):
        """Adds a new node, or if a matching node already exists nearby,
        merges into it (running position average + observation count bump).
        This is what keeps repeated sightings of the same physical object
        across frames from being double-counted -- critical for the
        numerical question type.
        """
        size = np.array(approx_size, dtype=np.float32)
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

    def _matches(self, node: ObjectNode, category: Optional[str], attributes: List[str]) -> bool:
        if category and category.lower() not in node.label.lower():
            return False
        for attr in attributes:
            if attr.lower() != node.color.lower():
                return False
        return True

    def find_matching(self, category: Optional[str], attributes: List[str]) -> List[ObjectNode]:
        return [n for n in self.nodes if self._matches(n, category, attributes)]

    def count(self, category: Optional[str], attributes: List[str], near_anchor_category: Optional[str] = None,
              near_radius_m: float = 2.0) -> int:
        candidates = self.find_matching(category, attributes)
        if near_anchor_category:
            anchors = self.find_matching(near_anchor_category, [])
            candidates = [
                c for c in candidates
                if any(np.linalg.norm(c.position - a.position) < near_radius_m for a in anchors)
            ]
        return len(candidates)

    def find_unique_referent(self, category: Optional[str], attributes: List[str],
                              anchor_categories: List[str] = None) -> Optional[ObjectNode]:
        """Object-reference queries guarantee a single correct answer in the
        scene -- if multiple candidates match, use the anchor objects
        (from the parsed relations, e.g. "closest to the fridge") to break
        the tie by picking the candidate nearest to all named anchors.
        """
        candidates = self.find_matching(category, attributes)
        if not candidates:
            return None
        if len(candidates) == 1 or not anchor_categories:
            return candidates[0]

        anchors = []
        for anchor_cat in anchor_categories:
            anchors.extend(self.find_matching(anchor_cat, []))
        if not anchors:
            return candidates[0]

        def total_dist(node):
            return sum(np.linalg.norm(node.position - a.position) for a in anchors)

        return min(candidates, key=total_dist)

    def nodes_by_category(self, category: str) -> List[ObjectNode]:
        return self.find_matching(category, [])

    def summary(self) -> str:
        return "\n".join(
            f"  {n.label} [{n.color}] at ({n.position[0]:.2f}, {n.position[1]:.2f}, {n.position[2]:.2f}), "
            f"seen {n.observation_count}x, conf={n.confidence:.2f}"
            for n in self.nodes
        )
