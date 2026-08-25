#!/usr/bin/env python3
"""
Top-level ROS2 node for the CMU-VLN-Challenge ai_module.

State machine per question (a fresh instance of this node is launched per
question per the README -- no state persists across questions):

  WAIT_FOR_QUESTION -> EXPLORE -> ANSWER -> DONE

Subscribes to exactly the topics the README lists as available at test time;
publishes to exactly the topics it lists as accepted inputs to the base
system. Swap CAMERA/LIDAR handling details in ros_utils.py / config.py
first if anything about the real message layout differs from assumptions
documented there.
"""

import io
import re
import time
from enum import Enum, auto

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, PointCloud2
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

from . import config
from .exploration import FrontierExplorer
from .path_planner import build_waypoint_sequence
from .perception import OpenVocabDetector
from .question_parser import QuerySpec, parse_question
from .ros_utils import build_marker_message, localize_detection_3d, pointcloud2_to_xyz_array
from .scene_graph import AnchorSpec, SceneGraph

SUPERLATIVE_QUALIFIER_TYPES = {"far_from", "furthest_from", "furthest", "closest_to", "nearest"}


class State(Enum):
    WAIT_FOR_QUESTION = auto()
    EXPLORE = auto()
    ANSWER = auto()
    DONE = auto()


def image_msg_to_pil(msg: CompressedImage) -> PILImage.Image:
    img = PILImage.open(io.BytesIO(bytes(msg.data)))
    return img.convert("RGB")


class VLNChallengeNode(Node):
    def __init__(self):
        super().__init__("vln_challenge_ai_module")

        self.detector = OpenVocabDetector()
        self.scene_graph = SceneGraph()
        self.explorer = FrontierExplorer()

        self.state = State.WAIT_FOR_QUESTION
        self.query_spec: QuerySpec = None
        self.question_received_at = None
        self.robot_position_map = np.zeros(3, dtype=np.float32)
        self.robot_yaw_deg = 0.0
        self._has_odometry = False
        self.latest_registered_scan = np.empty((0, 3), dtype=np.float32)
        self._image_frame_count = 0
        self._pending_waypoint_queue = []
        self._waypoint_in_flight = None
        self._stuck_check_time = None
        self._stuck_check_position = None
        self._current_target = None
        self._last_detection_xy = None
        self._last_detection_yaw_deg = None

        self.create_subscription(String, config.TOPIC_QUESTION, self.on_question, 1)
        self.create_subscription(CompressedImage, config.TOPIC_IMAGE, self.on_image, 1)
        self.create_subscription(PointCloud2, config.TOPIC_REGISTERED_SCAN, self.on_registered_scan, 1)
        self.create_subscription(PointCloud2, config.TOPIC_TERRAIN_MAP_EXT, self.on_terrain_map, 1)
        self.create_subscription(Odometry, config.TOPIC_STATE_ESTIMATION, self.on_odometry, 5)

        self.waypoint_pub = self.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 1)
        self.marker_pub = self.create_publisher(Marker, config.TOPIC_OBJECT_MARKER, 1)
        self.planned_path_pub = self.create_publisher(Marker, config.TOPIC_PLANNED_PATH_MARKER, 1)
        self.numerical_pub = self.create_publisher(Int32, config.TOPIC_NUMERICAL_RESPONSE, 1)

        self.create_timer(1.0 / config.MAIN_LOOP_HZ, self.on_main_loop)

        self.get_logger().info("VLN challenge ai_module ready, waiting for question...")

    def on_question(self, msg: String):
        if self.state != State.WAIT_FOR_QUESTION:
            return
        self.get_logger().info(f"Question received: {msg.data}")
        self.query_spec = parse_question(msg.data)
        self.get_logger().info(
            f"Parsed spec: type={self.query_spec.question_type!r} "
            f"target_category={self.query_spec.target_category!r} "
            f"target_attributes={self.query_spec.target_attributes!r} "
            f"relations={[(r.type, r.anchors) for r in self.query_spec.relations]!r} "
            f"path_constraints={[(c.type, c.anchors, c.order) for c in self.query_spec.path_constraints]!r}"
        )
        self.question_received_at = time.time()
        self.state = State.EXPLORE

    def on_odometry(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_position_map = np.array([p.x, p.y, p.z], dtype=np.float32)
        q = msg.pose.pose.orientation
        self.robot_yaw_deg = float(np.degrees(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))))
        self._has_odometry = True

    def on_registered_scan(self, msg: PointCloud2):
        self.latest_registered_scan = pointcloud2_to_xyz_array(msg)

    def on_terrain_map(self, msg: PointCloud2):
        xyz = pointcloud2_to_xyz_array(msg)
        self.explorer.update_known_cells(xyz, robot_xy=(self.robot_position_map[0], self.robot_position_map[1]))

    def on_image(self, msg: CompressedImage):
        if self.state not in (State.EXPLORE,):
            return
        self._image_frame_count += 1
        if self._image_frame_count % config.FRAME_SAMPLE_EVERY_N != 0:
            return
        if self._has_odometry and self._last_detection_xy is not None:
            dx = float(self.robot_position_map[0] - self._last_detection_xy[0])
            dy = float(self.robot_position_map[1] - self._last_detection_xy[1])
            moved_m = float(np.hypot(dx, dy))
            yaw_delta = abs(self.robot_yaw_deg - self._last_detection_yaw_deg)
            yaw_delta = min(yaw_delta, 360.0 - yaw_delta)
            if moved_m < config.DETECTION_MIN_MOVE_M and yaw_delta < config.DETECTION_MIN_YAW_DEG:
                return
        self._last_detection_xy = (float(self.robot_position_map[0]), float(self.robot_position_map[1]))
        self._last_detection_yaw_deg = self.robot_yaw_deg
        self._run_detection(msg)

    def _run_detection(self, image_msg: CompressedImage):
        if self.query_spec is None:
            return
        if not self._has_odometry:
            return
        vocabulary = self._build_vocabulary()
        needed_categories = self._needed_categories()
        pil_image = image_msg_to_pil(image_msg)
        detections = self.detector.detect_360(pil_image, vocabulary)
        for det in detections:
            label_lower = det.label.lower()
            for synonym, canonical in config.CATEGORY_SYNONYMS.items():
                if synonym in label_lower:
                    det.label = label_lower.replace(synonym, canonical)
                    break
            label_lower = det.label.lower()

            det.label = re.sub(r'^\s*((?:a|an|the)\s+)+', r'\1', det.label, flags=re.IGNORECASE)
            label_lower = det.label.lower()

            matched_needed = {cat for cat in needed_categories if re.search(rf"\b{re.escape(cat)}\b", label_lower)}
            has_repeated_category = any(
                len(re.findall(rf"\b{re.escape(cat)}\b", label_lower)) > 1 for cat in needed_categories
            )
            if len(matched_needed) > 1 or has_repeated_category:
                self.get_logger().warn(
                    f"Rejected concatenated-label detection {det.label!r} "
                    f"(matched categories: {matched_needed}, repeated_category={has_repeated_category})"
                )
                continue

            is_distractor = any(
                distractor.split(" ", 1)[1] in det.label.lower() for distractor in config.VOCABULARY_DISTRACTOR_CATEGORIES
            )
            is_needed = any(cat in det.label.lower() for cat in needed_categories)
            if is_distractor and not is_needed:
                continue
            localized = localize_detection_3d(
                det.box_xyxy, self.latest_registered_scan, self.robot_position_map
            )
            if localized is None:
                continue
            position_3d, size_3d = localized
            self.scene_graph.add_or_merge(det.label, det.color, position_3d, det.score, size=size_3d)

    def _needed_categories(self) -> set:
        def _strip_article(s: str) -> str:
            s = s.strip().lower()
            for prefix in ("a ", "an ", "the "):
                if s.startswith(prefix):
                    return s[len(prefix):]
            return s

        needed = set()
        if self.query_spec.target_category:
            needed.add(self.query_spec.target_category.lower())
        for rel in self.query_spec.relations:
            for anchor in rel.anchors:
                needed.add(_strip_article(anchor.category))
                if anchor.qualifier_category:
                    needed.add(_strip_article(anchor.qualifier_category))
        for pc in self.query_spec.path_constraints:
            for anchor in pc.anchors:
                needed.add(_strip_article(anchor.category))
                if anchor.qualifier_category:
                    needed.add(_strip_article(anchor.qualifier_category))
        return needed

    def _build_vocabulary(self) -> list:
        vocab = set()
        if self.query_spec.target_category:
            vocab.add(f"a {self.query_spec.target_category}")
        for rel in self.query_spec.relations:
            for anchor in rel.anchors:
                vocab.add(anchor.category if anchor.category.startswith(("a ", "an", "the")) else f"a {anchor.category}")
                if anchor.qualifier_category:
                    vocab.add(anchor.qualifier_category if anchor.qualifier_category.startswith(("a ", "an", "the")) else f"a {anchor.qualifier_category}")
        for pc in self.query_spec.path_constraints:
            for anchor in pc.anchors:
                vocab.add(anchor.category if anchor.category.startswith(("a ", "an", "the")) else f"a {anchor.category}")
                if anchor.qualifier_category:
                    vocab.add(anchor.qualifier_category if anchor.qualifier_category.startswith(("a ", "an", "the")) else f"a {anchor.qualifier_category}")
        vocab.update(config.VOCABULARY_DISTRACTOR_CATEGORIES)
        for cat in self._needed_categories():
            for base_cat, variants in config.CATEGORY_PROMPT_VARIANTS.items():
                if base_cat in cat:
                    vocab.update(variants)
        return sorted(vocab) or ["an object"]

    def _superlative_constraints_ready(self) -> bool:
        for constraint in self.query_spec.path_constraints:
            for anchor in constraint.anchors:
                if anchor.qualifier_type in SUPERLATIVE_QUALIFIER_TYPES:
                    target_cat = anchor.category.lower()
                    count = sum(1 for n in self.scene_graph.nodes if target_cat in n.label.lower())
                    if count < 2:
                        return False
        return True

    def on_main_loop(self):
        if self.state == State.WAIT_FOR_QUESTION:
            return

        elapsed = time.time() - self.question_received_at

        if self.state == State.EXPLORE:
            if elapsed > config.EXPLORATION_TIME_BUDGET_SEC:
                self.get_logger().info("Exploration budget spent, moving to ANSWER")
                self.state = State.ANSWER
                return
            if (self.query_spec is not None
                    and self.query_spec.question_type == "instruction_following"
                    and self.explorer.visited_cells):
                candidate_waypoints = build_waypoint_sequence(
                    self.query_spec.path_constraints,
                    self.scene_graph,
                    (self.robot_position_map[0], self.robot_position_map[1]),
                )
                expected_count = len(
                    [c for c in self.query_spec.path_constraints if c.type != "avoid"]
                )
                if (candidate_waypoints
                        and len(candidate_waypoints) == expected_count
                        and self._superlative_constraints_ready()):
                    self.get_logger().info(
                        f"All {expected_count} instruction-following anchor(s) resolved "
                        f"early (superlative constraints have enough candidates) -- "
                        f"skipping remaining exploration budget, moving to ANSWER"
                    )
                    self.state = State.ANSWER
                    return
            self._explore_step()
            return

        if self.state == State.ANSWER:
            self._answer()
            self.state = State.DONE
            return

    def _explore_step(self):
        if not self.explorer.visited_cells:
            return

        now = time.time()
        current_xy = self.robot_position_map[:2].copy()
        if self._stuck_check_time is None:
            self._stuck_check_time = now
            self._stuck_check_position = current_xy
        elif now - self._stuck_check_time > config.STUCK_CHECK_INTERVAL_SEC:
            moved = float(np.hypot(current_xy[0] - self._stuck_check_position[0],
                                    current_xy[1] - self._stuck_check_position[1]))
            if moved < config.STUCK_MOVE_THRESHOLD_M and self._current_target is not None:
                self.get_logger().warn(
                    f"Stuck: moved only {moved:.2f}m in {config.STUCK_CHECK_INTERVAL_SEC}s "
                    f"toward target {self._current_target}, blacklisting it and picking a new frontier"
                )
                self.explorer.blacklist_current_target(self._current_target)
                self._current_target = None
            self._stuck_check_time = now
            self._stuck_check_position = current_xy

        target = self.explorer.next_waypoint((self.robot_position_map[0], self.robot_position_map[1]))
        if target is None:
            self.get_logger().info("No frontiers left within radius, moving to ANSWER early")
            self.state = State.ANSWER
            return
        self._current_target = target
        self._publish_waypoint(target[0], target[1])

    def _publish_waypoint(self, x: float, y: float, theta: float = 0.0):
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), float(theta)
        self.waypoint_pub.publish(msg)

    def _publish_planned_path_marker(self, waypoints):
        from geometry_msgs.msg import Point
        marker = Marker()
        marker.header = self._map_header()
        marker.ns = "planned_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 1.0, 0.2, 0.9
        marker.pose.orientation.w = 1.0

        start = Point()
        start.x, start.y, start.z = float(self.robot_position_map[0]), float(self.robot_position_map[1]), 0.2
        marker.points.append(start)
        for x, y in waypoints:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), 0.2
            marker.points.append(p)

        self.planned_path_pub.publish(marker)
        self.get_logger().info(f"Published planned path marker with {len(marker.points)} points on {config.TOPIC_PLANNED_PATH_MARKER}")

    def _answer(self):
        spec = self.query_spec
        self.scene_graph.consolidate()
        self.get_logger().info(f"Scene graph at answer time:\n{self.scene_graph.summary()}")

        def _to_scene_graph_relations(relations):
            return [
                (rel.type, [
                    AnchorSpec(category=a.category, qualifier_type=a.qualifier_type, qualifier_category=a.qualifier_category)
                    for a in rel.anchors
                ])
                for rel in relations
            ]

        if spec.question_type == "numerical":
            relations = _to_scene_graph_relations(spec.relations)
            count = self.scene_graph.count(spec.target_category, spec.target_attributes, relations)
            self.numerical_pub.publish(Int32(data=count))
            self.get_logger().info(f"Published numerical answer: {count}")

        elif spec.question_type == "object_reference":
            relations = _to_scene_graph_relations(spec.relations)
            node = self.scene_graph.find_unique_referent(spec.target_category, spec.target_attributes, relations)
            if node is None:
                self.get_logger().warn("No matching object found in scene graph -- publishing best-effort empty marker")
                return
            header = self._map_header()
            marker = build_marker_message(Marker, header, node.label, node.position, node.size)
            self.marker_pub.publish(marker)
            self.get_logger().info(f"Published object marker for {node.label} at {node.position}")

        elif spec.question_type == "instruction_following":
            waypoints = build_waypoint_sequence(
                spec.path_constraints, self.scene_graph, (self.robot_position_map[0], self.robot_position_map[1])
            )
            if not waypoints:
                self.get_logger().warn("Could not resolve any path waypoints from scene graph")
                return
            self.get_logger().info(f"Computed {len(waypoints)} waypoint(s): {waypoints}")
            self._publish_planned_path_marker(waypoints)
            for x, y in waypoints:
                self.get_logger().info(f"Publishing waypoint ({x:.2f}, {y:.2f})")
                self._publish_waypoint(x, y)
                self._wait_until_reached(x, y)

    def _wait_until_reached(self, x: float, y: float, timeout_s: float = 60.0):
        # CREEPING-WAYPOINT FIX 2026-08-20: root-caused via the vendor
        # local_planner/pathFollower SOURCE (autonomy_stack_mecanum_wheel_
        # platform git submodule). localPlanner.cpp scores candidate
        # directions against obstacle geometry and only publishes a fresh
        # /path when some direction scores above 0 (selectedGroupID >= 0);
        # if every direction is obstacle-blocked, selectedGroupID stays -1
        # and no new path publishes that cycle -- no fallback. pathFollower
        # .cpp then runs the stale path down to pathSize <= 1 and hard-
        # zeroes velocity (joySpeed2 = 0), with nothing to re-trigger a
        # fresh scoring pass since the robot itself has stopped moving.
        # Confirmed live 3x via /cmd_vel: clean non-zero motion, then a
        # clean full stop that never resumes on its own.
        #
        # Republishing the SAME distant point does not help (tried first,
        # confirmed live: 0.00m moved across 3 attempts in one run) since
        # it doesn't change the obstacle geometry causing every direction
        # to score 0. Instead: when stuck, publish a CLOSER intermediate
        # point along the line from the robot's current position toward
        # the real goal. A nearer target changes the scoring geometry and
        # is far more likely to have at least one clear direction. Each
        # creep step is checked against WAYPOINT_REACHED_TOL_M by the loop
        # below, so successive stuck checks keep creeping toward the real
        # (x, y). Bounded by the same overall timeout_s.
        REPUBLISH_STUCK_CHECK_SEC = 10.0
        WAYPOINT_CREEP_FACTOR = 0.5
        MIN_CREEP_DIST_M = 0.3
        start = time.time()
        last_check_time = start
        last_check_position = self.robot_position_map[:2].copy()
        creep_attempts = 0
        while time.time() - start < timeout_s:
            dist_to_real_goal = np.hypot(self.robot_position_map[0] - x, self.robot_position_map[1] - y)
            if dist_to_real_goal < config.WAYPOINT_REACHED_TOL_M:
                self.get_logger().info(
                    f"Reached waypoint ({x:.2f}, {y:.2f}) -- dist={dist_to_real_goal:.2f}m, "
                    f"elapsed={time.time()-start:.1f}s, creep_attempts={creep_attempts}"
                )
                return
            now = time.time()
            if now - last_check_time > REPUBLISH_STUCK_CHECK_SEC:
                current_xy = self.robot_position_map[:2].copy()
                moved = float(np.hypot(current_xy[0] - last_check_position[0],
                                        current_xy[1] - last_check_position[1]))
                if moved < config.STUCK_MOVE_THRESHOLD_M:
                    robot_xy = self.robot_position_map[:2]
                    goal_xy = np.array([x, y], dtype=np.float32)
                    remaining = goal_xy - robot_xy
                    remaining_dist = float(np.linalg.norm(remaining))
                    creep_dist = max(MIN_CREEP_DIST_M, remaining_dist * WAYPOINT_CREEP_FACTOR)
                    creep_goal = (x, y)
                    if remaining_dist > 1e-3:
                        direction = remaining / remaining_dist
                        creep_xy = robot_xy + direction * min(creep_dist, remaining_dist)
                        creep_goal = (float(creep_xy[0]), float(creep_xy[1]))
                    creep_attempts += 1
                    self.get_logger().warn(
                        f"Waypoint ({x:.2f}, {y:.2f}): moved only {moved:.2f}m in "
                        f"{REPUBLISH_STUCK_CHECK_SEC:.0f}s (still {dist_to_real_goal:.2f}m from goal) -- "
                        f"likely no clear direction at the current target; creeping to intermediate "
                        f"point ({creep_goal[0]:.2f}, {creep_goal[1]:.2f}) instead of retrying the same "
                        f"distant point [attempt {creep_attempts}]"
                    )
                    self._publish_waypoint(creep_goal[0], creep_goal[1])
                last_check_time = now
                last_check_position = current_xy
            time.sleep(0.2)
        self.get_logger().warn(
            f"TIMED OUT waiting to reach waypoint ({x:.2f}, {y:.2f}) after {timeout_s:.0f}s "
            f"(creep_attempts={creep_attempts}) -- "
            f"robot ended at ({self.robot_position_map[0]:.2f}, {self.robot_position_map[1]:.2f}), "
            f"still {np.hypot(self.robot_position_map[0]-x, self.robot_position_map[1]-y):.2f}m away "
            f"(tolerance is {config.WAYPOINT_REACHED_TOL_M}m) -- moving on to the next waypoint anyway"
        )

    def _map_header(self):
        from std_msgs.msg import Header
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()
        return header


def main():
    rclpy.init()
    node = VLNChallengeNode()
    import os
    num_threads = max(2, min(os.cpu_count() or 4, config.MAX_EXECUTOR_THREADS))
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
