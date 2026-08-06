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
import time
from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, PointCloud2
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

import config
from exploration import FrontierExplorer
from path_planner import build_waypoint_sequence
from perception import OpenVocabDetector
from question_parser import QuerySpec, parse_question
from ros_utils import build_marker_message, localize_detection_3d, pointcloud2_to_xyz_array
from scene_graph import SceneGraph


class State(Enum):
    WAIT_FOR_QUESTION = auto()
    EXPLORE = auto()
    ANSWER = auto()
    DONE = auto()


def image_msg_to_pil(msg: CompressedImage) -> PILImage.Image:
    """Decode /camera/image/compressed (sensor_msgs/CompressedImage, format='png')
    directly with PIL. See the NOTE in config.py for why we read the
    compressed topic instead of the uncompressed /camera/image the README
    documents -- the repub node bridging the two is not working in the
    vendor-shipped sim image as of 2026-08-06."""
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
        self.latest_registered_scan = np.empty((0, 3), dtype=np.float32)
        self._image_frame_count = 0
        self._pending_waypoint_queue = []
        self._waypoint_in_flight = None

        self.create_subscription(String, config.TOPIC_QUESTION, self.on_question, 1)
        self.create_subscription(CompressedImage, config.TOPIC_IMAGE, self.on_image, 1)
        self.create_subscription(PointCloud2, config.TOPIC_REGISTERED_SCAN, self.on_registered_scan, 1)
        self.create_subscription(PointCloud2, config.TOPIC_TERRAIN_MAP_EXT, self.on_terrain_map, 1)
        self.create_subscription(Odometry, config.TOPIC_STATE_ESTIMATION, self.on_odometry, 5)

        self.waypoint_pub = self.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 1)
        self.marker_pub = self.create_publisher(Marker, config.TOPIC_OBJECT_MARKER, 1)
        self.numerical_pub = self.create_publisher(Int32, config.TOPIC_NUMERICAL_RESPONSE, 1)

        self.create_timer(1.0 / config.MAIN_LOOP_HZ, self.on_main_loop)

        self.get_logger().info("VLN challenge ai_module ready, waiting for question...")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def on_question(self, msg: String):
        if self.state != State.WAIT_FOR_QUESTION:
            return  # one question per node lifetime per the README
        self.get_logger().info(f"Question received: {msg.data}")
        self.query_spec = parse_question(msg.data)
        self.question_received_at = time.time()
        self.state = State.EXPLORE

    def on_odometry(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_position_map = np.array([p.x, p.y, p.z], dtype=np.float32)

    def on_registered_scan(self, msg: PointCloud2):
        self.latest_registered_scan = pointcloud2_to_xyz_array(msg)

    def on_terrain_map(self, msg: PointCloud2):
        xyz = pointcloud2_to_xyz_array(msg)
        self.explorer.update_known_cells(xyz)

    def on_image(self, msg: CompressedImage):
        if self.state not in (State.EXPLORE,):
            return
        self._image_frame_count += 1
        if self._image_frame_count % config.FRAME_SAMPLE_EVERY_N != 0:
            return
        self._run_detection(msg)

    # ------------------------------------------------------------------
    # Perception -> scene graph
    # ------------------------------------------------------------------
    def _run_detection(self, image_msg: CompressedImage):
        if self.query_spec is None:
            return
        vocabulary = self._build_vocabulary()
        pil_image = image_msg_to_pil(image_msg)
        detections = self.detector.detect(pil_image, vocabulary)
        for det in detections:
            position_3d = localize_detection_3d(
                det.box_xyxy, self.latest_registered_scan, self.robot_position_map
            )
            if position_3d is None:
                continue
            self.scene_graph.add_or_merge(det.label, det.color, position_3d, det.score)

    def _build_vocabulary(self) -> list:
        """Open-vocab prompts to search for: the target category plus every
        anchor category referenced in relations/path constraints, since
        those need to be in the scene graph too for relation queries to work."""
        vocab = set()
        if self.query_spec.target_category:
            vocab.add(f"a {self.query_spec.target_category}")
        for rel in self.query_spec.relations:
            for anchor in rel.anchors:
                vocab.add(anchor if anchor.startswith(("a ", "an", "the")) else f"a {anchor}")
        for pc in self.query_spec.path_constraints:
            for anchor in pc.anchors:
                vocab.add(anchor if anchor.startswith(("a ", "an", "the")) else f"a {anchor}")
        return sorted(vocab) or ["an object"]

    # ------------------------------------------------------------------
    # Main state machine
    # ------------------------------------------------------------------
    def on_main_loop(self):
        if self.state == State.WAIT_FOR_QUESTION:
            return

        elapsed = time.time() - self.question_received_at

        if self.state == State.EXPLORE:
            if elapsed > config.EXPLORATION_TIME_BUDGET_SEC:
                self.get_logger().info("Exploration budget spent, moving to ANSWER")
                self.state = State.ANSWER
                return
            self._explore_step()
            return

        if self.state == State.ANSWER:
            self._answer()
            self.state = State.DONE
            return

    def _explore_step(self):
        target = self.explorer.next_waypoint((self.robot_position_map[0], self.robot_position_map[1]))
        if target is None:
            self.get_logger().info("No frontiers left within radius, moving to ANSWER early")
            self.state = State.ANSWER
            return
        self._publish_waypoint(target[0], target[1])

    def _publish_waypoint(self, x: float, y: float, theta: float = 0.0):
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), float(theta)
        self.waypoint_pub.publish(msg)

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------
    def _answer(self):
        spec = self.query_spec
        self.get_logger().info(f"Scene graph at answer time:\n{self.scene_graph.summary()}")

        if spec.question_type == "numerical":
            near_anchor = spec.relations[0].anchors[0] if spec.relations and spec.relations[0].anchors else None
            count = self.scene_graph.count(spec.target_category, spec.target_attributes, near_anchor)
            self.numerical_pub.publish(Int32(data=count))
            self.get_logger().info(f"Published numerical answer: {count}")

        elif spec.question_type == "object_reference":
            anchor_categories = [a for rel in spec.relations for a in rel.anchors]
            node = self.scene_graph.find_unique_referent(spec.target_category, spec.target_attributes, anchor_categories)
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
            for x, y in waypoints:
                self._publish_waypoint(x, y)
                self._wait_until_reached(x, y)

    def _wait_until_reached(self, x: float, y: float, timeout_s: float = 60.0):
        start = time.time()
        while time.time() - start < timeout_s:
            dist = np.hypot(self.robot_position_map[0] - x, self.robot_position_map[1] - y)
            if dist < config.WAYPOINT_REACHED_TOL_M:
                return
            rclpy.spin_once(self, timeout_sec=0.2)

    def _map_header(self):
        from std_msgs.msg import Header
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()
        return header


def main():
    rclpy.init()
    node = VLNChallengeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
