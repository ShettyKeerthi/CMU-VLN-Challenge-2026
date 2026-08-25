#!/usr/bin/env python3
"""
Top-level ROS2 node for the CMU-VLN-Challenge ai_module.

  WAIT_FOR_QUESTION -> EXPLORE -> RETURN_HOME -> ANSWER -> EXECUTE -> DONE

Competition mode only. The scene graph is built entirely from perception:
open-vocabulary detections on /camera/image, lifted to 3D against
/registered_scan. No ground-truth semantics are read anywhere -- the challenge
provides none in either phase, so any code path depending on object_list.txt
would score zero at test time.

Inputs are limited to the topics the README lists as available at test time;
outputs to the topics it lists as accepted by the base system.

WHY RETURN_HOME EXISTS.
Instruction-following is scored on the trajectory the robot drives, and the
statement describes a path starting from where the robot began. If the answer
path is planned from wherever exploration happened to end, the first leg is
computed from the wrong origin: approach points land on the wrong side of
objects, and the driven path bears little resemblance to the described one.
"""

import argparse
import math
import os
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

import config
from exploration import FrontierExplorer
from path_planner import build_plan, missing_categories
from perception import OpenVocabDetector, save_detections_debug
from question_parser import QuerySpec, parse_question
from ros_utils import (
    build_marker_message,
    localize_detection_3d_with_extent,
    pointcloud2_to_xyz_array,
    pointcloud2_to_xyzi_array,
)
from scene_graph import SceneGraph

# Arrival band. The base autonomy stack keeps its own obstacle inflation and
# refuses to drive inside it, so a tight tolerance guarantees a timeout on
# every leg. Treat a plateau as arrival instead.
ARRIVE_TOL_M = 1.0
HOME_TOL_M = 0.6                # no obstacle at the start pose, so tighter
STALL_SECONDS = 10.0            # no progress for this long -> the planner is done
STALL_EPS_M = 0.05              # progress smaller than this does not count
WAYPOINT_TIMEOUT_S = 45.0
RETURN_HOME_TIMEOUT_S = 90.0


class State(Enum):
    WAIT_FOR_QUESTION = auto()
    EXPLORE = auto()
    RETURN_HOME = auto()
    ANSWER = auto()
    EXECUTE = auto()
    DONE = auto()


def image_msg_to_pil(msg: Image) -> PILImage.Image:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return PILImage.fromarray(arr, mode="RGB")


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class VLNChallengeNode(Node):
    def __init__(self, return_home: bool = True):
        super().__init__("vln_challenge_ai_module")

        self.return_home_enabled = return_home
        self.scene_graph = SceneGraph()
        self.explorer = FrontierExplorer()
        self.detector = OpenVocabDetector()

        self.state = State.WAIT_FOR_QUESTION
        self.query_spec: Optional[QuerySpec] = None
        self.question_received_at = None

        self.robot_position_map = np.zeros(3, dtype=np.float32)
        self.robot_orientation = None
        self.sensor_yaw = 0.0
        self.have_odom = False
        self.start_pose: Optional[Tuple[float, float, float]] = None   # x, y, yaw
        self.latest_registered_scan = np.empty((0, 3), dtype=np.float32)
        self.have_terrain = False
        self._image_frame_count = 0

        # Waypoint execution, driven by the main timer. A blocking spin_once
        # loop inside a timer callback is re-entrant and drops callbacks.
        self._plan_legs: List = []
        self._waypoints: List[Tuple[float, float, float]] = []
        self._wp_index = 0
        self._wp_started = 0.0
        self._best_dist = float("inf")
        self._stall_since = 0.0
        self._home_started = 0.0

        self.debug_dir = "/home/docker/debug_detections"
        os.makedirs(self.debug_dir, exist_ok=True)

        self.create_subscription(String, config.TOPIC_QUESTION, self.on_question, 1)
        self.create_subscription(Odometry, config.TOPIC_STATE_ESTIMATION, self.on_odometry, 5)
        self.create_subscription(PointCloud2, config.TOPIC_TERRAIN_MAP_EXT, self.on_terrain_map, 1)
        self.create_subscription(Image, config.TOPIC_IMAGE, self.on_image, 1)
        self.create_subscription(PointCloud2, config.TOPIC_REGISTERED_SCAN,
                                 self.on_registered_scan, 1)

        self.waypoint_pub = self.create_publisher(Pose2D, config.TOPIC_WAYPOINT, 1)
        self.marker_pub = self.create_publisher(Marker, config.TOPIC_OBJECT_MARKER, 5)
        self.numerical_pub = self.create_publisher(Int32, config.TOPIC_NUMERICAL_RESPONSE, 1)

        # Debug visualisations, published by us and therefore ours to clear.
        self.graph_pub = self.create_publisher(MarkerArray, "/scene_graph_markers", 1)
        self.trail_pub = self.create_publisher(MarkerArray, "/exploration_trail", 1)
        self.path_pub = self.create_publisher(MarkerArray, "/planned_path", 1)

        self._trail: List[Tuple[float, float]] = []
        self._publish_graph = True

        self.create_timer(1.0 / config.MAIN_LOOP_HZ, self.on_main_loop)
        self.create_timer(2.0, self.publish_graph_markers)

        self.get_logger().info("ai_module ready, waiting for question...")

    # ------------------------------------------------------------- callbacks

    def on_question(self, msg: String):
        if self.state != State.WAIT_FOR_QUESTION:
            return                            # one question per node lifetime
        if not self.have_odom:
            self.get_logger().warn("question arrived before odometry, ignoring")
            return

        self.get_logger().info(f"Question received: {msg.data}")
        self.query_spec = parse_question(msg.data)
        self.question_received_at = time.time()

        # Capture the start pose now, before anything moves. The instruction
        # path is described relative to here.
        self.start_pose = (float(self.robot_position_map[0]),
                           float(self.robot_position_map[1]),
                           float(self.sensor_yaw))
        self.get_logger().info(
            f"start pose ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}) "
            f"yaw {math.degrees(self.start_pose[2]):.0f} deg")

        self.state = State.EXPLORE

    def on_odometry(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_position_map = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.robot_orientation = msg.pose.pose.orientation
        self.sensor_yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.have_odom = True

        if self.state == State.EXPLORE:
            if not self._trail or math.hypot(p.x - self._trail[-1][0],
                                             p.y - self._trail[-1][1]) > 0.3:
                self._trail.append((p.x, p.y))

    def on_registered_scan(self, msg: PointCloud2):
        self.latest_registered_scan = pointcloud2_to_xyz_array(msg)

    def on_terrain_map(self, msg: PointCloud2):
        # Intensity is height above the estimated ground plane; without it
        # every cell looks drivable and free_check is useless.
        self.explorer.update_known_cells(pointcloud2_to_xyzi_array(msg))
        self.have_terrain = True

    def on_image(self, msg: Image):
        if self.state != State.EXPLORE:
            return
        self._image_frame_count += 1
        if self._image_frame_count % config.FRAME_SAMPLE_EVERY_N != 0:
            return
        self._run_detection(msg)

    # ------------------------------------------------- perception -> graph

    def _run_detection(self, image_msg: Image):
        if self.query_spec is None or not self.have_odom:
            return
        if len(self.latest_registered_scan) == 0:
            return

        # Snapshot pose and scan BEFORE inference. Detection takes seconds on a
        # slow GPU; the pose afterwards is wherever the robot has driven to.
        scan = self.latest_registered_scan
        pos = self.robot_position_map.copy()
        yaw = self.sensor_yaw

        vocabulary = self._build_vocabulary()
        pil = image_msg_to_pil(image_msg)
        detections = self.detector.detect(pil, vocabulary)
        save_detections_debug(pil, detections, self.debug_dir,
                              tag=self.query_spec.question_type, vocabulary=vocabulary)

        for det in detections:
            result = localize_detection_3d_with_extent(det.box_xyxy, scan, pos, yaw)
            if result is None:
                continue
            position_3d, size_3d = result
            self.scene_graph.add_or_merge(det.label, det.color, position_3d,
                                          det.score, size=size_3d)

    def _build_vocabulary(self) -> list:
        """Every category the question depends on, including anchors nested
        inside each leg's target -- miss those and the relation silently
        scores zero on every candidate."""
        cats = self.query_spec.required_categories() if self.query_spec else []
        vocab = {c if c.startswith(("a ", "an ", "the ")) else f"a {c}" for c in cats}
        return sorted(vocab) or ["an object"]

    # ------------------------------------------------------- state machine

    def on_main_loop(self):
        if self.state in (State.WAIT_FOR_QUESTION, State.DONE):
            return

        elapsed = time.time() - self.question_received_at

        if self.state == State.EXPLORE:
            if elapsed > config.EXPLORATION_TIME_BUDGET_SEC:
                self.get_logger().info("Exploration budget spent")
                self._finish_exploration()
                return
            self._explore_step()
            return

        if self.state == State.RETURN_HOME:
            self._return_home_step()
            return

        if self.state == State.ANSWER:
            if not self.have_odom:
                return
            self._answer()
            self.state = State.EXECUTE if self._waypoints else State.DONE
            return

        if self.state == State.EXECUTE:
            self._execute_step(elapsed)

    def _explore_step(self):
        target = self.explorer.next_waypoint(
            (self.robot_position_map[0], self.robot_position_map[1]))
        if target is None:
            self.get_logger().info("no frontiers left")
            self._finish_exploration()
            return
        self._publish_waypoint(target[0], target[1])

    def _finish_exploration(self):
        """Exploration over: stop detecting, clear debug visuals, head home."""
        self.get_logger().info(
            f"exploration complete: {len(self.scene_graph.nodes)} objects, "
            f"{self.explorer.stats()}")
        self._clear_markers()

        if not self.return_home_enabled or self.start_pose is None:
            self.state = State.ANSWER
            return

        hx, hy, hyaw = self.start_pose
        d = math.hypot(self.robot_position_map[0] - hx, self.robot_position_map[1] - hy)
        if d < HOME_TOL_M:
            self.get_logger().info("already at start pose")
            self.state = State.ANSWER
            return

        self.get_logger().info(f"returning to start pose, {d:.2f} m away")
        self._publish_waypoint(hx, hy, hyaw)
        self._home_started = time.time()
        self._best_dist = d
        self._stall_since = time.time()
        self.state = State.RETURN_HOME

    def _return_home_step(self):
        hx, hy, hyaw = self.start_pose
        d = math.hypot(self.robot_position_map[0] - hx, self.robot_position_map[1] - hy)
        now = time.time()

        if d < self._best_dist - STALL_EPS_M:
            self._best_dist, self._stall_since = d, now

        stalled = (now - self._stall_since) > STALL_SECONDS
        timed_out = (now - self._home_started) > RETURN_HOME_TIMEOUT_S

        if d < HOME_TOL_M:
            self.get_logger().info(f"back at start pose ({d:.2f} m)")
        elif stalled:
            self.get_logger().warn(f"return home plateaued at {d:.2f} m, continuing")
        elif timed_out:
            self.get_logger().warn(f"return home timed out at {d:.2f} m, continuing")
        else:
            # Keep republishing: the base system drops goals it has completed
            # or superseded, and a single publish can be lost.
            self._publish_waypoint(hx, hy, hyaw)
            return

        self.state = State.ANSWER

    def _publish_waypoint(self, x: float, y: float, theta: float = 0.0):
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(x), float(y), float(theta)
        self.waypoint_pub.publish(msg)

    # ------------------------------------------------------- marker clearing

    def _clear_markers(self):
        """Remove our own debug markers before answering, so exploration
        clutter does not obscure the answer.

        Only markers WE published can be cleared this way. The green waypoint
        sphere and path line come from the base autonomy stack's own
        visualisation nodes -- untick those displays in RViz if they get in
        the way.
        """
        self._publish_graph = False

        arr = MarkerArray()
        m = Marker()
        m.header = self._map_header()
        m.action = Marker.DELETEALL
        arr.markers.append(m)
        self.graph_pub.publish(arr)
        self.trail_pub.publish(arr)

        solo = Marker()
        solo.header = self._map_header()
        solo.action = Marker.DELETEALL
        self.marker_pub.publish(solo)

        self._trail.clear()
        self.get_logger().info("cleared debug markers")

    # ---------------------------------------------------------- answering

    def _free_check(self):
        """Drivability test from the terrain grid, or None if no terrain data
        has arrived (planner then falls back to raw standoffs)."""
        if not self.have_terrain or not self.explorer.free_cells:
            return None
        return lambda x, y: self.explorer._world_to_cell(x, y) in self.explorer.free_cells

    def _answer(self):
        spec = self.query_spec
        self.get_logger().info(
            f"Scene graph ({len(self.scene_graph.nodes)} nodes):\n{self.scene_graph.summary()}")

        if spec.question_type == "numerical":
            near_anchor = (spec.relations[0].anchors[0]
                           if spec.relations and spec.relations[0].anchors else None)
            count = self.scene_graph.count(spec.target_category,
                                           spec.target_attributes, near_anchor)
            # Never publish 0: these questions ask about something that is in
            # the scene, so a zero is always wrong, whereas a 1 is sometimes right.
            if count == 0:
                self.get_logger().warn("count is 0, publishing 1 instead")
                count = 1
            self.numerical_pub.publish(Int32(data=int(count)))
            self.get_logger().info(f"Published numerical answer: {count}")

        elif spec.question_type == "object_reference":
            node = self.scene_graph.find_unique_referent(
                spec.target_category, spec.target_attributes, relations=spec.relations)
            if node is None:
                self.get_logger().warn(f"no match for {spec.target_category!r}")
                return
            self.marker_pub.publish(build_marker_message(
                Marker, self._map_header(), node.label, node.position, node.size))
            self.get_logger().info(
                f"Published marker: {node.label} at "
                f"({node.position[0]:.2f}, {node.position[1]:.2f}, {node.position[2]:.2f}) "
                f"size ({node.size[0]:.2f}, {node.size[1]:.2f}, {node.size[2]:.2f})")

        elif spec.question_type == "instruction_following":
            missing = missing_categories(spec.path_constraints, self.scene_graph)
            if missing:
                self.get_logger().warn(f"not in scene graph: {missing}")

            # Plan from the START pose, not from wherever we are now -- the
            # statement describes a path beginning there.
            origin = ((self.start_pose[0], self.start_pose[1]) if self.start_pose
                      else (self.robot_position_map[0], self.robot_position_map[1]))

            self._plan_legs = build_plan(spec.path_constraints, self.scene_graph,
                                         origin, free_check=self._free_check())
            self._waypoints = [wp for leg in self._plan_legs for wp in leg.waypoints]

            self._print_full_path(origin)
            self._publish_path_markers()

            if not self._waypoints:
                self.get_logger().error("no waypoints resolved -- nowhere to drive")
                return

            self._wp_index = 0
            self._start_waypoint()

    def _print_full_path(self, origin):
        """Print the complete waypoint sequence in one block before driving.

        Waypoints are still PUBLISHED one at a time -- sending them all at once
        means the base system only acts on the last, and reaching path
        constraints out of order is penalised. This is for the log only.
        """
        lines = []
        lines.append("=" * 66)
        lines.append(f"PLANNED PATH  ({len(self._waypoints)} waypoints, "
                     f"{len(self._plan_legs)} legs)")
        lines.append(f"question: {self.query_spec.raw_question}")
        lines.append("-" * 66)
        lines.append(f"  start  ({origin[0]:7.2f}, {origin[1]:7.2f})")

        n = 0
        prev = (origin[0], origin[1])
        for leg in self._plan_legs:
            state = "ok  " if leg.resolved else "MISS"
            mark = "  <- FINAL" if leg.is_final else ""
            lines.append(f"  [{state}] leg {leg.index + 1}: {leg.description}{mark}")
            if not leg.waypoints:
                lines.append("           (no waypoints from this leg)")
                continue
            for (x, y, th) in leg.waypoints:
                n += 1
                step = math.hypot(x - prev[0], y - prev[1])
                flag = "  <-- too close to previous" if step < ARRIVE_TOL_M else ""
                lines.append(f"     {n:2d}. ({x:7.2f}, {y:7.2f})  "
                             f"theta {math.degrees(th):6.1f} deg  "
                             f"step {step:5.2f} m{flag}")
                prev = (x, y)

        total = 0.0
        p = (origin[0], origin[1])
        for (x, y, _) in self._waypoints:
            total += math.hypot(x - p[0], y - p[1])
            p = (x, y)
        lines.append("-" * 66)
        lines.append(f"  total path length {total:.2f} m")
        lines.append("=" * 66)

        self.get_logger().info("\n" + "\n".join(lines))

        # Copy-pasteable, for replaying a single waypoint from the terminal.
        for i, (x, y, th) in enumerate(self._waypoints, 1):
            print(f"# waypoint {i}\n"
                  f"ros2 topic pub -r 1 /way_point_with_heading "
                  f"geometry_msgs/msg/Pose2D "
                  f"'{{x: {x:.2f}, y: {y:.2f}, theta: {th:.2f}}}'")

    def _publish_path_markers(self):
        """Draw the whole planned path in RViz: a line strip plus numbered
        spheres, so the trajectory can be checked before the robot commits."""
        if not self._waypoints:
            return
        arr = MarkerArray()
        header = self._map_header()

        line = Marker()
        line.header = header
        line.ns = "planned_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.08
        line.pose.orientation.w = 1.0
        line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.6, 0.0, 0.9
        from geometry_msgs.msg import Point
        if self.start_pose:
            line.points.append(Point(x=float(self.start_pose[0]),
                                     y=float(self.start_pose[1]), z=0.1))
        for (x, y, _) in self._waypoints:
            line.points.append(Point(x=float(x), y=float(y), z=0.1))
        arr.markers.append(line)

        for i, (x, y, _) in enumerate(self._waypoints, 1):
            s = Marker()
            s.header = header
            s.ns = "planned_path"
            s.id = i
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x, s.pose.position.y, s.pose.position.z = float(x), float(y), 0.1
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.25
            s.color.r, s.color.g, s.color.b, s.color.a = 1.0, 0.9, 0.1, 0.9
            arr.markers.append(s)

            t = Marker()
            t.header = header
            t.ns = "planned_path_labels"
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = float(x), float(y), 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.3
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = str(i)
            arr.markers.append(t)

        self.path_pub.publish(arr)

    # ---------------------------------------------------------- execution

    def _start_waypoint(self):
        x, y, th = self._waypoints[self._wp_index]
        self._publish_waypoint(x, y, th)
        self._wp_started = time.time()
        self._stall_since = time.time()
        self._best_dist = math.hypot(self.robot_position_map[0] - x,
                                     self.robot_position_map[1] - y)
        self.get_logger().info(
            f"{self._wp_index + 1}/{len(self._waypoints)} -> ({x:.2f}, {y:.2f}) "
            f"dist {self._best_dist:.2f} m")

    def _execute_step(self, elapsed: float):
        """Advance the waypoint list one at a time.

        Sequential and non-blocking: publishing the whole list at once means
        the base system only acts on the last one, and reaching constraints out
        of order is penalised.
        """
        if self._wp_index >= len(self._waypoints):
            self.get_logger().info("path complete")
            self.state = State.DONE
            return

        if elapsed > config.TOTAL_TIME_BUDGET_SEC - 15:
            self.get_logger().warn("time budget nearly spent, stopping")
            self.state = State.DONE
            return

        x, y, _ = self._waypoints[self._wp_index]
        d = math.hypot(self.robot_position_map[0] - x, self.robot_position_map[1] - y)
        now = time.time()

        if d < self._best_dist - STALL_EPS_M:
            self._best_dist, self._stall_since = d, now

        arrived = d < ARRIVE_TOL_M
        stalled = (now - self._stall_since) > STALL_SECONDS
        timed_out = (now - self._wp_started) > WAYPOINT_TIMEOUT_S

        if not (arrived or stalled or timed_out):
            return

        if stalled and not arrived:
            # The base planner keeps its own obstacle inflation and will not
            # drive inside it. A plateau means it is done, not that it needs
            # another 35 seconds.
            self.get_logger().info(
                f"waypoint {self._wp_index + 1} plateaued at {d:.2f} m, advancing")
        elif timed_out and not arrived:
            self.get_logger().warn(
                f"waypoint {self._wp_index + 1} timed out at {d:.2f} m")

        self._wp_index += 1
        if self._wp_index >= len(self._waypoints):
            self.get_logger().info("path complete")
            self.state = State.DONE
            return
        self._start_waypoint()

    # --------------------------------------------------------------- debug

    def publish_graph_markers(self):
        if not self._publish_graph or not self.scene_graph.nodes:
            return
        arr = MarkerArray()
        header = self._map_header()
        for i, n in enumerate(self.scene_graph.nodes):
            m = Marker()
            m.header = header
            m.ns = "scene_graph"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(n.position[0])
            m.pose.position.y = float(n.position[1])
            m.pose.position.z = float(n.position[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(max(n.size[0], 0.15))
            m.scale.y = float(max(n.size[1], 0.15))
            m.scale.z = float(max(n.size[2], 0.15))
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.5, 1.0, 0.35
            arr.markers.append(m)
        self.graph_pub.publish(arr)

    def _map_header(self) -> Header:
        header = Header()
        header.frame_id = "map"
        header.stamp = self.get_clock().now().to_msg()
        return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-return-home", action="store_true",
                    help="answer from wherever exploration ended")
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = VLNChallengeNode(return_home=not args.no_return_home)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
