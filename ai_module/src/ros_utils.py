"""
Conversion helpers between ROS message wire formats and plain numpy/python,
plus the equirectangular camera model used to back-project a 2D detection
box into a 3D lidar point.

Kept dependency-light (no sensor_msgs_py requirement) since PointCloud2
parsing is simple enough to do by hand and this keeps the module portable
across ROS distros.
"""

import struct
from typing import List, Optional, Tuple

import numpy as np

from config import (
    CAMERA_HFOV_DEG,
    CAMERA_TO_SENSOR_TRANSLATION,
    CAMERA_VFOV_DEG,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_RAY_LIDAR_SEARCH_RADIUS_PIX,
)


def pointcloud2_to_xyz_array(cloud_msg) -> np.ndarray:
    """Parses a sensor_msgs/PointCloud2 with float32 x,y,z fields (the common
    case for /registered_scan, /sensor_scan, /terrain_map*) into an (N,3) array.
    Assumes no padding beyond x,y,z,[rgb/intensity...] -- adjust `point_step`
    field offsets here if your bag uses a different field layout; check
    `cloud_msg.fields` at runtime if unsure.
    """
    fmt = "fff"
    step = cloud_msg.point_step
    n_points = cloud_msg.width * cloud_msg.height
    data = cloud_msg.data
    pts = np.empty((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        offset = i * step
        x, y, z = struct.unpack_from(fmt, data, offset)
        pts[i] = (x, y, z)
    # Drop NaN/inf rows (common in registered scans near sensor origin)
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def equirect_pixel_to_ray(px: float, py: float) -> np.ndarray:
    """Converts an equirectangular pixel coordinate to a unit ray direction
    in the camera frame. Assumes:
      - horizontal axis spans the full 360 deg HFOV, wrapping at the image edges
      - vertical axis spans CAMERA_VFOV_DEG centered on the horizon (row = height/2)
    TODO: verify the vertical centering assumption against a real image from
    the sample bag -- if the camera is tilted or the FOV isn't horizon-centered
    this offset needs adjusting.
    """
    yaw_deg = (px / IMAGE_WIDTH) * CAMERA_HFOV_DEG - (CAMERA_HFOV_DEG / 2.0)
    pitch_deg = (IMAGE_HEIGHT / 2.0 - py) / IMAGE_HEIGHT * CAMERA_VFOV_DEG

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Camera-frame convention: x forward, y left, z up
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    return np.array([x, y, z], dtype=np.float32)


def localize_detection_3d(
    box_xyxy: Tuple[int, int, int, int],
    registered_scan_xyz: np.ndarray,
    sensor_position_map: np.ndarray,
) -> Optional[np.ndarray]:
    """Back-projects a 2D detection box center into a 3D map-frame position by
    finding the closest lidar return along the corresponding camera ray.

    This is the least mature part of the pipeline -- it works by angular
    proximity (yaw/pitch of each lidar point vs the ray) rather than a proper
    calibrated projection matrix, because the exact camera/lidar extrinsic
    calibration is only given in the real-robot sample data README, not in
    this repo's docs. Swap in a proper projection once you've pulled that.
    """
    if registered_scan_xyz.shape[0] == 0:
        return None

    cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
    cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
    ray = equirect_pixel_to_ray(cx, cy)

    rel = registered_scan_xyz - sensor_position_map  # point position relative to the sensor (lidar) origin
    # CAMERA_TO_SENSOR_TRANSLATION is the camera's physical offset FROM the
    # sensor (e.g. (0,0,0.1) = camera sits 0.1m above the lidar). To express
    # a point relative to the CAMERA instead of the sensor, subtract that
    # offset -- NOT add it. (Previously added by mistake when the constant
    # was a (0,0,0) placeholder, which hid the bug since the sign didn't
    # matter at zero. Now that it's a real nonzero offset, this must be a
    # subtraction or every ray-to-point angle will be systematically wrong.)
    rel = rel - np.array(CAMERA_TO_SENSOR_TRANSLATION, dtype=np.float32)
    dist = np.linalg.norm(rel, axis=1)
    dist_safe = np.where(dist < 1e-6, 1e-6, dist)
    rel_unit = rel / dist_safe[:, None]

    cos_angle = rel_unit @ ray  # cosine similarity between each point's direction and the ray
    angular_tol_cos = np.cos(np.radians(MAX_RAY_LIDAR_SEARCH_RADIUS_PIX * (CAMERA_HFOV_DEG / IMAGE_WIDTH)))
    candidates = np.where(cos_angle > angular_tol_cos)[0]
    if len(candidates) == 0:
        return None

    # Among angularly-aligned points, take the closest one (first surface hit)
    best_idx = candidates[np.argmin(dist[candidates])]
    return registered_scan_xyz[best_idx]


def build_marker_message(Marker, header, label: str, position: np.ndarray, size: np.ndarray, marker_id: int = 0):
    """Builds a visualization_msgs/Marker for /selected_object_marker.
    `Marker` and `header` are passed in so this module doesn't hard-depend on
    rclpy message types at import time (keeps it testable without a ROS env).
    """
    marker = Marker()
    marker.header = header
    marker.ns = "selected_object"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(max(size[0], 0.05))
    marker.scale.y = float(max(size[1], 0.05))
    marker.scale.z = float(max(size[2], 0.05))
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 0.9, 0.2, 0.6
    marker.text = label
    return marker
