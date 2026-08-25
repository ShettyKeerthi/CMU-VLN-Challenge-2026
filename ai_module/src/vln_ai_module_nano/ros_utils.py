"""
Conversion helpers between ROS message wire formats and plain numpy/python,
plus the equirectangular camera model used to back-project a 2D detection
box into a 3D lidar point.

Kept dependency-light (no sensor_msgs_py requirement) since PointCloud2
parsing is simple enough to do by hand and this keeps the module portable
across ROS distros.

NOTE ON FRAMES -- this is where the original localization went wrong.
/registered_scan is in the MAP frame. The camera image is in the BODY frame.
Comparing a body-frame ray against map-frame directions is only correct when
the robot's yaw is zero, and the error grows to exactly the yaw angle as it
turns. Every function below that mixes the two now takes `sensor_yaw`.
"""

import math
import struct
from typing import List, Optional, Tuple

import numpy as np

from config import (
    CAMERA_HFOV_DEG,          # must be 360.0 for a full equirect panorama
    CAMERA_TO_SENSOR_TRANSLATION,
    CAMERA_VFOV_DEG,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_RAY_LIDAR_SEARCH_RADIUS_PIX,
)

# Shrink a detection box toward its centre before selecting lidar points. Box
# edges overlap background; background points dragged into the cluster pull the
# centroid off the object.
BOX_SHRINK = 0.25

# Points within this depth of the nearest surface count as the same object.
DEPTH_BAND_M = 1.0

# Percentile used for "nearest surface", so one stray return cannot define the
# depth band.
NEAR_PERCENTILE = 10.0

MIN_CLUSTER_POINTS = 8


# --------------------------------------------------------------- point clouds

def _structured_view(cloud_msg, names: List[str]) -> np.ndarray:
    """Zero-copy structured view over a PointCloud2's buffer.

    Uses each field's declared offset rather than assuming a packed layout.
    This matters: /terrain_map_ext has a 4-byte pad after z, so intensity sits
    at offset 16, not 12, and positional unpacking reads garbage.
    """
    offsets = {f.name: f.offset for f in cloud_msg.fields}
    missing = [n for n in names if n not in offsets]
    if missing:
        raise KeyError(f"PointCloud2 has no field(s) {missing}; "
                       f"available: {sorted(offsets)}")
    if cloud_msg.is_bigendian:
        raise NotImplementedError("big-endian PointCloud2 not handled")

    dtype = np.dtype({
        "names": names,
        "formats": ["<f4"] * len(names),
        "offsets": [offsets[n] for n in names],
        "itemsize": cloud_msg.point_step,
    })
    n = cloud_msg.width * cloud_msg.height
    return np.frombuffer(cloud_msg.data, dtype=dtype, count=n)


def pointcloud2_to_xyz_array(cloud_msg) -> np.ndarray:
    """Parse a sensor_msgs/PointCloud2 into an (N,3) float32 array.

    Vectorized; the previous per-point struct.unpack loop cost real time at
    10 Hz with thousands of points.
    """
    try:
        view = _structured_view(cloud_msg, ["x", "y", "z"])
        pts = np.stack([view["x"], view["y"], view["z"]], axis=1).astype(np.float32)
    except (KeyError, NotImplementedError, ValueError):
        # Fallback: assume float32 x,y,z at the start of each point.
        step = cloud_msg.point_step
        n = cloud_msg.width * cloud_msg.height
        pts = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            pts[i] = struct.unpack_from("fff", cloud_msg.data, i * step)

    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def pointcloud2_to_xyzi_array(cloud_msg, intensity_field: str = "intensity") -> np.ndarray:
    """Parse a PointCloud2 into an (N,4) array of x, y, z, intensity.

    Needed for /terrain_map_ext, where intensity is height above the estimated
    ground plane (measured range 0.00 .. 1.50 m). Dropping that channel makes
    walls indistinguishable from floor.
    """
    view = _structured_view(cloud_msg, ["x", "y", "z", intensity_field])
    pts = np.stack([view["x"], view["y"], view["z"], view[intensity_field]],
                   axis=1).astype(np.float32)
    return pts[np.isfinite(pts).all(axis=1)]


# ------------------------------------------------------------- camera model

def equirect_pixel_to_ray(px: float, py: float) -> np.ndarray:
    """Equirectangular pixel -> unit ray direction in the camera frame.

    Assumes:
      - horizontal axis spans the full 360 deg HFOV, wrapping at the image edges
      - vertical axis spans CAMERA_VFOV_DEG centered on the horizon (row = height/2)

    Camera-frame convention: x forward, y left, z up. px = IMAGE_WIDTH/2 is
    straight ahead. project_points_to_equirect() is the exact inverse of this;
    if you change one, change both.
    """
    yaw_deg = -((px / IMAGE_WIDTH) * CAMERA_HFOV_DEG - (CAMERA_HFOV_DEG / 2.0))
    pitch_deg = (IMAGE_HEIGHT / 2.0 - py) / IMAGE_HEIGHT * CAMERA_VFOV_DEG

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    return np.array([x, y, z], dtype=np.float32)


def project_points_to_equirect(points_body: np.ndarray):
    """Body-frame points (N,3) -> (u, v, range) in equirect pixel coordinates.

    Inverse of equirect_pixel_to_ray. Projecting the whole cloud into image
    space and keeping what lands inside a detection box is more robust than
    testing angular proximity to the box-centre ray: it uses the whole box, so
    a single foreground return cannot hijack the answer.
    """
    x, y, z = points_body[:, 0], points_body[:, 1], points_body[:, 2]
    rng = np.linalg.norm(points_body, axis=1)
    rng = np.where(rng < 1e-6, 1e-6, rng)

    yaw = np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / rng, -1.0, 1.0))

    u = (-np.degrees(yaw) + CAMERA_HFOV_DEG / 2.0) / CAMERA_HFOV_DEG * IMAGE_WIDTH
    v = IMAGE_HEIGHT / 2.0 - np.degrees(pitch) / CAMERA_VFOV_DEG * IMAGE_HEIGHT
    return u, v, rng


def map_to_body(points_map: np.ndarray, sensor_position_map: np.ndarray,
                sensor_yaw: float) -> np.ndarray:
    """Map-frame points -> camera-frame points.

    Rotate by -yaw about z, then subtract the camera's body-frame offset. The
    original code added CAMERA_TO_SENSOR_TRANSLATION; expressing points
    relative to the camera means subtracting it.
    """
    rel = points_map - np.asarray(sensor_position_map, dtype=np.float32)
    c, s = math.cos(-sensor_yaw), math.sin(-sensor_yaw)
    body = np.empty_like(rel)
    body[:, 0] = c * rel[:, 0] - s * rel[:, 1]
    body[:, 1] = s * rel[:, 0] + c * rel[:, 1]
    body[:, 2] = rel[:, 2]
    return body - np.asarray(CAMERA_TO_SENSOR_TRANSLATION, dtype=np.float32)


def _points_in_box(u, v, rng, box_xyxy):
    """Indices of projected points inside the shrunk box, nearest surface only."""
    x0, y0, x1, y1 = box_xyxy
    sx, sy = (x1 - x0) * BOX_SHRINK * 0.5, (y1 - y0) * BOX_SHRINK * 0.5
    x0s, x1s, y0s, y1s = x0 + sx, x1 - sx, y0 + sy, y1 - sy

    in_v = (v >= y0s) & (v <= y1s)
    if x0s < 0 or x1s > IMAGE_WIDTH:
        # Box straddles the panorama seam: wrap the horizontal test.
        uu = np.where(u < IMAGE_WIDTH / 2.0, u + IMAGE_WIDTH, u)
        in_u = (((u >= x0s) & (u <= x1s)) |
                ((uu >= x0s + IMAGE_WIDTH) & (uu <= x1s + IMAGE_WIDTH)))
    else:
        in_u = (u >= x0s) & (u <= x1s)

    idx = np.where(in_u & in_v)[0]
    if len(idx) < MIN_CLUSTER_POINTS:
        return None

    d = rng[idx]
    near = np.percentile(d, NEAR_PERCENTILE)
    keep = idx[d <= near + DEPTH_BAND_M]
    return keep if len(keep) >= MIN_CLUSTER_POINTS else idx


def localize_detection_3d(
    box_xyxy: Tuple[int, int, int, int],
    registered_scan_xyz: np.ndarray,
    sensor_position_map: np.ndarray,
    sensor_yaw: float = 0.0,
) -> Optional[np.ndarray]:
    """Back-project a 2D detection box to a map-frame position.

    sensor_yaw: robot heading in radians, from /state_estimation. It defaults
    to 0.0 only so existing call sites do not break on import -- leaving it
    unset reproduces the original bug, where the estimate rotates around the
    robot as it turns.

    Returns a cluster centroid, not the nearest surface hit. The near face of
    an object sits half its depth toward the robot, which costs IoU even when
    the association is correct.
    """
    if registered_scan_xyz is None or registered_scan_xyz.shape[0] == 0:
        return None

    body = map_to_body(registered_scan_xyz, sensor_position_map, sensor_yaw)
    u, v, rng = project_points_to_equirect(body)
    keep = _points_in_box(u, v, rng, box_xyxy)
    if keep is None:
        return None
    return registered_scan_xyz[keep].mean(axis=0).astype(np.float32)


def localize_detection_3d_with_extent(
    box_xyxy: Tuple[int, int, int, int],
    registered_scan_xyz: np.ndarray,
    sensor_position_map: np.ndarray,
    sensor_yaw: float = 0.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """As above, but also returns an axis-aligned (l, w, h).

    Marker scoring is 3D IoU against ground-truth boxes, so a measured extent
    beats a fixed default. Caveat: lidar only sees the near surface, so depth
    is under-reported for objects viewed from one side.
    """
    if registered_scan_xyz is None or registered_scan_xyz.shape[0] == 0:
        return None

    body = map_to_body(registered_scan_xyz, sensor_position_map, sensor_yaw)
    u, v, rng = project_points_to_equirect(body)
    keep = _points_in_box(u, v, rng, box_xyxy)
    if keep is None:
        return None

    pts = registered_scan_xyz[keep]
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    centre = ((lo + hi) / 2.0).astype(np.float32)
    size = np.maximum(hi - lo, 0.15).astype(np.float32)
    return centre, size


# ------------------------------------------------------------------ markers

def build_marker_message(Marker, header, label: str, position: np.ndarray,
                         size: np.ndarray, marker_id: int = 0, heading: float = 0.0):
    """Build a visualization_msgs/Marker for /selected_object_marker.

    `Marker` and `header` are passed in so this module doesn't hard-depend on
    rclpy message types at import time (keeps it testable without a ROS env).

    The scale floor is 0.15 m, not 0.05: RViz renders a near-zero-volume box as
    nothing at all, and a degenerate scale scores zero IoU either way, so it is
    better to emit something visible and wrong than something invisible.
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
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = float(math.sin(heading / 2.0))
    marker.pose.orientation.w = float(math.cos(heading / 2.0))
    marker.scale.x = float(max(size[0], 0.15))
    marker.scale.y = float(max(size[1], 0.15))
    marker.scale.z = float(max(size[2], 0.15))
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 0.9, 0.2, 0.6
    marker.text = label
    return marker


# -------------------------------------------------------------------- checks

def _self_test():
    """python3 -m ros_utils -- verifies the pixel<->ray round trip.

    If this fails, localization cannot work no matter what else is correct.
    """
    assert abs(CAMERA_HFOV_DEG - 360.0) < 1e-6, \
        f"CAMERA_HFOV_DEG is {CAMERA_HFOV_DEG}, must be 360.0 for an equirect panorama"

    ok = True
    for px, py in [(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2), (100, 200),
                   (IMAGE_WIDTH - 50, IMAGE_HEIGHT - 100), (5, 20)]:
        ray = equirect_pixel_to_ray(px, py)
        u, v, _ = project_points_to_equirect(ray.reshape(1, 3) * 7.0)
        du, dv = abs(u[0] - px), abs(v[0] - py)
        status = "ok " if (du < 1.0 and dv < 1.0) else "FAIL"
        ok &= (du < 1.0 and dv < 1.0)
        print(f"  {status} ({px:7.1f},{py:6.1f}) -> ({u[0]:7.1f},{v[0]:6.1f})  "
              f"err ({du:.2f},{dv:.2f})")
    print("round trip:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    _self_test()
