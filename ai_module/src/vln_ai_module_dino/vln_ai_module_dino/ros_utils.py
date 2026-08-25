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

from .config import (
    CAMERA_HFOV_DEG,
    CAMERA_TO_SENSOR_TRANSLATION,
    CAMERA_VFOV_DEG,
    DEPTH_CLUSTER_TOLERANCE_M,
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
    pts = pts[mask]

    # SENSOR-ORIGIN SENTINEL FIX 2026-08-12: confirmed live (arabic_room,
    # multiple runs, recurring across sessions) that spurious detections
    # localize to EXACTLY (0.00, 0.00, 0.00) in map frame with a real,
    # tight lidar cluster behind them (not the generic single-ray fallback
    # -- see localize_detection_3d in this file) -- i.e. actual points in
    # registered_scan_xyz sit exactly at the map origin. This is a known,
    # common lidar-driver convention: invalid/no-return points are often
    # published as (0,0,0) rather than NaN, so the NaN/inf filter above
    # never catches them. A real point landing exactly at the map's global
    # origin (as opposed to merely near it) is astronomically unlikely in
    # any real scene -- treat it as a sentinel/invalid-return marker and
    # drop it here, at the parsing source, rather than trying to filter it
    # back out downstream in every consumer (detection localization, scene
    # graph, etc) individually. Fixing it here is scene-agnostic: it
    # applies to /registered_scan, /sensor_scan, and /terrain_map* alike,
    # on any scene, regardless of what's actually being detected.
    origin_dist = np.linalg.norm(pts, axis=1)
    mask_not_origin = origin_dist > 1e-3
    return pts[mask_not_origin]


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


def _ray_to_equirect_pixel(direction: np.ndarray) -> Tuple[float, float]:
    """Inverse of equirect_pixel_to_ray: world unit direction -> equirect
    pixel coordinates, wraparound-safe on yaw."""
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arcsin(np.clip(direction[2], -1.0, 1.0))
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    px = (yaw_deg + CAMERA_HFOV_DEG / 2.0) / CAMERA_HFOV_DEG * IMAGE_WIDTH
    py = IMAGE_HEIGHT / 2.0 - (pitch_deg / CAMERA_VFOV_DEG) * IMAGE_HEIGHT
    return px, py


def equirect_to_perspective_tile(
    equirect_arr: np.ndarray,
    yaw_center_deg: float,
    tile_hfov_deg: float,
    tile_vfov_deg: float,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    """Reprojects a rectilinear (normal-perspective) sub-view out of the full
    360 equirectangular frame, centered at yaw_center_deg, looking level
    (pitch=0).

    WHY THIS EXISTS 2026-08-10: confirmed live (arabic_room, both Owlv2 and
    Grounding DINO) that feeding the detector the raw, undistorted-nowhere
    360 equirect frame directly produces confident false-positive "sofa"
    detections landing specifically on bare walls -- not on any other real
    furniture category, no consistent single culprit object, just wall
    regions. Equirectangular projection warps straight lines (a flat wall)
    into curves, especially away from the horizontal center; a detector
    trained overwhelmingly on normal (rectilinear) photographs has likely
    never seen this kind of distortion and may be misreading warped wall
    geometry as furniture-like shapes. This function generates a normal,
    undistorted-looking crop the way a standard camera would see it, so the
    detector gets geometry it actually knows how to interpret.

    Nearest-neighbor sampling (not bilinear) -- simpler, and detection
    doesn't need photographic smoothness the way a human viewer would.
    """
    yaw_c = np.radians(yaw_center_deg)
    forward = np.array([np.cos(yaw_c), np.sin(yaw_c), 0.0])
    right = np.array([np.sin(yaw_c), -np.cos(yaw_c), 0.0])
    up = np.array([0.0, 0.0, 1.0])

    u = (np.arange(tile_w) + 0.5) / tile_w * 2 - 1  # [-1, 1]
    v = (np.arange(tile_h) + 0.5) / tile_h * 2 - 1
    uu, vv = np.meshgrid(u, v)  # (tile_h, tile_w)

    right_comp = uu * np.tan(np.radians(tile_hfov_deg / 2.0))
    up_comp = -vv * np.tan(np.radians(tile_vfov_deg / 2.0))  # image row increases downward

    # direction = forward + right_comp*right + up_comp*up, per output pixel
    directions = (
        forward[None, None, :]
        + right_comp[:, :, None] * right[None, None, :]
        + up_comp[:, :, None] * up[None, None, :]
    )
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    directions = directions / np.where(norms < 1e-9, 1e-9, norms)

    yaw = np.arctan2(directions[:, :, 1], directions[:, :, 0])
    pitch = np.arcsin(np.clip(directions[:, :, 2], -1.0, 1.0))
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)

    src_x = (yaw_deg + CAMERA_HFOV_DEG / 2.0) / CAMERA_HFOV_DEG * IMAGE_WIDTH
    src_y = IMAGE_HEIGHT / 2.0 - (pitch_deg / CAMERA_VFOV_DEG) * IMAGE_HEIGHT

    src_x = np.mod(np.round(src_x).astype(np.int32), IMAGE_WIDTH)  # wraps at the yaw seam
    src_y = np.clip(np.round(src_y).astype(np.int32), 0, IMAGE_HEIGHT - 1)

    return equirect_arr[src_y, src_x]


def perspective_box_to_equirect_box(
    box_xyxy: Tuple[int, int, int, int],
    yaw_center_deg: float,
    tile_hfov_deg: float,
    tile_vfov_deg: float,
    tile_w: int,
    tile_h: int,
) -> Tuple[float, float, float, float]:
    """Maps a detection box found in tile pixel space back to full
    equirectangular pixel space, via the same forward projection used to
    render the tile. Approximates the box as the min/max of its 4 corners'
    mapped positions -- a rectangle in perspective space isn't exactly a
    rectangle in equirect space, but this is a standard, good-enough
    approximation for typical detection box sizes.

    WRAPAROUND FIX 2026-08-10: confirmed live that this is NOT a rare edge
    case -- since tile yaw centers are spread across the full 360, at least
    one tile is always near the +-180 seam, and any box near THAT tile's
    own center legitimately straddles the seam (a 75deg-wide tile centered
    at 179deg spans 141.5..216.5deg, and 216.5 wraps to -143.5). Returns x1
    possibly > IMAGE_WIDTH (an "extended" un-wrapped representation) rather
    than forcing everything back into [0, IMAGE_WIDTH) -- localize_detection_3d
    is written to handle that extended range consistently rather than
    silently producing a nonsensical inverted-order box (x0 > x1) the way
    naively re-wrapping would.
    """
    yaw_c = np.radians(yaw_center_deg)
    forward = np.array([np.cos(yaw_c), np.sin(yaw_c), 0.0])
    right = np.array([np.sin(yaw_c), -np.cos(yaw_c), 0.0])
    up = np.array([0.0, 0.0, 1.0])

    x0, y0, x1, y1 = box_xyxy
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    eq_xs, eq_ys = [], []
    for (cu, cv) in corners:
        u_ndc = (cu / tile_w) * 2 - 1
        v_ndc = (cv / tile_h) * 2 - 1
        right_comp = u_ndc * np.tan(np.radians(tile_hfov_deg / 2.0))
        up_comp = -v_ndc * np.tan(np.radians(tile_vfov_deg / 2.0))
        direction = forward + right_comp * right + up_comp * up
        direction = direction / np.linalg.norm(direction)
        px, py = _ray_to_equirect_pixel(direction)
        eq_xs.append(px)
        eq_ys.append(py)

    eq_xs = np.array(eq_xs)
    if eq_xs.max() - eq_xs.min() > IMAGE_WIDTH / 2:  # straddles the seam
        eq_xs = np.where(eq_xs < IMAGE_WIDTH / 2, eq_xs + IMAGE_WIDTH, eq_xs)

    x_lo, x_hi = float(eq_xs.min()), float(eq_xs.max())
    y_lo = float(np.clip(min(eq_ys), 0, IMAGE_HEIGHT - 1))
    y_hi = float(np.clip(max(eq_ys), 0, IMAGE_HEIGHT - 1))
    return (x_lo, y_lo, x_hi, y_hi)


def localize_detection_3d(
    box_xyxy: Tuple[int, int, int, int],
    registered_scan_xyz: np.ndarray,
    sensor_position_map: np.ndarray,
) -> Optional[np.ndarray]:
    """Back-projects a 2D detection box into a 3D map-frame position.

    UPGRADED 2026-08-07: previously used only the box's CENTER pixel -> one
    ray -> nearest single lidar point along it. Confirmed live (arabic_room)
    that this causes real problems for large objects (sofas ~2.2m long):
    if only part of the object is visible/in-frame, the box center -- and
    therefore the localized position -- lands on whichever part is visible,
    not the object's true center. As the robot walks past, seeing different
    partial slices from different positions can localize the SAME physical
    sofa to points more than a meter apart, defeating scene-graph dedup and
    inflating counts with what are really repeat sightings of one object.

    Now uses the box's FULL angular footprint (not just its center) to
    select every lidar point whose direction falls inside it, and takes the
    MEDIAN of those points -- this naturally averages over whatever portion
    of the object is actually visible, landing much closer to the object's
    true center regardless of partial occlusion or viewing angle. Falls
    back to the old single-ray-nearest-point behavior if the footprint
    query finds nothing (e.g. very sparse lidar coverage at that moment).

    GEOMETRIC-SIZE ADDITION 2026-08-12: also returns a rough (width, depth,
    height) extent of the near-surface point cluster, in addition to its
    median position -- this is what geometric_category_check() in
    scene_graph.py needs to sanity-check a label against real 3D shape
    (e.g. catching the column-misclassified-as-sofa case). Uses the
    near-surface cluster's own axis-aligned bounding box in map frame as a
    cheap, dependency-free size estimate -- not a true oriented bbox, but
    good enough to separate "tall and thin" from "low and wide" shapes.
    Falls back to a generic default size when only the single-ray fallback
    path fires (no real cluster to measure), since in that case there's no
    hard shape data available and geometric_category_check should treat it
    as inconclusive rather than confidently wrong.

    Returns: (position (3,), size (3,)) or None if nothing could be localized.
    """
    FALLBACK_SIZE = np.array([0.4, 0.4, 0.4], dtype=np.float32)

    if registered_scan_xyz.shape[0] == 0:
        return None

    rel = registered_scan_xyz - sensor_position_map  # point position relative to the sensor (lidar) origin
    rel = rel - np.array(CAMERA_TO_SENSOR_TRANSLATION, dtype=np.float32)
    dist = np.linalg.norm(rel, axis=1)
    dist_safe = np.where(dist < 1e-6, 1e-6, dist)
    rel_unit = rel / dist_safe[:, None]

    # Per-point yaw/pitch, inverse of equirect_pixel_to_ray's forward
    # transform (camera-frame convention: x forward, y left, z up).
    point_yaw = np.arctan2(rel_unit[:, 1], rel_unit[:, 0])
    point_pitch = np.arctan2(rel_unit[:, 2], np.hypot(rel_unit[:, 0], rel_unit[:, 1]))

    x0, y0, x1, y1 = box_xyxy
    yaw0_deg = (x0 / IMAGE_WIDTH) * CAMERA_HFOV_DEG - (CAMERA_HFOV_DEG / 2.0)
    yaw1_deg = (x1 / IMAGE_WIDTH) * CAMERA_HFOV_DEG - (CAMERA_HFOV_DEG / 2.0)
    pitch0_deg = (IMAGE_HEIGHT / 2.0 - y0) / IMAGE_HEIGHT * CAMERA_VFOV_DEG
    pitch1_deg = (IMAGE_HEIGHT / 2.0 - y1) / IMAGE_HEIGHT * CAMERA_VFOV_DEG
    yaw_lo_deg, yaw_hi_deg = min(yaw0_deg, yaw1_deg), max(yaw0_deg, yaw1_deg)
    pitch_lo, pitch_hi = np.radians(min(pitch0_deg, pitch1_deg)), np.radians(max(pitch0_deg, pitch1_deg))

    # WRAPAROUND FIX 2026-08-10: box_xyxy can now come from
    # perspective_box_to_equirect_box with x1 > IMAGE_WIDTH (a tile near the
    # +-180 seam produces this "extended" representation -- see that
    # function's docstring). point_yaw from arctan2 is always in [-180,180]
    # though, so a point whose true yaw is e.g. -178 (== +182 wrapped) would
    # wrongly fail a direct comparison against yaw_hi=182. Shift any point
    # that's "too low" to plausibly belong up into the same extended domain
    # as the box before comparing -- without this, every detection from a
    # near-seam tile silently loses its lidar footprint and localizes to
    # nothing (or, worse, to a partial/wrong footprint).
    point_yaw_deg = np.degrees(point_yaw)
    if yaw_hi_deg > 180.0:
        point_yaw_deg = np.where(point_yaw_deg < yaw_lo_deg - 180.0, point_yaw_deg + 360.0, point_yaw_deg)
        point_yaw = np.radians(point_yaw_deg)
    yaw_lo, yaw_hi = np.radians(yaw_lo_deg), np.radians(yaw_hi_deg)

    in_footprint = np.where(
        (point_yaw >= yaw_lo) & (point_yaw <= yaw_hi) &
        (point_pitch >= pitch_lo) & (point_pitch <= pitch_hi)
    )[0]

    if len(in_footprint) > 0:
        # DEPTH-CLUSTERING FIX 2026-08-10: confirmed live (arabic_room,
        # Grounding DINO) that plain median-of-footprint was a real
        # regression for exactly the case it was meant to help -- a
        # detection box often also contains BACKGROUND (wall/floor visible
        # around or behind the object), and if the box catches more
        # background lidar points than object points, the median gets
        # pulled onto the background, not the object. Traced multiple
        # false-positive "sofa" detections directly onto real walls,
        # columns, and floors sitting near genuine detections -- a
        # localization bug, not a classification one; ground truth showed
        # no single object type being confused for sofa. Fix: within the
        # footprint, cluster by depth and keep only points near the
        # NEAREST surface -- the closest hit along most rays in a box is
        # very likely the actual foreground object the detector boxed, not
        # whatever's behind it.
        footprint_dist = dist[in_footprint]
        nearest_depth = np.min(footprint_dist)
        near_surface = in_footprint[footprint_dist < nearest_depth + DEPTH_CLUSTER_TOLERANCE_M]
        cluster_pts = registered_scan_xyz[near_surface]
        position = np.median(cluster_pts, axis=0)
        if cluster_pts.shape[0] >= 3:
            extent = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
            # Floor tiny/degenerate extents (e.g. a near-planar cluster
            # viewed edge-on) so downstream footprint/slenderness math in
            # geometric_category_check never divides by ~0.
            size = np.maximum(extent, 0.05).astype(np.float32)
        else:
            size = FALLBACK_SIZE.copy()
        return position, size

    # Fallback: original narrow single-ray nearest-point behavior.
    cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
    cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
    ray = equirect_pixel_to_ray(cx, cy)
    cos_angle = rel_unit @ ray
    angular_tol_cos = np.cos(np.radians(MAX_RAY_LIDAR_SEARCH_RADIUS_PIX * (CAMERA_HFOV_DEG / IMAGE_WIDTH)))
    candidates = np.where(cos_angle > angular_tol_cos)[0]
    if len(candidates) == 0:
        return None
    best_idx = candidates[np.argmin(dist[candidates])]
    # No real cluster here (single nearest point only) -- return the
    # generic fallback size rather than a fabricated (0,0,0) extent, so
    # geometric_category_check sees this as genuinely inconclusive.
    return registered_scan_xyz[best_idx], FALLBACK_SIZE.copy()


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
