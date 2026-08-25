"""
Central configuration for the CMU-VLN-Challenge AI module.

Everything a team would realistically need to tune per-scene or per-robot
lives here rather than scattered through the pipeline.
"""

# ---------------------------------------------------------------------------
# ROS topics (must match README exactly -- these are graded on exact names)
# ---------------------------------------------------------------------------
TOPIC_QUESTION = "/challenge_question"          # std_msgs/String, 1 Hz
TOPIC_IMAGE = "/camera/image"                   # sensor_msgs/Image, 10 Hz, 1920x640, 360 HFOV / 120 VFOV
TOPIC_REGISTERED_SCAN = "/registered_scan"      # sensor_msgs/PointCloud2, 5 Hz, map frame
TOPIC_SENSOR_SCAN = "/sensor_scan"              # sensor_msgs/PointCloud2, 5 Hz, sensor_at_scan frame
TOPIC_TERRAIN_MAP = "/terrain_map"              # sensor_msgs/PointCloud2, 5 Hz, map frame, 5m around vehicle
TOPIC_TERRAIN_MAP_EXT = "/terrain_map_ext"      # sensor_msgs/PointCloud2, 5 Hz, map frame, 20m around vehicle
TOPIC_STATE_ESTIMATION = "/state_estimation"    # nav_msgs/Odometry, 100-200 Hz, map -> sensor

TOPIC_WAYPOINT = "/way_point_with_heading"      # geometry_msgs/Pose2D  (output)
TOPIC_OBJECT_MARKER = "/selected_object_marker" # visualization_msgs/Marker (output)
TOPIC_NUMERICAL_RESPONSE = "/numerical_response"  # std_msgs/Int32 (output)

# ---------------------------------------------------------------------------
# Timing budget
# ---------------------------------------------------------------------------
TOTAL_TIME_BUDGET_SEC = 10 * 60          # hard limit per question (README)
EXPLORATION_TIME_BUDGET_SEC = 7 * 60     # stop exploring and commit to an answer with buffer left
MAIN_LOOP_HZ = 2.0

# ---------------------------------------------------------------------------
# Camera model (360 deg HFOV equirectangular, 120 deg VFOV, 1920x640)
# TODO: verify VFOV convention (centered on horizon vs offset) against a real
# bag from the Real-Robot Data sample -- this assumes centered on horizon.
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 640
CAMERA_HFOV_DEG = 360.0
CAMERA_VFOV_DEG = 120.0

# Camera frame -> sensor (lidar) frame extrinsic. README says this is
# measured from a CAD model and provided in the real-robot sample README.
# TODO: replace with the real calibration once you pull the sample data from
# the Google Drive link -- this identity placeholder assumes co-located
# origins, which is wrong and will bias 3D object localization.
CAMERA_TO_SENSOR_TRANSLATION = (0.0, 0.0, 0.0)  # (x, y, z) meters

# ---------------------------------------------------------------------------
# Detection / scene graph
# ---------------------------------------------------------------------------
OPEN_VOCAB_MODEL_NAME = "google/owlv2-base-patch16-ensemble"
DETECTION_SCORE_THRESHOLD = 0.3   # TODO: tune per-object like NanoOWL experience taught (0.1 start, per-object refine)
MAX_RAY_LIDAR_SEARCH_RADIUS_PIX = 15  # bbox-center pixel search radius when hunting for a lidar return
#DETECTION_DEDUP_DIST_M = 0.5       # merge detections of the same label within this 3D radius into one node
DETECTION_DEDUP_DIST_M = 1.2 
FRAME_SAMPLE_EVERY_N = 5           # only run the detector every Nth image (10 Hz camera -> 2 Hz detection)

# Basic named-color palette for attribute estimation from mean crop color.
# TODO: this is a coarse baseline -- fine-tune against VLA-3D color labels
# for your training scenes rather than relying on this fixed palette.
NAMED_COLORS_RGB = {
    "red": (200, 30, 30),
    "orange": (230, 120, 20),
    "yellow": (220, 200, 40),
    "green": (40, 150, 60),
    "blue": (40, 80, 190),
    "purple": (120, 50, 150),
    "pink": (230, 130, 180),
    "brown": (110, 70, 40),
    "black": (25, 25, 25),
    "white": (235, 235, 235),
    "gray": (130, 130, 130),
}

# ---------------------------------------------------------------------------
# Exploration (frontier-based)
# ---------------------------------------------------------------------------
OCC_GRID_CELL_SIZE_M = 0.25
OCC_GRID_RADIUS_M = 20.0          # matches terrain_map_ext coverage
MIN_FRONTIER_CLUSTER_SIZE = 4
# WAYPOINT_REACHED_TOL_M = 0.75
WAYPOINT_REACHED_TOL_M = 0.5

# ---------------------------------------------------------------------------
# LLM reasoning backend
# ---------------------------------------------------------------------------
# ANTHROPIC_MODEL = "claude-sonnet-4-6"
# ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

ANTHROPIC_MODEL = "openai/gpt-oss-120b"
ANTHROPIC_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_API_KEY = "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"
