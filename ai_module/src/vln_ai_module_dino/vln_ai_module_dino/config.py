"""
Central configuration for the CMU-VLN-Challenge AI module.

Everything a team would realistically need to tune per-scene or per-robot
lives here rather than scattered through the pipeline.
"""

# ---------------------------------------------------------------------------
# ROS topics (must match README exactly -- these are graded on exact names)
# ---------------------------------------------------------------------------
TOPIC_QUESTION = "/challenge_question"          # std_msgs/String, 1 Hz
# NOTE 2026-08-06: subscribing to the compressed topic directly, NOT the
# README's documented `/camera/image` (sensor_msgs/Image). Verified against
# a live sim run that the `sim_image_repub` node responsible for
# uncompressing `/camera/image/compressed` -> `/camera/image` never
# actually publishes anything (confirmed alive via `ros2 node info`, matching
# QoS via `ros2 topic info --verbose` on both ends, real PNG data confirmed
# flowing on the compressed topic via `ros2 topic echo` -- yet `ros2 topic hz
# /camera/image` and `ros2 topic echo /camera/image --once` both show zero
# messages, indefinitely). This looks like a bug in the vendor-shipped
# iros2026_system image, not anything in our config/QoS. Bypassing it and
# decoding the PNG ourselves in main_node.py is more robust regardless.
TOPIC_IMAGE = "/camera/image/compressed"        # sensor_msgs/CompressedImage, PNG, 10 Hz, 1920x640, 360 HFOV / 120 VFOV
TOPIC_REGISTERED_SCAN = "/registered_scan"      # sensor_msgs/PointCloud2, 5 Hz, map frame
TOPIC_SENSOR_SCAN = "/sensor_scan"              # sensor_msgs/PointCloud2, 5 Hz, sensor_at_scan frame
TOPIC_TERRAIN_MAP = "/terrain_map"              # sensor_msgs/PointCloud2, 5 Hz, map frame, 5m around vehicle
TOPIC_TERRAIN_MAP_EXT = "/terrain_map_ext"      # sensor_msgs/PointCloud2, 5 Hz, map frame, 20m around vehicle
TOPIC_STATE_ESTIMATION = "/state_estimation"    # nav_msgs/Odometry, 100-200 Hz, map -> sensor

TOPIC_WAYPOINT = "/way_point_with_heading"      # geometry_msgs/Pose2D  (output)
TOPIC_OBJECT_MARKER = "/selected_object_marker" # visualization_msgs/Marker (output)
TOPIC_PLANNED_PATH_MARKER = "/planned_path_marker"  # visualization_msgs/Marker LINE_STRIP (output)
# PLANNED PATH VISUALIZATION 2026-08-20: shows the FULL computed waypoint
# sequence for instruction_following as a line in RViz, published as soon
# as build_waypoint_sequence() returns -- before the robot has physically
# moved at all. Previously the only way to see the plan was to watch the
# robot move (slow, and /trajectory only shows where it's ALREADY been) or
# read raw coordinates off /way_point_with_heading in a terminal. This
# gives an immediate visual sanity check against the question, the same
# way the official ground-truth answer images show a green path -- useful
# for catching an obviously-wrong plan before spending the time watching
# the robot actually attempt it.
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

# Camera frame -> sensor (lidar) frame extrinsic.
# SOURCE (verified from the actual repo, not a guess): the base autonomy
# stack publishes this as a static TF in local_planner.launch:
#   tf2_ros static_transform_publisher: "0 0 $(var cameraOffsetZ) -1.5707963 0 -1.5707963 /sensor /camera"
# i.e. the camera sits cameraOffsetZ meters straight above the lidar origin
# along the sensor frame's z-axis, no x/y offset. cameraOffsetZ itself is
# set per-launch-file:
#   system_simulation.launch          -> cameraOffsetZ = 0.1   (sim / training scenes / first eval round)
#   system_real_robot.launch          -> cameraOffsetZ = 0.25  (final real-robot round)
# The README states images are remapped so the camera and lidar frames stay
# orientation-aligned, so only the translation (not the -90/-90 rotation in
# the TF above) needs to be applied here.
# Use the simulation value until you're actually testing on the real robot.
CAMERA_TO_SENSOR_TRANSLATION = (0.0, 0.0, 0.1)  # (x, y, z) meters -- SIMULATION value
CAMERA_TO_SENSOR_TRANSLATION_REAL_ROBOT = (0.0, 0.0, 0.25)  # swap in for the final round

# ---------------------------------------------------------------------------
# Detection / scene graph
# ---------------------------------------------------------------------------
# MODEL SWAP 2026-08-07: replaced Owlv2 with Grounding DINO after extensive
# live testing (arabic_room) showed Owlv2 producing a high false-positive
# rate (e.g. ~17-22 "sofa" detections against only 3 real ones in the
# scene) that no amount of downstream logic tuning (dedup, consolidation,
# edge-filtering, relation-radius tuning) could fully correct -- the
# detector itself was the precision bottleneck, not the pipeline around it.
# Grounding DINO generally scores meaningfully better than OWL-ViT on
# zero-shot/open-vocabulary detection benchmarks (LVIS zero-shot, ODinW),
# due to deeper text-image cross-attention fusion throughout the network
# rather than OWL-ViT's more CLIP-like late-fusion matching. Using the
# "tiny" checkpoint (Swin-T backbone) rather than "base" (Swin-B) since
# inference here is CPU-bound (see the CUDA driver-version warning at
# startup) -- base is more accurate but noticeably slower; swap to
# "IDEA-Research/grounding-dino-base" if running on hardware with working
# CUDA and inference speed isn't the bottleneck.
OPEN_VOCAB_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
# Grounding DINO uses two separate thresholds (box confidence, text-match
# confidence) rather than OWL-ViT's single score threshold. Both start at
# the values from Grounding DINO's own usage docs -- TODO: tune against
# real scenes the same way DETECTION_SCORE_THRESHOLD needed tuning for
# Owlv2, don't assume these are already right for this task.
# THRESHOLD TUNING 2026-08-12: raising both thresholds to 0.45/0.35 was
# tried first, but confirmed live (arabic_room) that it cost real recall on
# OTHER categories (window detections dropped from several per run to just
# 1), which directly capped numerical "below a window" counts regardless of
# how many real sofa-below-window pairs existed -- a net accuracy loss for
# this question type, not a gain. Reverted most of the way back toward the
# original values; precision on sofa-category noise is instead handled by
# is_degenerate_cluster() and geometric_category_check() in scene_graph.py,
# which cost zero recall (they only reject/relabel implausible SHAPES, not
# lower-confidence-but-real detections). A small bump (0.35->0.38,
# 0.30->0.32) is kept as a mild, lower-risk noise reduction; if this still
# hurts recall on low-contrast categories, revert fully to 0.35/0.30 and
# rely entirely on the geometric filters.
GROUNDING_DINO_BOX_THRESHOLD = 0.38
GROUNDING_DINO_TEXT_THRESHOLD = 0.32

# DISTRACTOR CATEGORIES 2026-08-10: confirmed live (arabic_room, multiple
# runs) that a real COLUMN at (-0.99, 1.08) was being confidently (conf
# 0.57-0.94) misclassified as "sofa" -- 0.10m from its true position, not a
# coincidence. Grounding DINO does phrase grounding: candidate text phrases
# compete for image regions. If the only options given are the actual
# target/anchor categories (e.g. just "sofa" and "window"), any visually
# ambiguous region gets forced into whichever of those it resembles most,
# however poorly -- there's no correct alternative to compete against.
# Always including these common architectural distractors as EXTRA
# competing phrases (never added to the scene graph themselves -- see
# _run_detection in main_node.py) gives the model a genuine right answer to
# route ambiguous regions to instead of a furniture category by default.
VOCABULARY_DISTRACTOR_CATEGORIES = [
    "a column", "a pillar", "a wall", "a door", "a door frame", "a curtain",
]

# CATEGORY SYNONYMS 2026-08-17: Grounding DINO's phrase grounding returns
# the ACTUAL matched text span, not a fixed category index -- if two
# related phrases are both in the vocabulary (e.g. "column" and "pillar"
# are both in VOCABULARY_DISTRACTOR_CATEGORIES above), the model may ground
# a real physical object to EITHER phrase depending on which one scores
# higher for that specific detection, not necessarily the one the current
# question actually asked for. Confirmed live: "take the path between the
# two columns" needs "column" in _needed_categories, but if DINO grounds
# the real column to the text "pillar" instead, the detection was
# previously silently discarded by the distractor filter in
# _run_detection (is_distractor=True since "pillar" is on the distractor
# list, is_needed=False since "column" != "pillar") -- even though it's
# the same physical object the question needs. Normalizing known synonyms
# to one canonical label BEFORE the distractor/needed check and before the
# scene graph ever sees it fixes this for both directions (a detection
# grounded to either word is treated identically downstream).
CATEGORY_SYNONYMS = {
    "pillar": "column",
}

# CATEGORY PROMPT VARIANTS 2026-08-17: confirmed live -- across 5
# independent exploration runs, Grounding DINO produced ZERO raw
# detections for either "a column" or "a pillar" against a scene with 2
# confirmed real, prominent columns (verified against the official
# ground-truth answer image for this question). This rules out both the
# geometric shape filter and the synonym-normalization gap as the cause --
# the raw detection debug print (see perception.py's detect()) never fired
# even once, meaning the model itself never proposes a matching box at
# either phrasing, at any distance/angle across 5 different exploration
# paths. Grounding models are typically trained on caption-style text, so
# a bare category noun sometimes scores far worse than a visually
# descriptive phrase for the same object -- add extra, richer phrasings to
# the vocabulary for known-hard categories, and let the existing
# CATEGORY_SYNONYMS normalization (substring match on "pillar"/"column")
# fold any of these back to the canonical label if one succeeds where the
# bare noun didn't.
CATEGORY_PROMPT_VARIANTS = {
    "column": [
        "a tall stone pillar", "a decorative column", "a round pillar",
        "a support column", "a carved pillar",
    ],
}

# --- 360-view tiling for detection (see detect_360() in perception.py) ---
# WHY: confirmed live that feeding the detector the raw equirectangular
# frame directly produced confident false-positive detections landing
# specifically on bare walls -- the projection warps straight lines into
# curves, and a detector trained on normal photographs has essentially
# never seen this kind of distortion. Splitting into several rectilinear
# (normal-perspective) sub-views gives the detector geometry it actually
# knows how to interpret.
#
# 6 tiles, 60deg apart, each 75deg wide -> 15deg of overlap between
# neighbors (7.5deg each side). The overlap matters for two reasons: (1) an
# object near one tile's edge is very likely fully visible, un-truncated,
# in a neighboring tile, so the partial-view edge-rejection in detect()
# has a better chance of still finding a clean view of it somewhere; (2) it
# reduces the odds of an object landing exactly on a tile boundary in every
# single tile that could see it.
TILE_YAW_CENTERS_DEG = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
TILE_HFOV_DEG = 75.0
# Less than the camera's full 120 VFOV -- perspective (rectilinear)
# projection distorts increasingly badly as FOV approaches 180, so this
# trades a bit of extreme up/down coverage (mostly ceiling/floor) for
# meaningfully less distortion in the range that actually matters (most
# furniture/fixtures are within +-50deg of the horizon, not at the extreme
# edges of the camera's vertical range).
TILE_VFOV_DEG = 100.0
# Chosen close to the tile's natural angular aspect ratio
# (tan(VFOV/2)/tan(HFOV/2) ~= 1.55) to avoid excessive stretching, rounded
# to convenient numbers rather than matched exactly.
TILE_OUTPUT_WIDTH = 640
TILE_OUTPUT_HEIGHT = 960
MAX_RAY_LIDAR_SEARCH_RADIUS_PIX = 15  # bbox-center pixel search radius when hunting for a lidar return

# Depth-clustering tolerance for localize_detection_3d: within a detection
# box's angular footprint, only keep lidar points within this distance of
# the NEAREST point found, discarding anything farther (almost certainly
# background wall/floor behind the actual object, not the object itself).
# Most furniture is well under a meter deep, so this stays well short of
# "another piece of furniture a meter behind this one" while still being
# generous enough to capture a whole sofa's front-to-back depth. TODO: this
# is a reasonable starting guess, not empirically tuned per-category --
# revisit if large flat objects (e.g. "wall lamp" against its own wall)
# behave differently than furniture with real depth.
DEPTH_CLUSTER_TOLERANCE_M = 0.6
DETECTION_DEDUP_DIST_M = 1.2       # merge detections of the same label within this 3D radius into one node
# TUNING NOTE 2026-08-07: raised from 0.5m after validating against real
# ground truth (arabic_room/object_list.txt). At 0.5m, the scene graph
# fractured 3 real sofas into 21 nodes and 4 real windows into 11 nodes --
# large furniture (sofas here are ~2.2m long per their bounding boxes)
# viewed from different angles easily produces centroid estimates further
# apart than 0.5m, so the old threshold was splitting single physical
# objects into many nodes instead of merging them. 1.2m is closer to half
# a typical sofa's length -- generous enough to merge same-object repeat
# sightings, still tight enough not to merge two genuinely distinct nearby
# objects. Chosen to track furniture scale (should transfer reasonably
# across scenes) rather than this scene's specific layout.
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

# Stuck-detection: if the robot hasn't moved this far in this many seconds
# while actively pursuing a frontier target, assume it's physically wedged
# (columns/furniture) and blacklist that target so a different frontier
# gets tried instead. See the STUCK-DETECTION comment in main_node.py for
# the live confirmation that motivated this.
STUCK_CHECK_INTERVAL_SEC = 30.0

# BLACKLIST DECAY -- TRIED AND REVERTED 2026-08-12: a time-based blacklist
# expiry (retry a stuck frontier after 90s instead of excluding it
# permanently) was implemented and tested live against 5 trials of the same
# numerical question. It made accuracy WORSE (1,0,2,1,4 vs the
# permanent-blacklist baseline's 2,1,1,2,3,2, ground truth 2) -- likely
# because retrying a genuinely-unreachable frontier costs a full
# STUCK_CHECK_INTERVAL_SEC before re-blacklisting, and this can repeat,
# burning more budget than it recovers. Reverted; exploration.py is back to
# permanent blacklisting. See the REVERTED comment in exploration.py's
# next_waypoint() for the full reasoning if this is revisited later.
# TUNING NOTE 2026-08-10: raised from 15.0 after confirming live (two
# separate runs, arabic_room) that exploration was consistently exhausting
# ALL reachable frontiers in ~80s / 3 attempts, ending in "No frontiers
# left" instead of using anywhere near the full exploration budget. Some
# blacklisted targets genuinely showed 0.00m movement (truly stuck), but
# others showed real, if slow, progress -- 0.13m, 0.25m, 0.26m in the old
# 15s window -- getting blacklisted anyway under the 0.3m threshold before
# they had a fair chance to actually arrive. Doubling the window gives
# genuine-but-slow navigation through a cluttered room more room to prove
# itself, while a truly wedged robot still shows ~0.00m over the longer
# window too.
STUCK_MOVE_THRESHOLD_M = 0.3

# Reject terrain points farther than this from the robot's current
# position -- /terrain_map_ext is documented as ~20m coverage around the
# vehicle, so anything much beyond that at time of receipt is very likely
# sensor noise, not real reachable terrain. See the GHOST-FRONTIER FIX
# comment in exploration.py.
MAX_TERRAIN_POINT_DIST_FROM_ROBOT_M = 12.0
WAYPOINT_REACHED_TOL_M = 0.75

# ---------------------------------------------------------------------------
# Perception frame gating (stability/memory optimization, 2026-08-11)
# ---------------------------------------------------------------------------
# FRAME_SAMPLE_EVERY_N (above, under Detection / scene graph) already caps
# detection to 2 Hz off the 10 Hz camera. This adds a SECOND, independent
# gate on top: even at that sampled rate, skip a detection pass entirely if
# the robot hasn't moved/turned meaningfully since the last one ran. A
# stationary or slow-turning robot produces near-duplicate frames -- running
# 6-tile Grounding DINO inference on each one anyway burns CPU/RAM for
# detections that would just re-confirm what's already in the scene graph.
# This directly targets the crash risk: each skipped pass is one less
# concurrent set of tile arrays + model activations competing for RAM
# alongside Ollama and the Unity sim. Cheap to compute (two floats from
# odometry, already being tracked) and only fires on the *decision* of
# whether to detect, not in the detection path itself, so it can't affect
# accuracy of a detection that does run.
DETECTION_MIN_MOVE_M = 0.3           # matches STUCK_MOVE_THRESHOLD_M's sense of "meaningful" motion
DETECTION_MIN_YAW_DEG = 10.0

# THREAD-COUNT FIX 2026-08-11: main_node.py previously hardcoded
# MultiThreadedExecutor(num_threads=4). On CPU-bound inference (no working
# CUDA -- see the OLLAMA_MODEL note above) that's real contention: if the
# host has fewer than 4 logical cores available to the container, this
# oversubscribes and each thread gets starved rather than adding
# throughput; if it has more, leaving cores idle wastes available headroom.
# Computed from os.cpu_count() at runtime in main_node.py instead of a
# fixed constant -- this constant is just a safety ceiling so a very
# large/shared host doesn't spin up an excessive number of threads.
MAX_EXECUTOR_THREADS = 12

# ---------------------------------------------------------------------------
# LLM reasoning backend
# ---------------------------------------------------------------------------
# Local LLM via Ollama -- no API key needed, no internet egress needed at
# grading time (the model is baked into the Docker image at build time, see
# ai_module/docker/Dockerfile). Switched away from the Anthropic API on
# 2026-08-06: the competition's submission format only allows changes under
# ai_module/, the top-level docker/compose.yml (owned by evaluators) has no
# mechanism to inject a secret, and the built image must be pushed publicly
# to Docker Hub -- baking a real API key into a public image is both a
# security risk and a personal-billing risk. A small local model removes
# the dependency entirely.
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# MEMORY FIX 2026-08-11: laptop was crashing (not just slow) during live
# runs -- driver 535.309.01 is below the 550+ CUDA needs, so BOTH Grounding
# DINO and Ollama fall back to CPU/RAM instead of GPU memory, and running
# Unity sim + both containers + a 3B-parameter local model concurrently on
# one machine's RAM is enough real memory pressure to trigger an OOM kill
# of the whole box, not just one process. Dropped to the 1B checkpoint --
# still handles this task's JSON-schema parsing fine (short, structured
# outputs, not open-ended reasoning), at roughly a third of the resident
# memory footprint. If you upgrade the driver to 550+ and get real CUDA
# working (nvidia-smi succeeds, torch.cuda.is_available() is True), this is
# safe to revert to "llama3.2:3b" for a bit more parsing robustness -- the
# memory pressure that motivated shrinking it goes away once inference
# actually runs on the GPU instead of competing with everything else for RAM.
OLLAMA_MODEL = "llama3.2:3b"
