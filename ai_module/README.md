# CMU-VLN-Challenge-2026 — ai_module scaffold

This replaces the dummy model under `ai_module/src`. It implements a full
explore → perceive → build 3D scene graph → answer pipeline covering all
three question types (numerical, object_reference, instruction_following),
wired to the exact ROS topics listed in the challenge README.

## What's implemented and unit-tested (pure logic, no ROS/GPU needed)

- `scene_graph.py` — object node accumulation with dedup-by-proximity,
  count queries, unique-referent resolution with anchor tie-breaking.
  **Verified**: dedup correctly merges repeat sightings, count/attribute
  filtering and referent tie-breaking behave as expected (see test
  commands below).
- `exploration.py` — frontier detection + clustering + nearest-frontier
  selection from a point cloud. **Verified** returns a sensible frontier
  direction from synthetic terrain data.
- `path_planner.py` — near/avoid/between constraint resolution into an
  ordered waypoint list with avoid-radius nudging. **Verified** on a
  synthetic 3-constraint example.
- `question_parser.py` — regex fallback parser **verified** on sample
  question phrasing; the primary path calls the Anthropic API and needs a
  live key to test (see Setup).

## What's wired but NOT yet validated against the real system

These need your actual Unity sim / sample bag data to test, exactly the
"empirically validate before trusting it" standard you'd apply on the TB4:

1. ~~`ros_utils.equirect_pixel_to_ray` / `localize_detection_3d`~~ **RESOLVED
   2026-08-04**: the real camera→lidar extrinsic was found directly in the
   `autonomy_stack_mecanum_wheel_platform` submodule (`local_planner.launch`
   static_transform_publisher: camera sits `cameraOffsetZ` meters above the
   lidar, no x/y offset), NOT in the Google Drive sample as originally
   assumed. `config.py` now uses the real simulation value (0.1m) with the
   real-robot value (0.25m) available as `CAMERA_TO_SENSOR_TRANSLATION_REAL_ROBOT`.
   Also fixed a sign bug in `ros_utils.py` (`localize_detection_3d` was
   *adding* the offset instead of subtracting it -- harmless at the old
   `(0,0,0)` placeholder, wrong now that it's nonzero).
   Still unverified: whether the README's claim that "images are remapped to
   keep camera frame and lidar frame aligned" really means zero *rotation*
   correction is needed on top of translation -- confirm this against a real
   training-scene run before fully trusting 3D positions (see step 2 below).
2. **`perception.estimate_dominant_color`** — coarse mean-RGB nearest-color
   heuristic. This is the same class of problem you hit with NanoOWL
   door/locker confusion — expect it to need real tuning (or replacement
   with a CLIP-text color+object joint prompt) against actual scene
   lighting before "blue chair" style attributes are reliable.
3. **`DETECTION_SCORE_THRESHOLD`** in `config.py` — starts at 0.15 per your
   own documented NanoOWL learning ("threshold starts at 0.1, per-object
   tuning required"). Needs per-category tuning against your 15 training
   scenes.
4. **PointCloud2 field layout** in `pointcloud2_to_xyz_array` — assumes
   plain `fff` (x,y,z float32) with no extra fields before them. Check
   `msg.fields` on a real message from this system before trusting it;
   if there's a leading `rgb`/`intensity` field the offsets will be wrong.
5. **Exploration timing split** (`EXPLORATION_TIME_BUDGET_SEC` = 7 of the
   10 minutes) is a placeholder guess, not measured against how long your
   detector + scene graph construction actually take per scene.

## Setup

```bash
source /opt/ros/jazzy/setup.bash
pip install -r requirements.txt --break-system-packages
export ANTHROPIC_API_KEY=sk-ant-...
cd ai_module/src
python3 main_node.py
```

## Running the unit-testable logic without ROS or a GPU

```bash
cd ai_module/src
python3 -c "
import numpy as np
from scene_graph import SceneGraph
sg = SceneGraph()
sg.add_or_merge('a chair', 'blue', np.array([1.0,2.0,0.0]), 0.8)
print(sg.summary())
"
```

## Suggested next steps, in order

1. Pull the Real-Robot Data sample bag, inspect a real `/camera/image` and
   `/registered_scan` message to fix items 1 and 4 above.
2. Run `main_node.py` against the docker simulator on a couple of training
   scenes, log the scene graph at answer time (`self.get_logger().info`
   already does this), and manually check it against the ground-truth
   object list for that scene.
3. Fine-tune the color/attribute step against VLA-3D's labeled attributes
   for your training scenes rather than the fixed palette in `config.py`.
4. Only then consider adding language-conditioned frontier scoring
   (VLFM-style) if blind frontier exploration proves too slow/undirected
   within the 10-minute budget.
