# CMU-VLN-Challenge-2026 — ai_module

Replaces the dummy model under `ai_module/src` with two model variants,
kept side by side, both implementing a full explore → perceive → build 3D
scene graph → answer pipeline covering all three question types
(numerical, object_reference, instruction_following), wired to the exact
ROS topics the challenge README lists as accepted inputs/outputs.

## Model variants

| Package | Detector | Status |
|---|---|---|
| `vln_ai_module_dino` | Grounding DINO (`IDEA-Research/grounding-dino-tiny`), open-vocabulary | **Primary — extensively validated live** (see below) |
| `vln_ai_module` | NanoOWL / Owlv2 (`google/owlv2-base-patch16-ensemble`), open-vocabulary | Earlier variant, kept for comparison |

Both packages share the same overall architecture (question parsing →
exploration → scene graph → answer) and differ primarily in the open-
vocabulary object detector used for perception. `vln_ai_module_dino` is
the version that received the majority of live testing, debugging, and
validation described in this README; `vln_ai_module` reflects an earlier
development stage using NanoOWL/Owlv2 instead of Grounding DINO.

To launch a specific variant, use its own launch file:

```bash
# Grounding DINO variant (primary)
ros2 launch vln_ai_module_dino vln_ai_module_dino.launch.py

# NanoOWL/Owlv2 variant
ros2 launch vln_ai_module vln_ai_module.launch.py
```

## Architecture (shared by both variants)

- **Detection**: open-vocabulary object detector (Grounding DINO or
  NanoOWL/Owlv2 depending on variant), run per-tile across the 360°
  equirectangular camera image.
- **Question parsing**: local LLM (Ollama, `llama3.2`) turns the natural-
  language question into a structured spec (target category, attributes,
  relations, path constraints). No API key and no internet egress needed
  at runtime — both the LLM weights and the detector weights are baked
  into the Docker image at build time (see `docker/Dockerfile`). Falls
  back to a rule-based regex parser on any Ollama/API failure so
  navigation never hangs on a network or model issue.
- **Scene graph**: incremental 3D object accumulation with dedup-by-
  proximity, geometric shape sanity-checking against raw label text
  (catches misclassifications like a column labeled "sofa"), and
  confidence-weighted merging across repeat sightings.
- **Exploration**: frontier-based, blind (no language-conditioned
  scoring) — see "Known limitations" below for why this matters.
- **Answering**: shared relation-scoring geometry (`_relation_score` in
  `scene_graph.py`) used identically by `count()` (numerical) and
  `find_unique_referent()` (object_reference), so fixing/improving
  relation semantics (below/above, between, near, superlatives) improves
  both question types at once. `path_planner.py` resolves
  `instruction_following` path constraints into an ordered waypoint
  sequence.

## Setup

Build and run via Docker (recommended — this is how the submission will
actually be evaluated):

```bash
cd ai_module
docker build -f docker/Dockerfile -t vln-ai-module:local .
docker run -it --rm vln-ai-module:local bash
source /opt/ros/jazzy/setup.bash
ros2 launch vln_ai_module_dino vln_ai_module_dino.launch.py
```

Publish a question to test (ROS2 must be running against the same
`ROS_DOMAIN_ID` as the base autonomy stack / Unity sim):

```bash
ros2 topic pub /challenge_question std_msgs/String \
  "data: 'How many chairs are near the window?'" --once
```

The node follows a fresh `WAIT_FOR_QUESTION → EXPLORE → ANSWER → DONE`
state machine per question, matching the challenge's stated evaluation
behavior ("the system will be relaunched for each language command
tested").

## What's validated

Confirmed live against the `arabic_room` Unity training scene (multiple
runs, real ROS2/Docker environment, not just unit tests) — **for the
`vln_ai_module_dino` (Grounding DINO) variant specifically**:

- **Question parsing** — all three question types parse correctly,
  including qualified anchors (e.g. "the stool under the picture"),
  superlatives (e.g. "farthest from the columns"), and `between`/plural
  anchor patterns that previously failed against the LLM's raw output
  (fixed via code-level post-parse repair — see comments in
  `question_parser.py`).
- **Object-reference `between` selection** — verified not just by
  inspection but by independently re-implementing the exact scoring
  geometry outside the codebase and checking it against real scene-graph
  data; confirmed the selection logic picks the geometrically correct
  candidate, including in cases where the correct anchor pair was not the
  visually nearest one.
- **Numerical counting** — functions correctly; one known limitation
  (see below).
- **Detection label-cleaning** — concatenated-label corruption (e.g. two
  adjacent detections merging into one label like `"a stool a table"`)
  and duplicate-article corruption (`"a a table"`) are both detected and
  filtered/repaired before entering the scene graph.

The `vln_ai_module` (NanoOWL/Owlv2) variant has not received the same
level of live validation this session — treat it as an earlier-stage,
less-tested alternative.

## Known limitations (Grounding DINO variant)

- **Instruction-following waypoint execution can stall.** Root-caused
  (via the `autonomy_stack_mecanum_wheel_platform` submodule's actual
  C++ source, not guesswork) to the vendor `localPlanner`: it scores
  candidate directions against real obstacle geometry and only publishes
  a fresh `/path` when some direction scores above zero; if every
  direction is obstacle-blocked from the robot's exact current position,
  no fresh path is published, `pathFollower` runs the stale path down to
  a single point, and hard-zeroes velocity with no built-in recovery.
  `main_node.py`'s `_wait_until_reached` includes a mitigation (creeps
  the target waypoint closer along the line to the goal when stuck, on
  the theory that a nearer target changes the obstacle-scoring geometry)
  but this has NOT been confirmed to reliably resolve the issue in all
  cases — confirmed to still stall in at least one test run even with the
  creep logic active. This is a genuine open risk for
  Instruction-Following scoring if evaluation scenes have similarly
  tight obstacle geometry near likely waypoint locations.
- **Numerical undercounting risk from the single-sighting noise filter**
  (`MIN_OBSERVATIONS_FOR_COUNT` in `scene_graph.py`). This filter exists
  to reject genuine detection noise (e.g. sparse-lidar-fallback
  detections), but can also filter out a real object that only received
  one confident sighting during the 7-minute exploration budget,
  producing an undercount. Confirmed live on one question
  ("how many sofas are below a window") where a real match was excluded
  this way.
- **Exploration coverage is the dominant source of run-to-run
  variance across all three question types.** Frontier exploration is
  blind (not language-conditioned) and the 7-minute budget within the
  10-minute total time limit means the same question can produce
  different scene graphs — and therefore different answers — across
  separate runs, purely from what got explored before the budget ran out.
- **Only tested against one training scene (`arabic_room`)** end-to-end.
  Detection/parsing logic should generalize to the other 14 training
  scenes and the 3 held-out test scenes, but this has not been directly
  confirmed.

## Repo layout

```
ai_module/
├── docker/Dockerfile       # bakes in Ollama+llama3.2, Grounding DINO,
│                           # and Owlv2 weights so the container needs no
│                           # network access at grading time
├── requirements.txt
└── src/
    ├── dummy_vlm/          # original challenge reference implementation,
    │                       # kept as-is for reference/fallback
    ├── vln_ai_module/      # NanoOWL/Owlv2 variant
    │   └── vln_ai_module/
    │       ├── main_node.py
    │       ├── question_parser.py
    │       ├── scene_graph.py
    │       ├── exploration.py
    │       ├── path_planner.py
    │       ├── perception.py
    │       ├── ros_utils.py
    │       └── config.py
    └── vln_ai_module_dino/ # Grounding DINO variant (primary submission)
        └── vln_ai_module_dino/
            ├── main_node.py       # ROS2 node, state machine, answering
            ├── question_parser.py # LLM + rule-based fallback parsing
            ├── scene_graph.py     # object accumulation + relation scoring
            ├── exploration.py     # frontier exploration
            ├── path_planner.py    # instruction_following waypoint sequencing
            ├── perception.py      # Grounding DINO wrapper
            ├── ros_utils.py       # message conversion, 3D localization
            └── config.py          # topic names, thresholds, tuning constants
```
