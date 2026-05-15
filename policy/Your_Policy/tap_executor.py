"""
TaP (Tool-as-Policy) v1/v1.5/v2 Executor.

Closed-loop executor that issues one tool/action per LLM turn, observes the
result, then decides the next action.  Up to ``MAX_ACTIONS_PER_SUBTASK`` tool
calls per planner subtask.

v1 (default, ``tap_use_vision_input=false``):
  Text-only.  Camera snapshots saved to disk for logging; LLM sees numeric
  robot state and snapshot file paths only.

v1.5 (``tap_use_vision_input=true``):
  After each side-effectful action, the head_camera snapshot is base64-encoded
  and attached to the next LLM user message.  An initial snapshot is captured
  before the first action.  Requires a VL model (e.g. ``qwen3-vl-plus``).

v2 (``tap_use_perception=true``, may combine with vision):
  Every observation auto-attaches ``scene_perception.objects[name]`` (3D pose
  + bbox + principal axis + extent) for every object the LLM has called
  ``resolve_reference()`` on.  Pose is computed from sim segmentation + sim
  depth via PCA-on-point-cloud (real perception math; no actor.get_pose()
  lookup).  Five new tools: ``world_to_pixel``, ``pixel_to_world_point``,
  ``get_depth_at_pixel``, ``rotate_gripper``, ``move_to_pose``.

TaP-vs-cuRobo boundary: TaP only emits decision-level intents (world-frame
targets, image queries).  All IK, trajectory planning, collision avoidance,
and dual-arm synchronisation stay inside cuRobo / TASK_ENV.move(...).

No privileged/expert APIs: no dual_grasp_actor, choose_grasp_pose, or
contact_metadata.  TaP uses only atomic robot actions (relative moves, gripper
open/close, perception queries, rotation).
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Lightweight imports (no SAPIEN/envs dependency) — safe at module level
from primitives.result import SUCCESS, FAILED, make_primitive_result
from primitives.program_validator import FORBIDDEN_SUBSTRINGS
from schema_executor import (
    _is_expert_unreachable_seed,
    _make_expert_unreachable_feedback,
    parse_json_response,
)
from hierarchical_executor import (
    _is_find_action,
    _extract_query,
    _resolve_reference,
    _extract_object_name,
)

# Heavy imports (depend on envs/SAPIEN) — lazy-loaded inside methods to avoid
# import failures in pure-Python test environments.
# Used in: _dispatch_tool, _build_observation, _execute_robot_subtask, _execute_find
#   primitives.perception: get_object_pose, get_gripper_pose, get_gripper_state,
#                          get_camera_snapshot, VALID_CAMERAS
#   primitives.motion: move_delta, move_to_home
#   primitives.gripper: open_gripper, close_gripper
#   primitives.verification: is_task_success
#   privileged_perception: get_scene_objects, get_robot_state

# NOTE: get_contact_metadata is NOT imported — TaP v1 does not use contact points


# ── Constants ────────────────────────────────────────────────────────────────

MAX_ACTIONS_PER_SUBTASK = 20
SNAPSHOT_DIR = Path(__file__).parent / "logs" / "tap_snapshots"
VALID_CAMERAS = ("head_camera", "left_camera", "right_camera")


# ── Tool Registry ────────────────────────────────────────────────────────────

TAP_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # v1 tools — always available
    "get_reference_names":  {"required": [],                            "side_effect": False},
    "resolve_reference":    {"required": ["query"],                     "side_effect": False},
    "get_arm_pose":         {"required": ["arm_tag"],                   "side_effect": False},
    "get_camera_snapshot":  {"required": ["camera_name"],               "side_effect": False},
    "move_by_displacement": {"required": ["arm_tag", "x", "y", "z"],   "side_effect": True},
    "open_gripper":         {"required": ["arm_tag"],                   "side_effect": True},
    "close_gripper":        {"required": ["arm_tag"],                   "side_effect": True},
    "back_to_origin":       {"required": ["arm_tag"],                   "side_effect": True},
    "check_task_success":   {"required": [],                            "side_effect": False},
    # v2 perception tools — only listed in prompt when use_perception=True,
    # but always registered so a stray call still validates structurally.
    "get_depth_at_pixel":   {"required": ["camera_name", "u", "v"],     "side_effect": False},
    "pixel_to_world_point": {"required": ["camera_name", "u", "v", "depth_m"],
                                                                         "side_effect": False},
    "world_to_pixel":       {"required": ["camera_name", "world_xyz"],  "side_effect": False},
    # v2 action tools — fully atomic (rotate-in-place; absolute 7-DoF target)
    "rotate_gripper":       {"required": ["arm_tag", "axis_world", "angle_deg"],
                                                                         "side_effect": True},
    "move_to_pose":         {"required": ["arm_tag", "target_xyz", "target_quat"],
                                                                         "side_effect": True},
}


# ── System Prompt ────────────────────────────────────────────────────────────

TAP_EXECUTOR_SYSTEM = """\
You are a step-by-step robot executor controlling a bimanual robot.

You receive a subtask from the planner. Execute it by calling ONE tool at a time.
After each robot action you will see: tool result, robot arm/gripper state, and a
note that a head camera snapshot was saved. Use the state numbers to decide your
next tool call.

## Available tools

get_reference_names()             -> list of scene object names
resolve_reference(query)          -> fuzzy match query to a scene object name
get_arm_pose(arm_tag)             -> [x, y, z, qw, qx, qy, qz] in world frame
get_camera_snapshot(camera_name)  -> save a camera image (for logging; you see the path only)
move_by_displacement(arm_tag, x, y, z)  -> move arm by world-frame displacement (meters)
open_gripper(arm_tag, pos=1.0)    -> open gripper (pos: 0.0=closed, 1.0=open)
close_gripper(arm_tag, pos=0.0)   -> close gripper (pos: 0.0=closed, 1.0=open)
back_to_origin(arm_tag)           -> return arm to home position
check_task_success()              -> check if the task is complete

## Coordinate frame

World frame: +x = forward, +y = left, +z = up.
Units: meters. Typical arm displacement steps: 0.02-0.10m.
arm_tag: "left" or "right".
camera_name: "head_camera", "left_camera", or "right_camera".

## Output format

Respond with exactly ONE JSON object per turn:
{"thought": "<brief reasoning>", "tool": "<tool_name>", "args": {<tool_args>}}

## Rules

- ONE tool call per response. Never return multiple tools.
- Never write Python code. Never access the raw simulator environment directly.
- No imports, no file I/O, no exec/eval.
- Call check_task_success() when you believe the subtask is done.
- A head_camera snapshot is automatically saved after every robot motion.
  Use get_camera_snapshot for additional cameras or for pre-motion observation.
"""

# Vision extension for v1.5 — appended to the text-only prompt when
# tap_use_vision_input=True.  Conservative: qualitative spatial hints only.
TAP_EXECUTOR_SYSTEM_VISION = TAP_EXECUTOR_SYSTEM + """

## Vision input

After every robot motion you will receive a head_camera RGB image showing the
current scene.  You also receive an initial scene image before your first action.
Use the image to:
- Judge the relative position of the gripper(s) and the target object.
- Verify whether a grasp attempt succeeded (object between fingers).
- Detect if the object has moved, tilted, or fallen.
- Decide the direction and magnitude of the next displacement.

Use the image qualitatively. Prefer small 0.02m moves for fine adjustments.
The image-left/image-right to world-axis mapping is approximate — always verify
by observing the next frame after each move. Combine the image with the numeric
robot_state for precise decisions: trust numeric state for exact coordinates,
trust the image for spatial relationships and grasp quality.

## Thought format (mandatory when vision is on)

Your `thought` MUST begin with a concrete, factual reading of the current
head_camera image in 1–3 sentences before any reasoning.  Cover at minimum:
  1. Where is each gripper RELATIVE to the target object in the image (e.g.
     "left gripper ~5 cm above the pot's left handle; right gripper still at
     home")?
  2. Are the grippers open or closed?  Is anything between the fingers?
  3. Is the target object stable, tilted, moved, or fallen compared to the
     previous frame?
  4. Any obstacle or unexpected scene change?

Then continue with your reasoning ("Therefore I will…") and finally pick the
tool.  Example response:

  {"thought": "Image: left gripper is roughly above the pot's left handle,
  fingers open; right gripper is still at home. Pot upright, unchanged.
  Therefore I will lower the left arm by 0.03 m to align with the handle.",
   "tool": "move_by_displacement",
   "args": {"arm_tag": "left", "x": 0.0, "y": 0.0, "z": -0.03}}

This explicit reading makes your visual reasoning auditable from logs.
"""

# Perception extension for v2 — appended after the base prompt (and vision
# extension if present) when tap_use_perception=True.  Describes
# scene_perception fields, the coordinate-frame helpers, and the rotation
# / absolute-pose action tools.
TAP_EXECUTOR_SYSTEM_PERCEPTION = """

## Scene perception (3D)

After resolve_reference(name) succeeds, every later observation contains
scene_perception.objects[name] with these fields:
  pos_world           [x, y, z]  — center from seg-mask + depth (REAL
                                    perception, not ground-truth lookup).
  bbox_pixels         [u, v, w, h]  — current image bbox.
  visible             bool       — true if any mask pixels were rendered.
  depth_valid_ratio   0.0-1.0    — fraction of mask pixels with valid depth.
                                    Below 0.5 → pos_world unreliable.
  principal_axis_world [x, y, z] — unit vector along the object's longest
                                    axis (from PCA).  Use it to choose
                                    approach direction.
  extent_world        [d1, d2, d3] — extent along the three PCA axes (m);
                                      d1 ≥ d2 ≥ d3.  Compare d3 (thinnest)
                                      with gripper_width_m to see if a grasp
                                      will close on the object.

To start tracking an object's 3D pose, call resolve_reference(query).  The
call returns the resolved name *and* an initial_perception block for that
object; subsequent observations auto-update it.

The initial observation also contains scene_perception.head_camera_viewpoint
describing how the head camera is mounted and which image axis maps to which
world axis.  Read it before planning the first approach.

## Coordinate-frame helpers

World frame: +x = forward, +y = left, +z = up (m).  Map between pixels and
world points by calling:
  world_to_pixel(camera_name, world_xyz)        — project a 3D point to (u, v).
  pixel_to_world_point(camera_name, u, v, depth_m) — unproject a pixel + depth
                                                     to world (x, y, z).
  get_depth_at_pixel(camera_name, u, v)          — sample depth at one pixel.

Do not guess image-to-axis mapping — use the helpers.

## Reorienting / absolute moves

Two new motion tools.  Both run via cuRobo: you choose the target, cuRobo
plans the joint trajectory.

  rotate_gripper(arm_tag, axis_world, angle_deg)
      Rotate the gripper around a world-frame axis by angle_deg, keeping its
      xyz position fixed.  Use when principal_axis_world tells you the
      object's long axis is parallel to your gripper fingers — rotate so the
      fingers close perpendicular to that axis.

  move_to_pose(arm_tag, target_xyz, target_quat)
      Move the gripper to an absolute 7-DoF world target.  target_quat is
      wxyz convention [qw, qx, qy, qz].  Use when you've computed an exact
      pre-grasp pose; cuRobo handles IK and trajectory.

Both tools return a structured result.  On planning failure (collision,
unreachable, etc.) the observation contains status="FAILED" with a reason —
read it and try a smaller angle / different axis / closer target.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _observation(data: dict) -> str:
    """Serialize an observation dict for the planner."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy arrays / non-JSON types to plain Python for serialization."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ── TaPExecutor ──────────────────────────────────────────────────────────────

class TaPExecutor:
    """
    Tool-as-Policy v1/v1.5 executor.

    Runs one tool/action per LLM turn inside a multi-turn conversation.

    v1 (``use_vision=False``): text-only; snapshots saved for logging.
    v1.5 (``use_vision=True``): head_camera image attached to each LLM call.
    """

    def __init__(self, llm_client, logger=None,
                 max_actions: int = MAX_ACTIONS_PER_SUBTASK,
                 use_vision: bool = False,
                 vision_model: str = None,
                 use_perception: bool = False,
                 perception_backend: str = "sim"):
        self.llm = llm_client
        self.logger = logger
        self.max_actions = max(1, int(max_actions))
        self.use_vision = use_vision
        self.vision_model = vision_model
        # v2 perception (default off — preserves v1/v1.5 byte-identical
        # behaviour when False)
        self.use_perception = bool(use_perception)
        self.perception_backend = str(perception_backend or "sim")
        # Lazy-constructed VisionPerception instance (only built when first
        # needed and only when use_perception=True)
        self._perception = None
        # Per-episode set of object names the LLM has called
        # resolve_reference() on.  Auto-populated; cleared at start of each
        # robot subtask.  Only relevant when use_perception=True.
        self._tracked_objects: set = set()

    # ── Perception helpers (v2) ──────────────────────────────────────────

    def _get_perception(self, TASK_ENV):
        """Lazily construct and return the VisionPerception instance.

        Only called when ``self.use_perception=True``; raises if the backend
        cannot be built.  Returns ``None`` if any import/instantiation step
        fails — the executor degrades gracefully (no scene_perception block).
        """
        if self._perception is not None:
            return self._perception
        try:
            from vision_perception import (
                VisionPerception, SimPerceptionBackend,
            )
            if self.perception_backend == "sim":
                backend = SimPerceptionBackend(TASK_ENV)
            else:
                raise NotImplementedError(
                    f"perception_backend={self.perception_backend!r} "
                    "not implemented in Phase 1 (sim only)"
                )
            self._perception = VisionPerception(TASK_ENV, backend=backend)
        except Exception as e:
            # Don't break the executor — log and keep _perception None.
            if self.logger:
                try:
                    self.logger.log_skill(
                        skill_name="tap:_get_perception",
                        args={"backend": self.perception_backend},
                        result=FAILED,
                        feedback=f"perception init failed: {e}",
                        success=False,
                        data={"error": str(e)},
                    )
                except Exception:
                    pass
            self._perception = None
        return self._perception

    def _scene_perception_snapshot(self, TASK_ENV) -> Dict[str, Any]:
        """Return ``{name: pose_dict}`` for every tracked object."""
        if not self.use_perception:
            return {}
        perception = self._get_perception(TASK_ENV)
        if perception is None:
            return {}
        result: Dict[str, Any] = {}
        for name in sorted(self._tracked_objects):
            try:
                result[name] = perception.get_object_pose(name, "head_camera")
            except Exception as e:
                result[name] = {
                    "name": name, "visible": False,
                    "reason": f"perception_error: {e}",
                }
        return result

    # ── Vision helpers ────────────────────────────────────────────────────

    @staticmethod
    def _encode_image_base64(image_path: str) -> Optional[str]:
        """Read a PNG from disk and return a ``data:`` URI, or ``None`` on failure."""
        try:
            p = Path(image_path)
            if not p.exists() or p.stat().st_size == 0:
                return None
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

    def _capture_initial_snapshot(
        self, TASK_ENV, episode_id: int,
    ) -> Optional[str]:
        """Capture an initial head_camera snapshot before any actions.

        Returns the file path on success, ``None`` on failure.
        """
        snap_path = str(
            SNAPSHOT_DIR / f"ep{episode_id:03d}_initial_head_camera.png"
        )
        try:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            TASK_ENV.save_camera_rgb(snap_path, "head_camera")
            return snap_path
        except Exception:
            return None

    # ── Entry point ──────────────────────────────────────────────────────

    def execute(self, action: str, TASK_ENV, planner_history: list) -> Tuple[bool, str]:
        """Execute one planner subtask. Returns (success, planner_observation_json)."""
        if _is_find_action(action):
            return self._execute_find(action, TASK_ENV)
        return self._execute_robot_subtask(action, TASK_ENV, planner_history)

    # ── Find action (deterministic, no LLM) ─────────────────────────────

    def _execute_find(self, action: str, TASK_ENV) -> Tuple[bool, str]:
        from privileged_perception import get_scene_objects
        query = _extract_query(action)
        objects = get_scene_objects(TASK_ENV)
        refs = list(objects.keys())
        selected = _resolve_reference(query, refs)
        if selected is None:
            obs = {
                "type": "object_resolution",
                "status": FAILED,
                "query": query,
                "selected_object": None,
                "candidates": refs,
                "details": f"No scene object matched query {query!r}.",
            }
            return False, _observation(obs)
        obs = {
            "type": "object_resolution",
            "status": SUCCESS,
            "query": query,
            "selected_object": selected,
            "candidates": refs,
            "details": f"Resolved {query!r} to {selected!r}.",
        }
        return True, _observation(obs)

    # ── Robot subtask (multi-turn tool loop) ─────────────────────────────

    def _execute_robot_subtask(
        self, action: str, TASK_ENV, planner_history: list
    ) -> Tuple[bool, str]:
        from privileged_perception import get_scene_objects, get_robot_state

        # Resolve target object
        objects = get_scene_objects(TASK_ENV)
        refs = list(objects.keys())
        selected_object = _extract_object_name(action, refs)

        # v2: reset per-episode tracked-set so each subtask starts cleanly.
        # Only meaningful when self.use_perception=True.
        self._tracked_objects = set()

        # Initial env query: objects + robot state ONLY (no contact_metadata)
        robot_state = get_robot_state(TASK_ENV)
        initial_obs = {
            "subtask": action,
            "selected_object": selected_object,
            "available_objects": refs,
            "robot_state": _sanitize_for_json(robot_state),
        }

        # v2: enrich the initial observation with the camera viewpoint
        # description (one-shot, episode-constant) and a stub objects map
        # (populated as the LLM calls resolve_reference).
        if self.use_perception:
            perception = self._get_perception(TASK_ENV)
            scene_block: Dict[str, Any] = {
                "objects": {},  # filled in by tracked-set as LLM resolves
                "scene_object_names": refs,
            }
            if perception is not None:
                try:
                    scene_block["head_camera_viewpoint"] = (
                        perception.describe_camera_viewpoint("head_camera")
                    )
                except Exception as e:
                    scene_block["head_camera_viewpoint"] = {
                        "camera": "head_camera",
                        "description": f"viewpoint unavailable: {e}",
                    }
            initial_obs["scene_perception"] = scene_block

        # Build conversation
        initial_prompt = self._build_initial_prompt(action, initial_obs, planner_history)
        conversation: List[Dict[str, str]] = [
            {"role": "user", "content": initial_prompt}
        ]

        action_log: List[Dict[str, Any]] = []
        # side_effects_occurred tracking removed — TaP v1 continues on
        # failure and lets the LLM adapt via the closed-loop observation.
        snapshot_counter = 0
        episode_id = int(getattr(TASK_ENV, "ep_num", 0))

        # Ensure snapshot directory exists once
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        # Capture initial head_camera image before any actions (vision mode)
        if self.use_vision:
            latest_snapshot = self._capture_initial_snapshot(TASK_ENV, episode_id)
        else:
            latest_snapshot = None

        for step_idx in range(self.max_actions):
            # ── Record which image THIS LLM call will see ───────────
            image_sent_this_turn = latest_snapshot if self.use_vision else None

            # ── LLM call ─────────────────────────────────────────────
            response, latency, llm_error = self._call_llm(
                conversation, latest_snapshot_path=latest_snapshot,
            )

            if self.logger:
                try:
                    prompt_text = conversation[-1]["content"] if conversation else ""
                    if self.use_vision and self.use_perception:
                        exec_type = "tap_vision_perception"
                    elif self.use_vision:
                        exec_type = "tap_vision"
                    elif self.use_perception:
                        exec_type = "tap_perception"
                    else:
                        exec_type = "tap"
                    self.logger.log_executor_call(
                        prompt=prompt_text,
                        code=response,
                        latency_s=latency,
                        model=(self.vision_model
                               if (self.use_vision and self.vision_model)
                               else self.llm.model),
                        executor_type=exec_type,
                    )
                except Exception:
                    pass

            if llm_error:
                action_log.append({
                    "step": step_idx, "error": "llm_error",
                    "details": llm_error,
                })
                # Feed error back as observation so loop can retry
                conversation.append({"role": "assistant", "content": response or "(empty)"})
                conversation.append({"role": "user", "content": json.dumps({
                    "type": "error", "error": f"LLM call failed: {llm_error}",
                })})
                continue

            # ── Parse tool call ──────────────────────────────────────
            tool_call, parse_error = self._parse_tool_call(response)
            if parse_error:
                action_log.append({
                    "step": step_idx, "error": "parse_error",
                    "details": parse_error,
                })
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": json.dumps({
                    "type": "parse_error",
                    "error": parse_error,
                    "hint": "Respond with exactly one JSON: {\"thought\": ..., \"tool\": ..., \"args\": {...}}",
                })})
                continue

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            # ── Validate tool call ───────────────────────────────────
            validation_error = self._validate_tool_call(tool_name, tool_args)
            if validation_error:
                action_log.append({
                    "step": step_idx, "tool": tool_name,
                    "error": "validation_error",
                    "details": validation_error,
                })
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": json.dumps({
                    "type": "validation_error",
                    "tool": tool_name,
                    "error": validation_error,
                })})
                continue

            # ── Dispatch tool ────────────────────────────────────────
            tool_result = self._dispatch_tool(
                tool_name, tool_args, TASK_ENV,
                episode_id=episode_id,
                snapshot_counter=snapshot_counter,
            )

            is_side_effect = TAP_TOOL_REGISTRY[tool_name]["side_effect"]
            result_status = tool_result.get("status", FAILED)
            # ── Build observation ────────────────────────────────────
            obs = self._build_observation(
                tool_name, tool_args, tool_result, TASK_ENV,
                capture_camera=is_side_effect,
                episode_id=episode_id,
                snapshot_counter=snapshot_counter,
            )
            if is_side_effect and result_status == SUCCESS:
                snapshot_counter += 1

            # Update latest_snapshot for the NEXT LLM call
            snap_path = obs.get("snapshot_saved")
            if isinstance(snap_path, str) and not snap_path.startswith("error:"):
                latest_snapshot = snap_path
            # If no new snapshot (non-side-effect tool or failed capture),
            # keep previous latest_snapshot so the next LLM call still has it.

            action_log.append({
                "step": step_idx,
                "tool": tool_name,
                "args": _sanitize_for_json(tool_args),
                "result_status": result_status,
                "side_effect": is_side_effect,
            })

            # Record vision metadata in action_log
            if self.use_vision:
                action_log[-1]["vision_enabled"] = True
                action_log[-1]["vision_model"] = self.vision_model
                action_log[-1]["image_sent_this_turn"] = image_sent_this_turn
                snap_after = obs.get("snapshot_saved")
                if isinstance(snap_after, str) and not snap_after.startswith("error:"):
                    action_log[-1]["snapshot_saved_after_action"] = snap_after

            # Record perception metadata in action_log
            if self.use_perception:
                action_log[-1]["perception_enabled"] = True
                action_log[-1]["perception_backend"] = self.perception_backend
                action_log[-1]["tracked_objects"] = sorted(self._tracked_objects)
                scene = obs.get("scene_perception")
                if isinstance(scene, dict):
                    # Only persist the object names + key fields (no base64,
                    # no image data; everything in scene is already small JSON)
                    action_log[-1]["scene_perception_snapshot"] = scene

            # Log tool dispatch via logger.log_skill — with vision+perception metadata
            if self.logger:
                try:
                    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
                    log_data = _sanitize_for_json(tool_result.get("data")) or {}
                    if self.use_vision or self.use_perception:
                        log_data = dict(log_data) if isinstance(log_data, dict) else {}
                    if self.use_vision:
                        log_data["vision_enabled"] = True
                        log_data["vision_model"] = self.vision_model
                        log_data["image_sent_this_turn"] = image_sent_this_turn
                        snap_after = obs.get("snapshot_saved")
                        if isinstance(snap_after, str) and not snap_after.startswith("error:"):
                            log_data["snapshot_saved_after_action"] = snap_after
                    if self.use_perception:
                        log_data["perception_enabled"] = True
                        log_data["perception_backend"] = self.perception_backend
                        log_data["tracked_objects"] = sorted(self._tracked_objects)
                        scene = obs.get("scene_perception")
                        if isinstance(scene, dict):
                            log_data["scene_perception"] = scene
                    self.logger.log_skill(
                        skill_name=f"tap:{tool_name}",
                        args=_sanitize_for_json(tool_args),
                        result=result_status,
                        feedback=tool_result.get("details", ""),
                        step_num=sim_step,
                        success=(result_status == SUCCESS),
                        data=log_data,
                    )
                except Exception:
                    pass

            # ── Exit conditions ──────────────────────────────────────

            # Task success
            if tool_name == "check_task_success":
                env_success = False
                data = tool_result.get("data", {})
                if isinstance(data, dict):
                    env_success = bool(data.get("env_success", False))
                if env_success:
                    return True, self._build_planner_summary(
                        True, action_log, step_idx + 1,
                    )

            # Expert unreachable seed
            if _is_expert_unreachable_seed(tool_result):
                unreachable_json = _make_expert_unreachable_feedback(tool_result)
                unreachable_obs = json.loads(unreachable_json)
                unreachable_obs.update({
                    "type": "robot_subtask_result",
                    "subtask": action,
                    "selected_object": selected_object,
                    "executor_mode": "tap",
                    "failure_type": "environment_feasibility_error",
                })
                return False, _observation(unreachable_obs)

            # NOTE: We intentionally do NOT abort on failure after side
            # effects.  Motion-planning failures happen *before* env mutation,
            # so the state is still valid.  Even if an action partially
            # executed, the next observation includes fresh robot state so the
            # LLM can adapt.  The loop continues until max_actions or success.

            # ── Feed observation into conversation ───────────────────
            conversation.append({"role": "assistant", "content": response})
            conversation.append({"role": "user", "content": json.dumps(
                _sanitize_for_json(obs), ensure_ascii=False, default=str,
            )})

        # Max actions exceeded
        return False, self._build_planner_summary(
            False, action_log, self.max_actions,
            reason="max_actions_exceeded",
        )

    # ── Initial Prompt ───────────────────────────────────────────────────

    def _build_initial_prompt(
        self, action: str, initial_obs: dict, planner_history: list,
    ) -> str:
        history_brief = []
        for h in planner_history[-3:]:  # Last 3 planner turns only
            history_brief.append({
                "planner_action": h.get("action"),
                "result": h.get("observation", "")[:200],
            })
        payload = {
            "subtask": action,
            "initial_environment": initial_obs,
            "planner_history_recent": history_brief,
            "instructions": (
                "Execute this subtask step by step. "
                "Call tools one at a time. "
                "Observe the result after each action. "
                "Call check_task_success() when you believe the subtask is complete."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    # ── LLM Call (multi-turn) ────────────────────────────────────────────

    def _call_llm(
        self, conversation: list, latest_snapshot_path: str = None,
    ) -> Tuple[str, float, Optional[str]]:
        """Call the LLM with full multi-turn conversation history.

        When vision is enabled and *latest_snapshot_path* is provided, the
        last user message is transformed into a multimodal content block with
        both the text observation and the head_camera image.  The original
        *conversation* list is never mutated.
        """
        started = time.time()
        try:
            base_prompt = (
                TAP_EXECUTOR_SYSTEM_VISION if self.use_vision
                else TAP_EXECUTOR_SYSTEM
            )
            system_prompt = (
                base_prompt + TAP_EXECUTOR_SYSTEM_PERCEPTION
                if self.use_perception
                else base_prompt
            )
            model_name = (
                self.vision_model
                if (self.use_vision and self.vision_model)
                else self.llm.model
            )

            # Deep-copy each message dict so caller's conversation is safe
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in conversation
            ]
            messages = [
                {"role": "system", "content": system_prompt},
            ] + messages

            # Attach image to last user message if vision is active
            if self.use_vision and latest_snapshot_path:
                data_uri = self._encode_image_base64(latest_snapshot_path)
                if data_uri and messages and messages[-1]["role"] == "user":
                    text_content = messages[-1]["content"]
                    messages[-1] = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_content},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                elif messages and messages[-1]["role"] == "user":
                    # Encode failed — inject a note so debug is clear
                    text_content = messages[-1]["content"]
                    messages[-1]["content"] = (
                        text_content
                        + '\n{"vision_image_attach_error": "failed_to_encode_image"}'
                    )

            resp = self.llm.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.llm.default_temperature,
                max_tokens=self.llm.default_max_tokens,
            )
            text = resp.choices[0].message.content.strip()
            return text, time.time() - started, None
        except Exception as e:
            return "", time.time() - started, str(e)

    # ── Parse Tool Call ──────────────────────────────────────────────────

    @staticmethod
    def _parse_tool_call(response: str) -> Tuple[Optional[dict], Optional[str]]:
        """Extract one {thought, tool, args} from LLM response."""
        try:
            parsed = parse_json_response(response)
        except Exception as e:
            return None, f"JSON parse failed: {e}"

        if not isinstance(parsed, dict):
            return None, f"Expected JSON object, got {type(parsed).__name__}"

        tool = parsed.get("tool")
        if not tool or not isinstance(tool, str):
            return None, "Missing or invalid 'tool' key in response"

        args = parsed.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None, f"'args' must be a dict, got {type(args).__name__}"

        return {
            "thought": parsed.get("thought", ""),
            "tool": tool.strip(),
            "args": args,
        }, None

    # ── Validate Tool Call ───────────────────────────────────────────────

    @staticmethod
    def _validate_tool_call(tool_name: str, tool_args: dict) -> Optional[str]:
        """Return error string if invalid, None if OK."""
        # Check tool name
        if tool_name not in TAP_TOOL_REGISTRY:
            return (
                f"Unknown tool {tool_name!r}. "
                f"Available: {list(TAP_TOOL_REGISTRY.keys())}"
            )

        spec = TAP_TOOL_REGISTRY[tool_name]

        # Check required args
        for req in spec["required"]:
            if req not in tool_args:
                return f"Tool {tool_name!r} requires arg {req!r}"

        # Check forbidden substrings in string values
        for key, val in tool_args.items():
            if isinstance(val, str):
                for tok in FORBIDDEN_SUBSTRINGS:
                    if tok in val:
                        return f"Forbidden substring {tok!r} found in arg {key!r}"

        # Validate arm_tag
        if "arm_tag" in tool_args:
            if tool_args["arm_tag"] not in ("left", "right"):
                return f"arm_tag must be 'left' or 'right', got {tool_args['arm_tag']!r}"

        # Validate camera_name
        if "camera_name" in tool_args:
            if tool_args["camera_name"] not in VALID_CAMERAS:
                return (
                    f"camera_name must be one of {VALID_CAMERAS}, "
                    f"got {tool_args['camera_name']!r}"
                )

        # Validate v2 list-typed args (world_xyz, axis_world, target_xyz must
        # be length-3; target_quat must be length-4)
        for key, expected_len in (
            ("world_xyz", 3),
            ("axis_world", 3),
            ("target_xyz", 3),
            ("target_quat", 4),
        ):
            if key in tool_args:
                val = tool_args[key]
                if not isinstance(val, (list, tuple)):
                    return f"{key!r} must be a list, got {type(val).__name__}"
                if len(val) != expected_len:
                    return f"{key!r} must have length {expected_len}, got {len(val)}"
                for i, x in enumerate(val):
                    if not isinstance(x, (int, float)):
                        return (
                            f"{key!r}[{i}] must be a number, got "
                            f"{type(x).__name__}"
                        )

        # Validate u, v as ints (perception pixel queries)
        for key in ("u", "v"):
            if key in tool_args:
                val = tool_args[key]
                if not isinstance(val, (int, float)):
                    return f"{key!r} must be a number, got {type(val).__name__}"

        return None

    # ── Tool Dispatch ────────────────────────────────────────────────────

    def _dispatch_tool(
        self, tool_name: str, tool_args: dict, TASK_ENV,
        episode_id: int = 0, snapshot_counter: int = 0,
    ) -> dict:
        """Execute a single tool call and return a PrimitiveResult dict."""
        from privileged_perception import get_scene_objects
        from primitives.perception import get_gripper_pose, get_camera_snapshot
        from primitives import motion as _motion
        from primitives import gripper as _gripper
        from primitives.verification import is_task_success

        try:
            if tool_name == "get_reference_names":
                names = list(get_scene_objects(TASK_ENV).keys())
                return make_primitive_result(
                    "get_reference_names", SUCCESS,
                    f"Found {len(names)} objects.",
                    names=names,
                )

            if tool_name == "resolve_reference":
                refs = list(get_scene_objects(TASK_ENV).keys())
                query = str(tool_args.get("query", ""))
                resolved = _resolve_reference(query, refs)
                if resolved is None:
                    return make_primitive_result(
                        "resolve_reference", FAILED,
                        f"No match for {query!r}. Available: {refs}",
                        query=query, resolved=None, candidates=refs,
                    )
                # v2: enrol the resolved name in tracking + bundle initial
                # perception so the LLM gets 3D info in the same turn (no
                # wasted "perception-only" call).
                extra: Dict[str, Any] = {}
                if self.use_perception:
                    self._tracked_objects.add(resolved)
                    perception = self._get_perception(TASK_ENV)
                    if perception is not None:
                        try:
                            extra["initial_perception"] = (
                                perception.get_object_pose(resolved, "head_camera")
                            )
                        except Exception as e:
                            extra["initial_perception"] = {
                                "name": resolved, "visible": False,
                                "reason": f"perception_error: {e}",
                            }
                return make_primitive_result(
                    "resolve_reference", SUCCESS,
                    f"Resolved {query!r} to {resolved!r}.",
                    query=query, resolved=resolved, **extra,
                )

            if tool_name == "get_arm_pose":
                arm = str(tool_args["arm_tag"])
                return get_gripper_pose(TASK_ENV, arm)

            if tool_name == "get_camera_snapshot":
                cam = str(tool_args["camera_name"])
                path = str(
                    SNAPSHOT_DIR
                    / f"ep{episode_id:03d}_action{snapshot_counter:03d}_{cam}.png"
                )
                return get_camera_snapshot(TASK_ENV, cam, path)

            if tool_name == "move_by_displacement":
                arm = str(tool_args["arm_tag"])
                x = float(tool_args["x"])
                y = float(tool_args["y"])
                z = float(tool_args["z"])
                return _motion.move_delta(TASK_ENV, arm, dx=x, dy=y, dz=z)

            if tool_name == "open_gripper":
                arm = str(tool_args["arm_tag"])
                pos = float(tool_args.get("pos", 1.0))
                return _gripper.open_gripper(TASK_ENV, arm, pos=pos)

            if tool_name == "close_gripper":
                arm = str(tool_args["arm_tag"])
                pos = float(tool_args.get("pos", 0.0))
                return _gripper.close_gripper(TASK_ENV, arm, pos=pos)

            if tool_name == "back_to_origin":
                arm = str(tool_args["arm_tag"])
                return _motion.move_to_home(TASK_ENV, arm)

            if tool_name == "check_task_success":
                return is_task_success(TASK_ENV)

            # ── v2 perception helper tools ────────────────────────────
            if tool_name == "get_depth_at_pixel":
                if not self.use_perception:
                    return make_primitive_result(
                        "get_depth_at_pixel", FAILED,
                        "get_depth_at_pixel requires tap_use_perception=true.",
                    )
                perception = self._get_perception(TASK_ENV)
                if perception is None:
                    return make_primitive_result(
                        "get_depth_at_pixel", FAILED,
                        "VisionPerception unavailable.",
                    )
                cam = str(tool_args["camera_name"])
                u = int(tool_args["u"]); v = int(tool_args["v"])
                data = perception.get_depth_at_pixel(u, v, cam)
                return make_primitive_result(
                    "get_depth_at_pixel",
                    SUCCESS if data.get("valid") else FAILED,
                    f"depth at ({u},{v}) on {cam}: {data.get('depth_m')}",
                    camera_name=cam, u=u, v=v, **data,
                )

            if tool_name == "pixel_to_world_point":
                if not self.use_perception:
                    return make_primitive_result(
                        "pixel_to_world_point", FAILED,
                        "pixel_to_world_point requires tap_use_perception=true.",
                    )
                perception = self._get_perception(TASK_ENV)
                if perception is None:
                    return make_primitive_result(
                        "pixel_to_world_point", FAILED,
                        "VisionPerception unavailable.",
                    )
                cam = str(tool_args["camera_name"])
                u = int(tool_args["u"]); v = int(tool_args["v"])
                d = float(tool_args["depth_m"])
                data = perception.pixel_to_world_point(u, v, d, cam)
                ok = data.get("world_xyz") is not None
                return make_primitive_result(
                    "pixel_to_world_point",
                    SUCCESS if ok else FAILED,
                    f"({u},{v},{d:.3f}m) → {data.get('world_xyz')}",
                    camera_name=cam, u=u, v=v, depth_m=d, **data,
                )

            if tool_name == "world_to_pixel":
                if not self.use_perception:
                    return make_primitive_result(
                        "world_to_pixel", FAILED,
                        "world_to_pixel requires tap_use_perception=true.",
                    )
                perception = self._get_perception(TASK_ENV)
                if perception is None:
                    return make_primitive_result(
                        "world_to_pixel", FAILED,
                        "VisionPerception unavailable.",
                    )
                cam = str(tool_args["camera_name"])
                xyz = tool_args["world_xyz"]
                if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                    return make_primitive_result(
                        "world_to_pixel", FAILED,
                        f"world_xyz must be a length-3 list, got {xyz!r}",
                    )
                data = perception.world_to_pixel([float(v) for v in xyz], cam)
                ok = data.get("pixel_uv") is not None
                return make_primitive_result(
                    "world_to_pixel",
                    SUCCESS if ok else FAILED,
                    f"world{tuple(xyz)} → pixel {data.get('pixel_uv')}",
                    camera_name=cam, world_xyz=list(xyz), **data,
                )

            # ── v2 action tools ──────────────────────────────────────
            if tool_name == "rotate_gripper":
                arm = str(tool_args["arm_tag"])
                axis = tool_args["axis_world"]
                if not (isinstance(axis, (list, tuple)) and len(axis) == 3):
                    return make_primitive_result(
                        "rotate_gripper", FAILED,
                        f"axis_world must be a length-3 list, got {axis!r}",
                    )
                angle = float(tool_args["angle_deg"])
                return _motion.rotate_delta(
                    TASK_ENV, arm,
                    [float(v) for v in axis], angle,
                )

            if tool_name == "move_to_pose":
                arm = str(tool_args["arm_tag"])
                xyz = tool_args["target_xyz"]
                quat = tool_args["target_quat"]
                if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                    return make_primitive_result(
                        "move_to_pose", FAILED,
                        f"target_xyz must be a length-3 list, got {xyz!r}",
                    )
                if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
                    return make_primitive_result(
                        "move_to_pose", FAILED,
                        f"target_quat must be a length-4 list (wxyz), got {quat!r}",
                    )
                target = [float(v) for v in xyz] + [float(v) for v in quat]
                return _motion.move_to_pose(TASK_ENV, arm, target)

            return make_primitive_result(
                tool_name, FAILED, f"Unhandled tool {tool_name!r}",
            )

        except Exception as e:
            return make_primitive_result(
                tool_name, FAILED, f"Dispatch error: {e}",
            )

    # ── Observation Builder ──────────────────────────────────────────────

    def _build_observation(
        self, tool_name: str, tool_args: dict, tool_result: dict,
        TASK_ENV, capture_camera: bool,
        episode_id: int = 0, snapshot_counter: int = 0,
    ) -> dict:
        """Build the observation dict that the LLM sees as the next user turn."""
        from privileged_perception import get_robot_state

        obs: Dict[str, Any] = {
            "type": "tool_result",
            "tool": tool_name,
            "status": tool_result.get("status"),
            "details": tool_result.get("details"),
            "data": tool_result.get("data", {}),
        }

        if capture_camera:
            # Auto-capture head_camera snapshot
            snap_path = str(
                SNAPSHOT_DIR
                / f"ep{episode_id:03d}_action{snapshot_counter:03d}_head_camera.png"
            )
            try:
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                TASK_ENV.save_camera_rgb(snap_path, "head_camera")
                obs["snapshot_saved"] = snap_path
            except Exception as e:
                obs["snapshot_saved"] = f"error: {e}"

            # Fresh robot state after action
            obs["robot_state"] = _sanitize_for_json(get_robot_state(TASK_ENV))

        # v2 perception: auto-attach scene_perception for every tracked
        # object on EVERY observation (not just side-effect ones — read-only
        # tool calls can still want fresh 3D info).
        if self.use_perception and self._tracked_objects:
            try:
                scene_objects = self._scene_perception_snapshot(TASK_ENV)
            except Exception as e:
                scene_objects = {"_error": f"perception_snapshot_failed: {e}"}
            obs["scene_perception"] = {
                "objects": _sanitize_for_json(scene_objects),
            }

        return obs

    # ── Planner Summary ──────────────────────────────────────────────────

    @staticmethod
    def _build_planner_summary(
        success: bool, action_log: list,
        steps_taken: int, reason: str = None,
    ) -> str:
        """Build the structured observation the planner sees. No raw env data."""
        return json.dumps({
            "type": "robot_subtask_result",
            "status": SUCCESS if success else FAILED,
            "executor_mode": "tap",
            "steps_taken": steps_taken,
            "max_steps": MAX_ACTIONS_PER_SUBTASK,
            "reason": reason,
            "tool_sequence": [
                {"tool": a["tool"], "status": a.get("result_status")}
                for a in action_log if "tool" in a
            ],
        }, ensure_ascii=False, default=str)
