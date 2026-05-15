"""
Pure-Python tests for TaP (Tool-as-Policy) v1 executor.

Tests cover: JSON parsing, tool validation, planner summary isolation,
deterministic Find actions, system prompt safety, tool registry completeness,
and initial env query contents.

Run:
    cd ~/Documents/GitHub/RoboTwin
    python policy/Your_Policy/examples/tap_executor_tests.py
"""

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICY_DIR = _HERE.parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

import tap_executor as te  # noqa: E402
from tap_executor import (  # noqa: E402
    TaPExecutor,
    TAP_EXECUTOR_SYSTEM,
    TAP_TOOL_REGISTRY,
    VALID_CAMERAS,
)

_RESULTS = []


def _record(name, ok, details=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}")
    if details:
        print(f"      {details}")
    _RESULTS.append((name, ok))
    return ok


# ── Parse tool call tests ────────────────────────────────────────────────────

def case_parse_tool_call_valid():
    """Valid JSON with thought/tool/args parses correctly."""
    response = json.dumps({
        "thought": "Move left arm up",
        "tool": "move_by_displacement",
        "args": {"arm_tag": "left", "x": 0.0, "y": 0.0, "z": 0.1},
    })
    tc, err = TaPExecutor._parse_tool_call(response)
    ok = (err is None
          and tc["tool"] == "move_by_displacement"
          and tc["args"]["arm_tag"] == "left"
          and tc["args"]["z"] == 0.1)
    return _record("parse_tool_call_valid", ok, err)


def case_parse_tool_call_with_fences():
    """JSON wrapped in ```json fences still parses."""
    response = '```json\n{"thought": "check", "tool": "check_task_success", "args": {}}\n```'
    tc, err = TaPExecutor._parse_tool_call(response)
    ok = err is None and tc["tool"] == "check_task_success"
    return _record("parse_tool_call_with_fences", ok, err)


def case_parse_rejects_missing_tool():
    """JSON without 'tool' key -> error."""
    response = json.dumps({"thought": "hmm", "action": "move"})
    tc, err = TaPExecutor._parse_tool_call(response)
    ok = tc is None and err is not None and "tool" in err.lower()
    return _record("parse_rejects_missing_tool", ok, err)


def case_parse_rejects_non_dict_args():
    """args must be a dict."""
    response = json.dumps({"thought": "x", "tool": "get_arm_pose", "args": ["left"]})
    tc, err = TaPExecutor._parse_tool_call(response)
    ok = tc is None and err is not None and "dict" in err
    return _record("parse_rejects_non_dict_args", ok, err)


# ── Validation tests ─────────────────────────────────────────────────────────

def case_validate_rejects_unknown_tool():
    """Tool not in registry -> error."""
    err = TaPExecutor._validate_tool_call("dual_grasp_actor", {"object": "pot"})
    ok = err is not None and "Unknown" in err
    return _record("validate_rejects_unknown_tool", ok, err)


def case_validate_rejects_forbidden_substring():
    """TASK_ENV in string args -> error."""
    err = TaPExecutor._validate_tool_call("resolve_reference", {"query": "TASK_ENV.pot"})
    ok = err is not None and "Forbidden" in err
    return _record("validate_rejects_forbidden_substring", ok, err)


def case_validate_arm_tag():
    """arm_tag must be 'left' or 'right'."""
    err = TaPExecutor._validate_tool_call("get_arm_pose", {"arm_tag": "both"})
    ok = err is not None and "left" in err and "right" in err
    return _record("validate_arm_tag", ok, err)


def case_validate_arm_tag_valid():
    """Valid arm_tag passes."""
    err = TaPExecutor._validate_tool_call("get_arm_pose", {"arm_tag": "left"})
    ok = err is None
    return _record("validate_arm_tag_valid", ok, err)


def case_validate_camera_name():
    """Invalid camera name -> error."""
    err = TaPExecutor._validate_tool_call("get_camera_snapshot", {"camera_name": "rear_camera"})
    ok = err is not None and "camera_name" in err
    return _record("validate_camera_name", ok, err)


def case_validate_missing_required_arg():
    """Missing required arg -> error."""
    err = TaPExecutor._validate_tool_call("move_by_displacement", {"arm_tag": "left", "x": 0.0})
    ok = err is not None and "requires" in err
    return _record("validate_missing_required_arg", ok, err)


def case_validate_check_task_success_no_args():
    """check_task_success requires no args -> OK."""
    err = TaPExecutor._validate_tool_call("check_task_success", {})
    ok = err is None
    return _record("validate_check_task_success_no_args", ok, err)


# ── Planner summary tests ───────────────────────────────────────────────────

def case_planner_summary_no_raw_env():
    """Planner summary contains no file paths, endpose, gripper_val."""
    log = [
        {"step": 0, "tool": "get_arm_pose", "result_status": "SUCCESS", "side_effect": False},
        {"step": 1, "tool": "move_by_displacement", "result_status": "SUCCESS", "side_effect": True},
        {"step": 2, "tool": "check_task_success", "result_status": "SUCCESS", "side_effect": False},
    ]
    summary = TaPExecutor._build_planner_summary(True, log, 3)
    forbidden = ["endpose", "gripper_val", ".png", "tap_snapshots", "position", "orientation"]
    ok = all(tok not in summary for tok in forbidden)
    parsed = json.loads(summary)
    ok = ok and parsed["status"] == "SUCCESS" and parsed["steps_taken"] == 3
    ok = ok and len(parsed["tool_sequence"]) == 3
    return _record("planner_summary_no_raw_env", ok, summary[:300])


# ── System prompt tests ──────────────────────────────────────────────────────

def case_system_prompt_no_raw_env_api():
    """System prompt does not expose raw simulator API names."""
    # These are raw env internals that should never appear in the TaP prompt
    forbidden = [
        "TASK_ENV",
        "grasp_actor",
        "choose_grasp_pose",
        "contact_matrix",
        "contact_point",
        "cuRobo",
        "motion_planner",
    ]
    violations = [tok for tok in forbidden if tok in TAP_EXECUTOR_SYSTEM]
    ok = len(violations) == 0
    return _record("system_prompt_no_raw_env_api", ok, f"violations: {violations}")


# ── Tool registry tests ─────────────────────────────────────────────────────

def case_tool_registry_complete():
    """All 14 tools (9 v1 + 5 v2) have 'required' and 'side_effect' keys."""
    expected_tools = {
        # v1
        "get_reference_names", "resolve_reference", "get_arm_pose",
        "get_camera_snapshot", "move_by_displacement", "open_gripper",
        "close_gripper", "back_to_origin", "check_task_success",
        # v2 perception
        "get_depth_at_pixel", "pixel_to_world_point", "world_to_pixel",
        # v2 action
        "rotate_gripper", "move_to_pose",
    }
    ok = set(TAP_TOOL_REGISTRY.keys()) == expected_tools
    for name, spec in TAP_TOOL_REGISTRY.items():
        if "required" not in spec or "side_effect" not in spec:
            ok = False
    return _record("tool_registry_complete", ok,
                    f"registered: {sorted(TAP_TOOL_REGISTRY.keys())}")


# ── Find action tests ───────────────────────────────────────────────────────

def case_find_action_deterministic():
    """Find action resolves without LLM using deterministic fuzzy match."""
    # _execute_find imports get_scene_objects from privileged_perception lazily.
    # We mock it at the module level that _execute_find will import from.
    import privileged_perception as pp
    original = pp.get_scene_objects
    pp.get_scene_objects = lambda _env: {
        "060_kitchenpot": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]},
    }
    try:
        class _FakeLLM:
            model = "fake"
            client = None
            default_temperature = 0.1
            default_max_tokens = 1024
        executor = TaPExecutor(_FakeLLM())
        success, feedback = executor._execute_find(
            "Find target object for: pot", object(),
        )
        obs = json.loads(feedback)
        ok = success and obs["selected_object"] == "060_kitchenpot"
        return _record("find_action_deterministic", ok, feedback)
    finally:
        pp.get_scene_objects = original


# ── Initial env query tests ──────────────────────────────────────────────────

def case_initial_env_query_no_contact():
    """The initial prompt should NOT contain contact_metadata."""
    class _FakeLLM:
        model = "fake"
        client = None
        default_temperature = 0.1
        default_max_tokens = 1024
    executor = TaPExecutor(_FakeLLM())
    initial_obs = {
        "subtask": "test",
        "selected_object": "pot",
        "available_objects": ["pot"],
        "robot_state": {"left_arm": {}, "right_arm": {}},
    }
    prompt = executor._build_initial_prompt("test subtask", initial_obs, [])
    forbidden = ["contact_metadata", "contact_point", "contact_matrix", "side_hint"]
    violations = [tok for tok in forbidden if tok in prompt]
    ok = len(violations) == 0
    return _record("initial_env_query_no_contact", ok,
                    f"violations: {violations}, prompt: {prompt[:300]}")


# ── Vision tests (v1.5) ─────────────────────────────────────────────────────

def case_vision_system_prompt_extends_text():
    """Vision system prompt extends the text-only prompt with image guidance."""
    from tap_executor import TAP_EXECUTOR_SYSTEM_VISION
    ok = (TAP_EXECUTOR_SYSTEM in TAP_EXECUTOR_SYSTEM_VISION
          and "head_camera" in TAP_EXECUTOR_SYSTEM_VISION
          and "image" in TAP_EXECUTOR_SYSTEM_VISION.lower()
          and len(TAP_EXECUTOR_SYSTEM_VISION) > len(TAP_EXECUTOR_SYSTEM))
    return _record("vision_system_prompt_extends_text", ok)


def case_encode_image_nonexistent_returns_none():
    """_encode_image_base64 returns None for missing file."""
    result = TaPExecutor._encode_image_base64("/tmp/_nonexistent_12345.png")
    ok = result is None
    return _record("encode_image_nonexistent_returns_none", ok)


def case_encode_image_valid_file():
    """_encode_image_base64 returns a data URI for a valid PNG."""
    import base64
    import os
    import tempfile
    # Pre-computed valid 1x1 red PNG
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "2mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )
    png_bytes = base64.b64decode(png_b64)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_bytes)
        tmp_path = f.name
    try:
        data_uri = TaPExecutor._encode_image_base64(tmp_path)
        ok = (data_uri is not None
              and data_uri.startswith("data:image/png;base64,")
              and len(data_uri) > 30)
        return _record("encode_image_valid_file", ok,
                        f"URI prefix ok, length={len(data_uri) if data_uri else 0}")
    finally:
        os.unlink(tmp_path)


def case_vision_flag_defaults_off():
    """TaPExecutor defaults to vision disabled."""
    class _FakeLLM:
        model = "fake"
        client = None
        default_temperature = 0.1
        default_max_tokens = 1024
    executor = TaPExecutor(_FakeLLM())
    ok = executor.use_vision is False and executor.vision_model is None
    return _record("vision_flag_defaults_off", ok)


def case_vision_flag_on():
    """TaPExecutor accepts use_vision=True and vision_model."""
    class _FakeLLM:
        model = "fake"
        client = None
        default_temperature = 0.1
        default_max_tokens = 1024
    executor = TaPExecutor(_FakeLLM(), use_vision=True, vision_model="qwen3-vl-plus")
    ok = executor.use_vision is True and executor.vision_model == "qwen3-vl-plus"
    return _record("vision_flag_on", ok)


# ── Perception tests (v2) ───────────────────────────────────────────────────

class _FakeLLM:
    model = "fake"
    client = None
    default_temperature = 0.1
    default_max_tokens = 1024


class _FakeBackend:
    """Minimal PerceptionBackend stub for VisionPerception math tests."""
    def __init__(self, mask, depth, K, Rt, cam2world, names=None):
        self.mask = mask
        self.depth = depth
        self.K = K
        self.Rt = Rt
        self.cam2world = cam2world
        self.names = names or []

    def get_object_mask(self, name, camera):
        return self.mask

    def get_depth_image(self, camera):
        return self.depth

    def get_camera_matrices(self, camera):
        return self.K, self.Rt, self.cam2world

    def get_object_names(self):
        return list(self.names)

    def get_image_size(self, camera):
        return self.depth.shape if self.depth is not None else None


def case_perception_pixel_world_roundtrip():
    """world_to_pixel ∘ pixel_to_world_point is identity for visible points."""
    import numpy as np
    from vision_perception import VisionPerception

    # Identity-ish camera: K is a 320x180 D435-like intrinsic, Rt = I → camera at world origin facing +z.
    K = np.array([[320.0, 0.0, 160.0],
                  [0.0, 320.0, 90.0],
                  [0.0, 0.0, 1.0]])
    Rt = np.eye(4)
    cam2world = np.eye(4)
    backend = _FakeBackend(
        mask=np.zeros((180, 320), dtype=bool),  # unused here
        depth=np.full((180, 320), 1.0, dtype=np.float64),
        K=K, Rt=Rt, cam2world=cam2world,
    )
    vp = VisionPerception(TASK_ENV=None, backend=backend)

    # World point (0.2, 0.1, 1.0) in cam = same in world (Rt=I).
    target = [0.2, 0.1, 1.0]
    r = vp.world_to_pixel(target)
    assert r["pixel_uv"] is not None, r
    u, v = r["pixel_uv"]
    depth = r["depth_m"]

    back = vp.pixel_to_world_point(u, v, depth)
    rec = back["world_xyz"]
    err = max(abs(rec[i] - target[i]) for i in range(3))
    ok = err < 5e-3   # 5 mm tolerance after rounding to int pixels
    return _record("perception_pixel_world_roundtrip", ok, f"err={err:.5f} m, uv={(u,v)}, depth={depth}")


def case_perception_pca_rod_axis():
    """PCA on a synthetic rod-shaped point cloud returns ±x principal axis."""
    import numpy as np
    from vision_perception import VisionPerception

    # Build a synthetic 320x180 scene with depth=1.0 everywhere; the mask
    # selects a horizontal strip of pixels.  After unprojection, this strip
    # becomes a rod along the camera's +x axis (which == +x world for Rt=I).
    H, W = 180, 320
    mask = np.zeros((H, W), dtype=bool)
    mask[H // 2, 50:270] = True   # horizontal strip
    depth = np.full((H, W), 1.0, dtype=np.float64)
    K = np.array([[320.0, 0.0, 160.0],
                  [0.0, 320.0, 90.0],
                  [0.0, 0.0, 1.0]])
    Rt = np.eye(4)
    backend = _FakeBackend(mask=mask, depth=depth, K=K, Rt=Rt, cam2world=np.eye(4))
    vp = VisionPerception(TASK_ENV=None, backend=backend)
    pose = vp.get_object_pose("rod")

    visible = pose["visible"]
    axis = pose.get("principal_axis_world")
    ok = visible and axis is not None and abs(axis[0]) > 0.95 and abs(axis[1]) < 0.1 and abs(axis[2]) < 0.1
    return _record("perception_pca_rod_axis", ok, f"axis={axis}, visible={visible}")


def case_perception_empty_mask_invisible():
    """All-False mask → visible=False, pos_world=None."""
    import numpy as np
    from vision_perception import VisionPerception

    H, W = 180, 320
    mask = np.zeros((H, W), dtype=bool)
    depth = np.full((H, W), 1.0, dtype=np.float64)
    K = np.eye(3); Rt = np.eye(4)
    backend = _FakeBackend(mask=mask, depth=depth, K=K, Rt=Rt, cam2world=np.eye(4))
    vp = VisionPerception(None, backend=backend)
    pose = vp.get_object_pose("ghost")
    ok = pose["visible"] is False and pose["pos_world"] is None
    return _record("perception_empty_mask_invisible", ok, str(pose))


def case_perception_low_depth_valid_ratio_flagged():
    """Mask with mostly-NaN depth has depth_valid_ratio < 0.3."""
    import numpy as np
    from vision_perception import VisionPerception

    H, W = 180, 320
    mask = np.zeros((H, W), dtype=bool)
    mask[80:100, 150:170] = True   # 400 pixels
    depth = np.full((H, W), np.nan, dtype=np.float64)
    # Only 5% of the mask has valid depth — below threshold but enough for math.
    depth[80:81, 150:170] = 1.0
    K = np.array([[320.0, 0.0, 160.0],
                  [0.0, 320.0, 90.0],
                  [0.0, 0.0, 1.0]])
    Rt = np.eye(4)
    backend = _FakeBackend(mask=mask, depth=depth, K=K, Rt=Rt, cam2world=np.eye(4))
    vp = VisionPerception(None, backend=backend)
    pose = vp.get_object_pose("partial")
    ok = pose["visible"] is True and pose["depth_valid_ratio"] < 0.3
    return _record("perception_low_depth_valid_ratio_flagged", ok,
                    f"ratio={pose.get('depth_valid_ratio')}")


def case_perception_world_to_pixel_behind_camera():
    """A point behind the camera returns in_view=False."""
    import numpy as np
    from vision_perception import VisionPerception

    K = np.eye(3)
    Rt = np.eye(4)
    backend = _FakeBackend(
        mask=np.zeros((10, 10), dtype=bool),
        depth=np.zeros((10, 10), dtype=np.float64),
        K=K, Rt=Rt, cam2world=np.eye(4),
    )
    vp = VisionPerception(None, backend=backend)
    # In OpenCV cam frame +z is forward.  Rt=I → world z negative is "behind"
    r = vp.world_to_pixel([0.0, 0.0, -1.0])
    ok = r["in_view"] is False
    return _record("perception_world_to_pixel_behind_camera", ok, str(r))


def case_perception_flag_defaults_off():
    """TaPExecutor defaults to perception disabled."""
    ex = TaPExecutor(_FakeLLM())
    ok = (ex.use_perception is False
          and ex.perception_backend == "sim"
          and ex._perception is None
          and ex._tracked_objects == set())
    return _record("perception_flag_defaults_off", ok)


def case_perception_flag_on_and_tool_registry():
    """TaPExecutor accepts use_perception=True and the 5 new tools are registered."""
    ex = TaPExecutor(_FakeLLM(), use_perception=True)
    expected_new = {"get_depth_at_pixel", "pixel_to_world_point", "world_to_pixel",
                    "rotate_gripper", "move_to_pose"}
    ok = (ex.use_perception is True
          and ex.perception_backend == "sim"
          and expected_new.issubset(set(TAP_TOOL_REGISTRY.keys())))
    return _record("perception_flag_on_and_tool_registry", ok,
                    f"missing={expected_new - set(TAP_TOOL_REGISTRY.keys())}")


def case_perception_system_prompt_extends_when_enabled():
    """TAP_EXECUTOR_SYSTEM_PERCEPTION text exists and mentions key fields."""
    from tap_executor import TAP_EXECUTOR_SYSTEM_PERCEPTION
    needed = ["scene_perception", "principal_axis_world", "extent_world",
              "world_to_pixel", "pixel_to_world_point", "rotate_gripper",
              "move_to_pose", "head_camera_viewpoint"]
    missing = [k for k in needed if k not in TAP_EXECUTOR_SYSTEM_PERCEPTION]
    ok = len(missing) == 0
    return _record("perception_system_prompt_extends_when_enabled", ok,
                    f"missing={missing}")


def case_validate_v2_list_args_lengths():
    """Length validation on world_xyz/axis_world/target_xyz/target_quat."""
    err = TaPExecutor._validate_tool_call(
        "world_to_pixel",
        {"camera_name": "head_camera", "world_xyz": [0.0, 0.0]},  # too short
    )
    ok = err is not None and ("length" in err.lower())
    err2 = TaPExecutor._validate_tool_call(
        "move_to_pose",
        {"arm_tag": "left",
         "target_xyz": [0.0, 0.0, 0.0],
         "target_quat": [1.0, 0.0, 0.0]},  # too short
    )
    ok = ok and (err2 is not None) and ("length" in err2.lower())
    err3 = TaPExecutor._validate_tool_call(
        "rotate_gripper",
        {"arm_tag": "left",
         "axis_world": [0.0, 0.0, 1.0],
         "angle_deg": 30.0},
    )
    ok = ok and (err3 is None)
    return _record("validate_v2_list_args_lengths", ok,
                    f"err={err}, err2={err2}, err3={err3}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cases = [
        case_parse_tool_call_valid,
        case_parse_tool_call_with_fences,
        case_parse_rejects_missing_tool,
        case_parse_rejects_non_dict_args,
        case_validate_rejects_unknown_tool,
        case_validate_rejects_forbidden_substring,
        case_validate_arm_tag,
        case_validate_arm_tag_valid,
        case_validate_camera_name,
        case_validate_missing_required_arg,
        case_validate_check_task_success_no_args,
        case_planner_summary_no_raw_env,
        case_system_prompt_no_raw_env_api,
        case_tool_registry_complete,
        case_find_action_deterministic,
        case_initial_env_query_no_contact,
        # v1.5 vision tests
        case_vision_system_prompt_extends_text,
        case_encode_image_nonexistent_returns_none,
        case_encode_image_valid_file,
        case_vision_flag_defaults_off,
        case_vision_flag_on,
        # v2 perception tests
        case_perception_pixel_world_roundtrip,
        case_perception_pca_rod_axis,
        case_perception_empty_mask_invisible,
        case_perception_low_depth_valid_ratio_flagged,
        case_perception_world_to_pixel_behind_camera,
        case_perception_flag_defaults_off,
        case_perception_flag_on_and_tool_registry,
        case_perception_system_prompt_extends_when_enabled,
        case_validate_v2_list_args_lengths,
    ]
    for case in cases:
        try:
            case()
        except Exception as e:
            _record(case.__name__, False, f"raised: {e}")
        print()
    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print("=" * 60)
    print(f"TaP executor test summary: {passed}/{total} passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
