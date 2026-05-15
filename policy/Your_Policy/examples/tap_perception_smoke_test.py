"""
TaP v2 perception smoke test (sim-backed; no LLM).

Loads a deterministic ``lift_pot`` scene with seed 0, builds a
``VisionPerception`` with the sim backend, and verifies:

  1. ``get_object_pose("pot")`` returns a visible result with
     ``‖pos_world − env.GT‖₂ < 0.05 m`` (5 cm).
  2. ``depth_valid_ratio > 0.5``.
  3. ``world_to_pixel`` ∘ ``pixel_to_world_point`` recovers the original
     world coords within 1 cm.
  4. ``describe_camera_viewpoint`` returns a non-empty description string
     that mentions image-axis-to-world-axis mapping.

Run:
    cd ~/Documents/GitHub/RoboTwin
    python policy/Your_Policy/examples/tap_perception_smoke_test.py

The test reuses RoboTwin's standard task-loading path (see
``script/eval_policy.py``) so it requires a working SAPIEN install + GPU
(or EGL for headless rendering).
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_POLICY_DIR = _HERE.parent
_ROBOTWIN_ROOT = _POLICY_DIR.parent.parent
for p in (_POLICY_DIR, _ROBOTWIN_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


_RESULTS = []


def _record(name, ok, details=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}")
    if details:
        for line in str(details).split("\n"):
            print(f"      {line}")
    _RESULTS.append((name, ok))
    return ok


def _load_lift_pot_seed_zero():
    """Boot the lift_pot env with the demo_clean_aloha config at seed 0.

    Mirrors the loading path used by ``script/eval_policy.py:main`` so the
    smoke test stays valid even if helpers change.  Returns a TASK_ENV
    instance ready for camera + perception queries.
    """
    import os
    import yaml

    # Resolve config files in the same way eval_policy.py does
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    task_config_path = Path(_ROBOTWIN_ROOT) / "task_config" / "demo_clean_aloha.yml"
    with open(task_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["seed"] = 0
    args["task_name"] = "lift_pot"
    args["task_config"] = "demo_clean_aloha"
    args["skip_expert_check"] = True
    args["eval_video_log"] = False

    # Embodiment resolution
    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"),
              "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def _get_embodiment_file(name):
        rf = _embodiment_types[name]["file_path"]
        if rf is None:
            raise RuntimeError(f"No embodiment files for {name!r}")
        return rf

    def _get_embodiment_config(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r",
                  encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items must be 1 or 3")
    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])

    # Camera config (eval_policy.py:108-112)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"),
              "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # Import the task class
    from envs.lift_pot import lift_pot
    env = lift_pot()
    env.setup_demo(**args)
    if hasattr(env, "delay"):
        env.delay(10)
    return env


def case_perception_smoke():
    try:
        env = _load_lift_pot_seed_zero()
    except Exception as e:
        return _record(
            "perception_smoke_env_load", False,
            f"failed to load env: {e}\n{traceback.format_exc()[:600]}",
        )

    try:
        from vision_perception import VisionPerception, SimPerceptionBackend
        from privileged_perception import get_scene_objects
        backend = SimPerceptionBackend(env)
        vp = VisionPerception(env, backend=backend)
    except Exception as e:
        return _record(
            "perception_smoke_init", False,
            f"failed to build VisionPerception: {e}\n{traceback.format_exc()[:600]}",
        )

    objs = get_scene_objects(env)
    if not objs:
        return _record("perception_smoke_scene_empty", False,
                       "get_scene_objects returned empty")

    # Pick a target — anything with "pot" in the name; else the first object.
    target = None
    for name in objs:
        if "pot" in name.lower():
            target = name
            break
    target = target or sorted(objs.keys())[0]
    print(f"[smoke] target object: {target}")
    gt_pos = objs[target]["position"]
    print(f"[smoke] GT position: {gt_pos}")

    # ── Get 3D pose from perception ─────────────────────────────────
    pose = vp.get_object_pose(target, "head_camera")
    print(f"[smoke] perception result: {pose}")
    if not pose.get("visible") or pose.get("pos_world") is None:
        return _record(
            "perception_smoke_visible", False,
            f"target {target} not visible; pose={pose}",
        )
    pred = np.array(pose["pos_world"])
    err = float(np.linalg.norm(pred - np.array(gt_pos)))
    _record("perception_smoke_pose_within_5cm", err < 0.05,
            f"err={err*100:.2f} cm  pred={pred.tolist()}  GT={gt_pos}")

    # ── Depth valid ratio ──────────────────────────────────────────
    ratio = pose.get("depth_valid_ratio", 0)
    _record("perception_smoke_depth_valid_ratio_gt_0p5", ratio > 0.5,
            f"depth_valid_ratio={ratio}")

    # ── world↔pixel roundtrip ──────────────────────────────────────
    proj = vp.world_to_pixel(pred.tolist())
    if proj["pixel_uv"] is None:
        _record("perception_smoke_pixel_roundtrip", False,
                f"projection out of view: {proj}")
    else:
        u, v = proj["pixel_uv"]
        d = proj["depth_m"]
        back = vp.pixel_to_world_point(u, v, d)
        if back["world_xyz"] is None:
            _record("perception_smoke_pixel_roundtrip", False,
                    f"unprojection failed: {back}")
        else:
            err_rt = float(np.linalg.norm(np.array(back["world_xyz"]) - pred))
            _record("perception_smoke_pixel_roundtrip", err_rt < 0.01,
                    f"roundtrip err={err_rt*100:.2f} cm via pixel {(u,v)}")

    # ── Viewpoint description ──────────────────────────────────────
    vp_info = vp.describe_camera_viewpoint("head_camera")
    desc = vp_info.get("description", "")
    keywords_present = ("mounted at" in desc and "image-right" in desc.lower())
    _record("perception_smoke_viewpoint_description", keywords_present,
            f"description={desc!r}")

    return True


def main():
    case_perception_smoke()
    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print("=" * 60)
    print(f"TaP v2 perception smoke: {passed}/{total} passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
