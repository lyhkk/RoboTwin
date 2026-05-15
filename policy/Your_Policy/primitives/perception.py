"""
Perception primitives — read-only views of the simulation state.

These delegate to `privileged_perception.get_scene_objects` and the robot's
own EE pose accessors. No side effects on the env.
"""

from typing import Optional

from .result import (
    SUCCESS, FAILED, STAGE_PERCEPTION,
    make_primitive_result,
)


# Lazy import to avoid circulars; privileged_perception lives next to this pkg
def _scene_objects(TASK_ENV) -> dict:
    from privileged_perception import get_scene_objects
    return get_scene_objects(TASK_ENV)


# ── Object pose ───────────────────────────────────────────────────────────

def get_object_pose(TASK_ENV, object_name: str) -> dict:
    """
    Read world pose of `object_name` from SAPIEN.

    Returns a PrimitiveResult whose `data` contains:
        position: [x, y, z]
        orientation: [qw, qx, qy, qz]  (SAPIEN convention)
    """
    objects = _scene_objects(TASK_ENV)
    if object_name not in objects:
        return make_primitive_result(
            "get_object_pose", FAILED,
            f"Object '{object_name}' not found in scene. "
            f"Known names: {list(objects.keys())[:10]}",
            object_name=object_name, position=None, orientation=None,
        )
    info = objects[object_name]
    return make_primitive_result(
        "get_object_pose", SUCCESS,
        f"Pose of '{object_name}' read.",
        object_name=object_name,
        position=list(info["position"]),
        orientation=list(info["orientation"]),
    )


# ── Gripper EE pose ───────────────────────────────────────────────────────

# Approximate maximum gripper opening width (meters).  Used to convert the
# normalised gripper_val [0,1] returned by the robot wrapper into a
# meters-scale `gripper_width_m`, so LLMs can compare it directly with
# `extent_world` numbers from VisionPerception.  Per embodiment the exact
# max width varies; this constant is the typical aloha-agilex value.
_APPROX_MAX_GRIPPER_WIDTH_M = 0.08


def _read_gripper_width(TASK_ENV, arm: str) -> Optional[float]:
    """Best-effort read of the current gripper opening in meters.

    Source of truth is ``robot.get_*_gripper_val()`` which returns a value
    in ``[0, 1]`` (open = 1).  We rescale by an approximate max width so the
    LLM sees a meters-scale number it can compare with ``extent_world``.
    Returns ``None`` on failure.
    """
    try:
        if arm == "left":
            val = float(TASK_ENV.robot.get_left_gripper_val())
        else:
            val = float(TASK_ENV.robot.get_right_gripper_val())
    except Exception:
        return None
    val = max(0.0, min(1.0, val))
    return val * _APPROX_MAX_GRIPPER_WIDTH_M


def get_gripper_pose(TASK_ENV, arm: str) -> dict:
    """
    Return current end-effector pose for `arm` ("left" or "right").

    Pose format follows env.get_*_ee_pose: [x, y, z, qw, qx, qy, qz].
    Additionally returns ``gripper_width_m`` (approximate finger opening in
    meters, derived from the normalised gripper joint value).
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "get_gripper_pose", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}",
            arm=arm, pose=None, gripper_width_m=None,
        )
    try:
        if arm == "left":
            pose = list(TASK_ENV.robot.get_left_ee_pose())
        else:
            pose = list(TASK_ENV.robot.get_right_ee_pose())
    except Exception as e:
        return make_primitive_result(
            "get_gripper_pose", FAILED,
            f"Failed to read {arm} EE pose: {e}",
            arm=arm, pose=None, gripper_width_m=None,
        )
    gripper_width_m = _read_gripper_width(TASK_ENV, arm)
    return make_primitive_result(
        "get_gripper_pose", SUCCESS,
        f"{arm} EE pose read.",
        arm=arm, pose=pose,
        gripper_width_m=(round(gripper_width_m, 4)
                         if gripper_width_m is not None else None),
    )


# ── Gripper open/close state ──────────────────────────────────────────────

def get_gripper_state(TASK_ENV, arm: str) -> dict:
    """
    Return open/closed status and raw gripper value for `arm`.
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "get_gripper_state", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}",
            arm=arm,
        )
    try:
        if arm == "left":
            val = float(TASK_ENV.robot.get_left_gripper_val())
            is_closed = bool(TASK_ENV.robot.is_left_gripper_close())
        else:
            val = float(TASK_ENV.robot.get_right_gripper_val())
            is_closed = bool(TASK_ENV.robot.is_right_gripper_close())
    except Exception as e:
        return make_primitive_result(
            "get_gripper_state", FAILED,
            f"Failed to read {arm} gripper state: {e}",
            arm=arm,
        )
    return make_primitive_result(
        "get_gripper_state", SUCCESS,
        f"{arm} gripper state: {'closed' if is_closed else 'open'} (val={val:.3f}).",
        arm=arm, gripper_val=val, is_closed=is_closed,
    )


# ── Camera snapshot ──────────────────────────────────────────────────────

VALID_CAMERAS = ("head_camera", "left_camera", "right_camera")


def get_camera_snapshot(TASK_ENV, camera_name: str, save_path: str) -> dict:
    """
    Render and save an RGB image from the specified camera.

    Returns a PrimitiveResult whose ``data`` contains:
        camera_name: str
        save_path: str   (always a plain str, JSON-safe)

    The image is written to *save_path* on disk.  The caller (TaP runtime)
    is responsible for creating the parent directory beforehand.
    """
    if camera_name not in VALID_CAMERAS:
        return make_primitive_result(
            "get_camera_snapshot", FAILED,
            f"camera_name must be one of {VALID_CAMERAS}, got {camera_name!r}",
            camera_name=camera_name, save_path=None,
        )
    try:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        TASK_ENV.save_camera_rgb(str(save_path), camera_name)
    except Exception as e:
        return make_primitive_result(
            "get_camera_snapshot", FAILED,
            f"Failed to save camera RGB: {e}",
            camera_name=camera_name, save_path=str(save_path),
        )
    return make_primitive_result(
        "get_camera_snapshot", SUCCESS,
        f"Snapshot saved from {camera_name}.",
        camera_name=camera_name, save_path=str(save_path),
    )
