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

def get_gripper_pose(TASK_ENV, arm: str) -> dict:
    """
    Return current end-effector pose for `arm` ("left" or "right").

    Pose format follows env.get_*_ee_pose: [x, y, z, qw, qx, qy, qz].
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "get_gripper_pose", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}",
            arm=arm, pose=None,
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
            arm=arm, pose=None,
        )
    return make_primitive_result(
        "get_gripper_pose", SUCCESS,
        f"{arm} EE pose read.",
        arm=arm, pose=pose,
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
