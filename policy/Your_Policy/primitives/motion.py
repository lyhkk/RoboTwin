"""
Motion primitives — thin wrappers over `env.move(env.<primitive>(...))`.

Contract for every motion primitive in this module:

  1. Reset TASK_ENV.plan_success = True *before* calling env.move().
     (plan_success is sticky-False; it must be re-armed each motion.)
  2. Call env.move(...) which returns True/False/None.
  3. Inspect TASK_ENV.plan_success *after* the call as the source of truth.
  4. Return a PrimitiveResult with ee_before / ee_after / plan_success in `data`.

Pose convention: SAPIEN wxyz = [x, y, z, qw, qx, qy, qz]. See pose_utils.py.

Step-budget note: env.move() goes through env.take_dense_action() which does
NOT increment TASK_ENV.take_action_cnt. The eval loop watches that counter,
so a runner using these primitives must explicitly bump the counter to exit.
See examples/primitive_lift_pot.py for the recommended pattern.
"""

from typing import Optional, Sequence

from envs.utils.action import ArmTag

from .result import (
    SUCCESS, FAILED, STAGE_MOTION,
    make_primitive_result,
)
from .perception import get_gripper_pose
from . import pose_utils


# ── Helpers ───────────────────────────────────────────────────────────────

def _arm_tag(arm: str) -> ArmTag:
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return ArmTag(arm)


def _safe_ee_pose(TASK_ENV, arm: str):
    """Return the current EE pose for `arm`, or None on failure."""
    r = get_gripper_pose(TASK_ENV, arm)
    return r["data"].get("pose") if r["status"] == SUCCESS else None


def _validate_target_pose(target_pose: Sequence[float]) -> Optional[str]:
    if target_pose is None:
        return "target_pose is None"
    if len(target_pose) != 7:
        return f"target_pose must be length 7, got {len(target_pose)}"
    ok, msg = pose_utils.check_workspace_bounds(target_pose[:3])
    if not ok:
        return msg
    return None


def _validate_delta(dx: float, dy: float, dz: float) -> Optional[str]:
    ok, mag = pose_utils.check_delta_magnitude(dx, dy, dz)
    if not ok:
        return f"delta magnitude {mag:.3f}m exceeds limit {pose_utils.MAX_DELTA_MAGNITUDE}m"
    return None


# ── Single-arm absolute move ──────────────────────────────────────────────

def move_to_pose(TASK_ENV, arm: str, target_pose: Sequence[float]) -> dict:
    """
    Move one arm to absolute pose [x, y, z, qw, qx, qy, qz].
    """
    msg = _validate_target_pose(target_pose)
    if msg:
        return make_primitive_result(
            "move_to_pose", FAILED, f"Invalid target pose: {msg}",
            arm=arm, target_pose=list(target_pose) if target_pose is not None else None,
            plan_success=False,
        )

    ee_before = _safe_ee_pose(TASK_ENV, arm)
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(TASK_ENV.move_to_pose(_arm_tag(arm), list(target_pose)))
    except Exception as e:
        return make_primitive_result(
            "move_to_pose", FAILED, f"env.move raised: {e}",
            arm=arm, target_pose=list(target_pose),
            ee_before=ee_before, ee_after=_safe_ee_pose(TASK_ENV, arm),
            plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    ee_after = _safe_ee_pose(TASK_ENV, arm)
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_to_pose", SUCCESS if ok else FAILED,
        f"{arm} arm reached target." if ok else f"{arm} arm motion planning failed.",
        arm=arm, target_pose=list(target_pose),
        ee_before=ee_before, ee_after=ee_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Single-arm delta move ─────────────────────────────────────────────────

def move_delta(TASK_ENV, arm: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
    """
    Move one arm by world-frame displacement.
    """
    msg = _validate_delta(dx, dy, dz)
    if msg:
        return make_primitive_result(
            "move_delta", FAILED, msg,
            arm=arm, delta=[dx, dy, dz], plan_success=False,
        )

    ee_before = _safe_ee_pose(TASK_ENV, arm)
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(TASK_ENV.move_by_displacement(_arm_tag(arm), x=dx, y=dy, z=dz))
    except Exception as e:
        return make_primitive_result(
            "move_delta", FAILED, f"env.move raised: {e}",
            arm=arm, delta=[dx, dy, dz], ee_before=ee_before,
            ee_after=_safe_ee_pose(TASK_ENV, arm), plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    ee_after = _safe_ee_pose(TASK_ENV, arm)
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_delta", SUCCESS if ok else FAILED,
        f"{arm} arm displaced by [{dx:.3f}, {dy:.3f}, {dz:.3f}]." if ok
        else f"{arm} arm displacement motion planning failed.",
        arm=arm, delta=[dx, dy, dz],
        ee_before=ee_before, ee_after=ee_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Dual-arm absolute move ────────────────────────────────────────────────

def move_both_to_poses(TASK_ENV, left_pose: Sequence[float],
                       right_pose: Sequence[float]) -> dict:
    """
    Move both arms simultaneously to the given absolute poses.
    Both poses must be [x, y, z, qw, qx, qy, qz].
    """
    for name, p in (("left_pose", left_pose), ("right_pose", right_pose)):
        msg = _validate_target_pose(p)
        if msg:
            return make_primitive_result(
                "move_both_to_poses", FAILED, f"Invalid {name}: {msg}",
                left_pose=list(left_pose) if left_pose is not None else None,
                right_pose=list(right_pose) if right_pose is not None else None,
                plan_success=False,
            )

    left_before = _safe_ee_pose(TASK_ENV, "left")
    right_before = _safe_ee_pose(TASK_ENV, "right")
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(
            TASK_ENV.move_to_pose(ArmTag("left"), list(left_pose)),
            TASK_ENV.move_to_pose(ArmTag("right"), list(right_pose)),
        )
    except Exception as e:
        return make_primitive_result(
            "move_both_to_poses", FAILED, f"env.move raised: {e}",
            left_pose=list(left_pose), right_pose=list(right_pose),
            left_ee_before=left_before, right_ee_before=right_before,
            left_ee_after=_safe_ee_pose(TASK_ENV, "left"),
            right_ee_after=_safe_ee_pose(TASK_ENV, "right"),
            plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    left_after = _safe_ee_pose(TASK_ENV, "left")
    right_after = _safe_ee_pose(TASK_ENV, "right")
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_both_to_poses", SUCCESS if ok else FAILED,
        "Both arms reached target poses." if ok
        else "Dual-arm motion planning failed.",
        left_pose=list(left_pose), right_pose=list(right_pose),
        left_ee_before=left_before, right_ee_before=right_before,
        left_ee_after=left_after, right_ee_after=right_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Dual-arm delta move ───────────────────────────────────────────────────

def move_both_delta(TASK_ENV,
                    left_delta: Sequence[float],
                    right_delta: Sequence[float]) -> dict:
    """
    Move both arms simultaneously by world-frame displacements.
    Each delta is [dx, dy, dz].
    """
    for name, d in (("left_delta", left_delta), ("right_delta", right_delta)):
        if d is None or len(d) != 3:
            return make_primitive_result(
                "move_both_delta", FAILED,
                f"Invalid {name}: must be length 3, got {d!r}",
                left_delta=list(left_delta) if left_delta is not None else None,
                right_delta=list(right_delta) if right_delta is not None else None,
                plan_success=False,
            )
        msg = _validate_delta(d[0], d[1], d[2])
        if msg:
            return make_primitive_result(
                "move_both_delta", FAILED, f"Invalid {name}: {msg}",
                left_delta=list(left_delta), right_delta=list(right_delta),
                plan_success=False,
            )

    left_before = _safe_ee_pose(TASK_ENV, "left")
    right_before = _safe_ee_pose(TASK_ENV, "right")
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(
            TASK_ENV.move_by_displacement(ArmTag("left"),
                                          x=float(left_delta[0]),
                                          y=float(left_delta[1]),
                                          z=float(left_delta[2])),
            TASK_ENV.move_by_displacement(ArmTag("right"),
                                          x=float(right_delta[0]),
                                          y=float(right_delta[1]),
                                          z=float(right_delta[2])),
        )
    except Exception as e:
        return make_primitive_result(
            "move_both_delta", FAILED, f"env.move raised: {e}",
            left_delta=list(left_delta), right_delta=list(right_delta),
            left_ee_before=left_before, right_ee_before=right_before,
            left_ee_after=_safe_ee_pose(TASK_ENV, "left"),
            right_ee_after=_safe_ee_pose(TASK_ENV, "right"),
            plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    left_after = _safe_ee_pose(TASK_ENV, "left")
    right_after = _safe_ee_pose(TASK_ENV, "right")
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_both_delta", SUCCESS if ok else FAILED,
        f"Both arms displaced (L={list(left_delta)}, R={list(right_delta)})." if ok
        else "Dual-arm displacement motion planning failed.",
        left_delta=list(left_delta), right_delta=list(right_delta),
        left_ee_before=left_before, right_ee_before=right_before,
        left_ee_after=left_after, right_ee_after=right_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Home recovery ─────────────────────────────────────────────────────────

def move_to_home(TASK_ENV, arm: Optional[str] = None) -> dict:
    """
    Move one or both arms back to their original (home) pose.
    `arm=None` → both arms (left then right).
    """
    if arm is not None and arm not in ("left", "right"):
        return make_primitive_result(
            "move_to_home", FAILED,
            f"arm must be 'left', 'right', or None; got {arm!r}",
            arm=arm, plan_success=False,
        )

    arms = ("left", "right") if arm is None else (arm,)
    all_ok = True
    sub_results = []
    for a in arms:
        TASK_ENV.plan_success = True
        try:
            TASK_ENV.move(TASK_ENV.back_to_origin(ArmTag(a)))
            arm_ok = bool(TASK_ENV.plan_success)
        except Exception as e:
            arm_ok = False
            sub_results.append({"arm": a, "ok": False, "error": str(e)})
        else:
            sub_results.append({"arm": a, "ok": arm_ok})
        all_ok = all_ok and arm_ok

    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_to_home", SUCCESS if all_ok else FAILED,
        f"Home recovery: {sub_results}.",
        arm=arm, sub_results=sub_results,
        plan_success=all_ok, motion_completed=all_ok, sim_step=sim_step,
    )
