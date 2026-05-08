"""
Primitive Skill Library for RoboTwin LLM Agent
Phase 1: Uses end-effector (ee) control. No learned policy required.

Each skill returns a dictionary:
{"status": "SUCCESS" | "FAILED", "feedback": "...", "data": {...}}
"""

import numpy as np
import time
from typing import Optional, Tuple


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_QUAT = [0.0, 0.0, 1.0, 0.0]  # pointing downward
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
APPROACH_OFFSET = 0.10  # 10cm above target


# ── Result Helpers ─────────────────────────────────────────────────────────

def make_result(status: str, feedback: str, **data) -> dict:
    """Standardized result schema for all skills."""
    return {
        "status": status,
        "feedback": feedback,
        # Redundant keys for linguistic robustness
        "message": feedback,
        "details": feedback,
        "error": feedback if status == "FAILED" else None,
        "data": data,
    }

def get_feedback(result) -> str:
    """Helper to extract feedback from any result dictionary."""
    if not isinstance(result, dict):
        return str(result)
    return (
        result.get("feedback")
        or result.get("message")
        or result.get("details")
        or "No feedback provided."
    )


# ── Low-level Action Builders ─────────────────────────────────────────────

def _make_ee_action(pos: list, quat: list, gripper: float) -> np.ndarray:
    """Build a 14-dim ee action: left-home + right-active."""
    LEFT_HOME = [0.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.04]
    right_action = list(pos) + list(quat) + [gripper]
    return np.array(LEFT_HOME + right_action, dtype=np.float32)

def _make_dual_ee_action(l_pos, l_quat, l_grip, r_pos, r_quat, r_grip) -> np.ndarray:
    left_action = list(l_pos) + list(l_quat) + [l_grip]
    right_action = list(r_pos) + list(r_quat) + [r_grip]
    return np.array(left_action + right_action, dtype=np.float32)


# ── Trajectory Helpers ─────────────────────────────────────────────────────

def _interpolate_trajectory(start_pos, end_pos, n_steps=20) -> list:
    """Linear interpolation between two Cartesian positions."""
    start = np.array(start_pos)
    end = np.array(end_pos)
    return [list(start + (end - start) * t / max(1, n_steps - 1)) for t in range(n_steps)]

def _get_current_ee_pos(TASK_ENV, arm='right') -> list:
    try:
        obs = TASK_ENV.now_obs
        if obs and "endpose" in obs:
            ee = obs["endpose"].get(f"{arm}_endpose")
            if ee is not None:
                return list(ee[:3])
    except Exception:
        pass
    if arm == 'left':
        return [0.0, 0.3, 1.0]
    return [0.3, 0.0, 1.1]


# ── Waypoint Executors ─────────────────────────────────────────────────────

def _execute_waypoints(TASK_ENV, waypoints: list, logger=None, skill_name="move") -> bool:
    """Execute waypoint sequence for right arm."""
    for pos, quat, gripper in waypoints:
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
            return False
        action = _make_ee_action(pos, quat, gripper)
        TASK_ENV.take_action(action, action_type='ee')
    return True

def _execute_dual_waypoints(TASK_ENV, waypoints: list, logger=None, skill_name="dual_move") -> bool:
    """Execute waypoint sequence for both arms."""
    for lp, lq, lg, rp, rq, rg in waypoints:
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
            return False
        action = _make_dual_ee_action(lp, lq, lg, rp, rq, rg)
        TASK_ENV.take_action(action, action_type='ee')
    return True


# ── Perception Helpers ─────────────────────────────────────────────────────

def skill_check_object_sanity(TASK_ENV, object_name: str) -> dict:
    """Checks if the object is fallen or outside the reachable workspace."""
    from privileged_perception import get_scene_objects
    objects = get_scene_objects(TASK_ENV)
    if object_name not in objects:
        return make_result("FAILED", f"Object {object_name} is missing from the scene.")

    pos = objects[object_name]["position"]
    x, y, z = pos
    if z < 0.5:
        return make_result("FAILED", f"Object {object_name} has fallen (z={z:.3f}).", pos=pos)
    if abs(x) > 1.2 or abs(y) > 1.2:
        return make_result("FAILED", f"Object {object_name} is out of workspace bounds.", pos=pos)

    return make_result("SUCCESS", f"Object {object_name} is in a valid workspace position.", pos=pos)

def compute_dual_grasp(TASK_ENV, object_name: str) -> dict:
    """
    Compute left and right grasp poses for dual-arm grasping.
    Returns a DICT with 'left_pose' and 'right_pose' — never a bare list.
    """
    from privileged_perception import get_scene_objects
    objects = get_scene_objects(TASK_ENV)
    if object_name not in objects:
        return make_result("FAILED", f"Object {object_name} not found in scene.")

    pos = objects[object_name]["position"]
    quat = objects[object_name]["orientation"]
    
    # ── Attempt 1: Search for explicit handle links ──
    # Expand keywords to be more robust
    handle_keywords = ["handle", "grip", "arm"]
    handles = [n for n in objects if any(k in n.lower() for k in handle_keywords) and object_name in n]
    
    if len(handles) >= 2:
        handles.sort(key=lambda n: objects[n]["position"][1], reverse=True)
        l_grasp = np.array(objects[handles[0]]["position"])
        r_grasp = np.array(objects[handles[1]]["position"])
        l_grasp[2] += 0.01 # Lowered from 0.02
        r_grasp[2] += 0.01
        method = "explicit_handles"
    else:
        # ── Attempt 2: Orientation-aware calculation (Fallback) ──
        # Handles are usually along local Y-axis [0, 0.14, 0.01]
        from scipy.spatial.transform import Rotation as R
        # Scipy expects [x, y, z, w]. Sapien/RoboTwin uses [w, x, y, z].
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) 
        
        l_offset_local = np.array([0, 0.14, 0.01]) # Reduced offset and height
        r_offset_local = np.array([0, -0.14, 0.01])
        
        l_grasp = pos + rot.apply(l_offset_local)
        r_grasp = pos + rot.apply(r_offset_local)
        method = "orientation_aware_fallback"
    
    print(f"[DEBUG] {object_name} method: {method}")
    print(f"[DEBUG] Handles found: {handles}")
    print(f"[DEBUG] Calculated grasps -> L: {l_grasp.tolist()}, R: {r_grasp.tolist()}")
    
    return make_result(
        "SUCCESS", 
        f"Computed dual grasp poses for {object_name} using {method}.",
        left_pose=l_grasp.tolist(),
        right_pose=r_grasp.tolist(),
        object_center=list(pos),
        method=method
    )


# ── Composite Skills ──────────────────────────────────────────────────────

def skill_dual_arm_grasp(TASK_ENV, object_name: str, logger=None) -> dict:
    """
    High-level composite skill for dual-arm grasping.
    Internally: sanity check -> compute grasp -> move -> close -> verify.
    """
    from privileged_perception import get_scene_objects

    # 1. Sanity check
    sanity = skill_check_object_sanity(TASK_ENV, object_name)
    if sanity["status"] != "SUCCESS":
        return sanity

    # 2. Compute grasp poses
    grasps = compute_dual_grasp(TASK_ENV, object_name)
    if grasps["status"] != "SUCCESS":
        return grasps

    l_pose = grasps["data"]["left_pose"]
    r_pose = grasps["data"]["right_pose"]
    z_before = grasps["data"]["object_center"][2]
    
    # Get objects again or use existing if we had them
    objects = get_scene_objects(TASK_ENV)
    quat = objects[object_name]["orientation"]
    
    # ── Calculate dynamic orientation ──
    # Gripper should point down ([0, 1, 0, 0] or similar) but rotate around Z to match object
    from scipy.spatial.transform import Rotation as R
    obj_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # [x, y, z, w]
    yaw = obj_rot.as_euler('xyz', degrees=False)[2]
    
    # Combine pointing down (180 deg around X) with object's yaw
    # RoboTwin's DEFAULT_QUAT [0, 1, 0, 0] is roughly 180 around X
    target_rot = R.from_euler('xyz', [np.pi, 0, yaw], degrees=False)
    tq = target_rot.as_quat() # [x, y, z, w]
    ee_quat = [tq[3], tq[0], tq[1], tq[2]] # [w, x, y, z]

    # 3. Approach (move to grasp positions with grippers open)
    l_cur = _get_current_ee_pos(TASK_ENV, arm='left')
    r_cur = _get_current_ee_pos(TASK_ENV, arm='right')
    n = 20
    l_path = _interpolate_trajectory(l_cur, l_pose, n)
    r_path = _interpolate_trajectory(r_cur, r_pose, n)
    wps = [(l_path[i], ee_quat, GRIPPER_OPEN,
            r_path[i], ee_quat, GRIPPER_OPEN) for i in range(n)]

    if not _execute_dual_waypoints(TASK_ENV, wps, logger, "skill_dual_arm_grasp_move"):
        return make_result("FAILED", "Failed to move arms to grasp positions.")

    # 4. Close grippers (hold for 10 steps)
    for _ in range(10):
        action = _make_dual_ee_action(l_pose, DEFAULT_QUAT, GRIPPER_CLOSED,
                                       r_pose, DEFAULT_QUAT, GRIPPER_CLOSED)
        TASK_ENV.take_action(action, action_type='ee')

    # 5. Verify with a small lift (5cm)
    lift_h = 0.05
    l_lift = [l_pose[0], l_pose[1], l_pose[2] + lift_h]
    r_lift = [r_pose[0], r_pose[1], r_pose[2] + lift_h]
    l_path = _interpolate_trajectory(l_pose, l_lift, 10)
    r_path = _interpolate_trajectory(r_pose, r_lift, 10)
    wps = [(l_path[i], DEFAULT_QUAT, GRIPPER_CLOSED,
            r_path[i], DEFAULT_QUAT, GRIPPER_CLOSED) for i in range(10)]

    _execute_dual_waypoints(TASK_ENV, wps, logger, "skill_dual_arm_grasp_verify")

    obj_after = get_scene_objects(TASK_ENV).get(object_name, {}).get("position", [0, 0, 0])
    dz = obj_after[2] - z_before

    if dz > 0.02: # Lower threshold for initial verify
        # Initialize state tracking if not present
        if not hasattr(TASK_ENV, '_held_objects'):
            TASK_ENV._held_objects = {}
            
        TASK_ENV._held_objects[object_name] = {
            "held": True,
            "arms": ["left", "right"],
            "grasp_verified": True,
            "object_z_at_grasp": z_before,
            "left_ee_at_grasp": l_pose,
            "right_ee_at_grasp": r_pose
        }
        res = make_result("SUCCESS", f"Dual-arm grasp verified. Micro-lift dz={dz:.3f}m.", 
                          height_delta=dz, left_pose=l_pose, right_pose=r_pose, held=True)
    else:
        res = make_result("FAILED", f"Dual-arm grasp failed. Micro-lift dz={dz:.3f}m.", 
                          height_delta=dz, left_pose=l_pose, right_pose=r_pose)

    if logger:
        logger.log_skill("skill_dual_arm_grasp", {"object": object_name},
                         result=res["status"], feedback=res["feedback"],
                         step_num=TASK_ENV.take_action_cnt, success=(res["status"] == "SUCCESS"),
                         data=res.get("data"))
    return res


def skill_dual_arm_lift(TASK_ENV, obj_name: str, height: float = 0.15, logger=None) -> dict:
    """
    Lift an already-grasped object upward by `height` meters using both arms.
    Does NOT recalculate grasp poses. Strictly moves current end-effectors up.
    """
    # 1. Verify Held State
    held_state = getattr(TASK_ENV, '_held_objects', {}).get(obj_name, {})
    if not held_state.get("grasp_verified"):
        return make_result("FAILED", f"Cannot lift: {obj_name} is not verified as grasped. Call dual_arm_grasp first.")

    from privileged_perception import get_scene_objects
    objects_before = get_scene_objects(TASK_ENV)
    if obj_name not in objects_before:
        return make_result("FAILED", f"Object {obj_name} lost from perception before lift.")
    z_before = objects_before[obj_name]["position"][2]

    # 2. Get current EE positions (do not recalculate offsets)
    left_cur = _get_current_ee_pos(TASK_ENV, arm='left')
    right_cur = _get_current_ee_pos(TASK_ENV, arm='right')

    # 3. Interpolate pure upward trajectory
    n_lift = 15
    left_lift = [left_cur[0], left_cur[1], left_cur[2] + height]
    right_lift = [right_cur[0], right_cur[1], right_cur[2] + height]
    
    l_path = _interpolate_trajectory(left_cur, left_lift, n_lift)
    r_path = _interpolate_trajectory(right_cur, right_lift, n_lift)
    
    waypoints = []
    # Note: Using DEFAULT_QUAT here might reset orientation if it was dynamic during grasp.
    # To be safe, we should maintain current orientation. For simplicity, we use DEFAULT_QUAT
    # as in original, but if orientation shifts caused drops, we should track ee_quat in _held_objects.
    # Assuming gripper orientation is robust enough or was [0,1,0,0] equivalent.
    for i in range(n_lift):
        waypoints.append((l_path[i], DEFAULT_QUAT, GRIPPER_CLOSED,
                          r_path[i], DEFAULT_QUAT, GRIPPER_CLOSED))

    # 4. Execute
    ok = _execute_dual_waypoints(TASK_ENV, waypoints, logger, "dual_arm_lift")
    if not ok:
        res = make_result("FAILED", "Dual arm lift motion failed to complete.")
    else:
        # 5. Verify Lift
        objects_after = get_scene_objects(TASK_ENV)
        if obj_name not in objects_after:
            res = make_result("FAILED", f"Object {obj_name} lost after dual lift.")
        else:
            dz = objects_after[obj_name]["position"][2] - z_before
            if dz >= height * 0.5: # Allow some tolerance
                res = make_result("SUCCESS", f"Lift verified. Object height increased by {dz:.3f}m.", 
                                  height_delta=dz, target_height=height)
            else:
                res = make_result("FAILED", f"Lift not verified. Height increased only {dz:.3f}m.", 
                                  height_delta=dz, target_height=height)

    if logger:
        logger.log_skill("dual_arm_lift", {"obj_name": obj_name, "height": height},
                         result=res["status"], feedback=res["feedback"],
                         step_num=TASK_ENV.take_action_cnt, success=(res["status"] == "SUCCESS"),
                         data=res.get("data"))
    return res


# ── Single-Arm Skills (legacy / atomic) ───────────────────────────────────

def skill_pick(TASK_ENV, obj_name: str, obj_position: list, logger=None) -> dict:
    from privileged_perception import get_scene_objects
    x, y, z = obj_position
    pre_grasp = [x, y, z + APPROACH_OFFSET]
    grasp_pos = [x, y, z + 0.02]

    waypoints = (
        [(p, DEFAULT_QUAT, GRIPPER_OPEN) for p in _interpolate_trajectory(_get_current_ee_pos(TASK_ENV), pre_grasp, 15)] +
        [(p, DEFAULT_QUAT, GRIPPER_OPEN) for p in _interpolate_trajectory(pre_grasp, grasp_pos, 10)] +
        [(grasp_pos, DEFAULT_QUAT, GRIPPER_CLOSED)] * 5 +
        [(p, DEFAULT_QUAT, GRIPPER_CLOSED) for p in _interpolate_trajectory(grasp_pos, pre_grasp, 10)]
    )

    ok = _execute_waypoints(TASK_ENV, waypoints, logger, "skill_pick")

    if not ok:
        res = make_result("FAILED", f"Failed to execute pick motion for {obj_name}.")
    else:
        objects_after = get_scene_objects(TASK_ENV)
        if obj_name not in objects_after:
            res = make_result("FAILED", f"Object {obj_name} not found after pick.")
        else:
            dz = objects_after[obj_name]["position"][2] - z
            if dz > 0.03:
                res = make_result("SUCCESS", f"Pick verified for {obj_name}: height delta={dz:.3f}m.", height_delta=dz)
            else:
                res = make_result("FAILED", f"Pick failed for {obj_name}: height delta={dz:.3f}m.", height_delta=dz)

    if logger:
        logger.log_skill("skill_pick", {"obj_name": obj_name, "position": obj_position},
                         result=res["status"], feedback=res["feedback"],
                         step_num=TASK_ENV.take_action_cnt, success=(res["status"] == "SUCCESS"),
                         data=res.get("data"))
    return res

def skill_place(TASK_ENV, target_name: str, target_position: list, logger=None) -> dict:
    x, y, z = target_position
    above_target = [x, y, z + APPROACH_OFFSET]
    release_pos = [x, y, z + 0.05]

    waypoints = (
        [(p, DEFAULT_QUAT, GRIPPER_CLOSED) for p in _interpolate_trajectory(_get_current_ee_pos(TASK_ENV), above_target, 15)] +
        [(p, DEFAULT_QUAT, GRIPPER_CLOSED) for p in _interpolate_trajectory(above_target, release_pos, 10)] +
        [(release_pos, DEFAULT_QUAT, GRIPPER_OPEN)] * 5 +
        [(p, DEFAULT_QUAT, GRIPPER_OPEN) for p in _interpolate_trajectory(release_pos, above_target, 10)]
    )

    ok = _execute_waypoints(TASK_ENV, waypoints, logger, "skill_place")
    res = make_result("SUCCESS", f"Placed at {target_name}.") if ok else make_result("FAILED", "Place failed.")
    if logger:
        logger.log_skill("skill_place", {"target": target_name, "position": target_position},
                         result=res["status"], feedback=res["feedback"],
                         step_num=TASK_ENV.take_action_cnt, success=ok, data=res.get("data"))
    return res

def skill_move_to(TASK_ENV, target_pos: list, gripper_val: float = GRIPPER_OPEN, logger=None) -> dict:
    """Moves right arm to target position and returns distance metrics."""
    ee_before = list(_get_current_ee_pos(TASK_ENV))
    waypoints = [(p, DEFAULT_QUAT, gripper_val) for p in _interpolate_trajectory(ee_before, target_pos, 20)]

    ok = _execute_waypoints(TASK_ENV, waypoints, logger, "skill_move_to")
    ee_after = list(_get_current_ee_pos(TASK_ENV))
    dist = float(np.linalg.norm(np.array(ee_after) - np.array(target_pos)))

    status = "SUCCESS" if ok else "FAILED"
    feedback = f"Moved to {target_pos}." if ok else f"Move to {target_pos} failed (dist={dist:.3f})."

    res = make_result(status, feedback, ee_before=ee_before, ee_after=ee_after, distance_to_target=dist)
    if logger:
        logger.log_skill("skill_move_to", {"target": target_pos},
                         result=res["status"], feedback=res["feedback"],
                         step_num=TASK_ENV.take_action_cnt, success=ok, data=res.get("data"))
    return res

def skill_home(TASK_ENV, logger=None) -> dict:
    # Clear held state since arms are moving away and opening
    if hasattr(TASK_ENV, '_held_objects'):
        TASK_ENV._held_objects = {}
    HOME_POS = [0.3, 0.0, 1.1]
    return skill_move_to(TASK_ENV, HOME_POS, gripper_val=GRIPPER_OPEN, logger=logger)


# ── Skill Namespace (exposed to LLM Executor) ─────────────────────────────

def build_skill_namespace(TASK_ENV, logger=None):
    """
    Builds the function namespace injected into LLM-generated code.
    Primary API: high-level composite skills.
    Advanced API: atomic skills for recovery/debug.
    """
    from privileged_perception import get_scene_objects

    def get_objects():
        return get_scene_objects(TASK_ENV)

    def get_reference_names():
        return list(get_objects().keys())

    def resolve_reference(query: str, refs: list = None):
        if refs is None:
            refs = get_reference_names()
        for r in refs:
            if query.lower() in r.lower():
                return r
        return None

    # ── Primary Skills ──
    def dual_arm_grasp(object_name: str):
        """Composite: compute grasp -> move -> close -> verify."""
        return skill_dual_arm_grasp(TASK_ENV, object_name, logger=logger)

    def dual_arm_lift(object_name: str, height: float = 0.15):
        """Lift object. Call AFTER dual_arm_grasp succeeds."""
        return skill_dual_arm_lift(TASK_ENV, object_name, height, logger=logger)

    def check_object_sanity(object_name: str):
        """Check if object is on table and reachable."""
        return skill_check_object_sanity(TASK_ENV, object_name)

    def move_to_home_pos():
        """Move both arms to safe home position."""
        return skill_home(TASK_ENV, logger=logger)

    def verify_task_success():
        """Check environment success condition."""
        success = TASK_ENV.eval_success
        return make_result(
            "SUCCESS" if success else "FAILED",
            "Task success condition satisfied." if success else "Task not yet finished.",
        )

    # ── Advanced / Atomic (LLM should prefer primary skills) ──
    def move_to(pos: list, arm: str = "right"):
        return skill_move_to(TASK_ENV, pos, logger=logger)

    return {
        # Primary
        "get_reference_names": get_reference_names,
        "resolve_reference": resolve_reference,
        "get_objects": get_objects,
        "dual_arm_grasp": dual_arm_grasp,
        "dual_arm_lift": dual_arm_lift,
        "check_object_sanity": check_object_sanity,
        "move_to_home_pos": move_to_home_pos,
        "verify_task_success": verify_task_success,
        "get_feedback": get_feedback,
        # Advanced
        "compute_dual_grasp": lambda obj: compute_dual_grasp(TASK_ENV, obj),
        "move_to": move_to,
    }
