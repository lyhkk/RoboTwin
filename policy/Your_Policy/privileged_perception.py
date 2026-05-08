"""
Privileged Perception Module (Phase 1 - Validation Only)
Reads object poses directly from RoboTwin's Sapien simulation.
No vision required — used to validate LLM planning before adding real perception.

Replace this module with SAM2 + depth-based perception in Phase 2.
"""

import numpy as np
from typing import Dict, List, Optional


# Names to skip — infrastructure, not task objects
_SKIP_NAMES = {"table", "wall", "ground", "robot"}


def get_scene_objects(TASK_ENV) -> Dict[str, dict]:
    """
    Returns dict of all task-relevant objects with their poses.

    Returns:
        {
            "roller_0": {
                "position": [x, y, z],          # meters, world frame
                "orientation": [qw, qx, qy, qz], # quaternion
                "size_approx": [dx, dy, dz]      # rough bounding box estimate
            },
            ...
        }
    """
    objects = {}
    try:
        # Get simple actors
        for actor in TASK_ENV.scene.get_all_actors():
            name = actor.get_name()
            if not name or any(skip in name.lower() for skip in _SKIP_NAMES):
                continue

            pose = actor.get_pose()
            p = pose.p
            q = pose.q

            objects[name] = {
                "position": [round(float(p[0]), 4),
                              round(float(p[1]), 4),
                              round(float(p[2]), 4)],
                "orientation": [round(float(q[0]), 4),
                                 round(float(q[1]), 4),
                                 round(float(q[2]), 4),
                                 round(float(q[3]), 4)],
            }
        
        # Get articulations (important for many task objects)
        for art in TASK_ENV.scene.get_all_articulations():
            name = art.get_name()
            if not name or any(skip in name.lower() for skip in _SKIP_NAMES):
                continue

            # Root pose
            pose = art.get_root_pose()
            objects[name] = {
                "position": [round(float(pose.p[0]), 4),
                               round(float(pose.p[1]), 4),
                               round(float(pose.p[2]), 4)],
                "orientation": [round(float(pose.q[0]), 4),
                                 round(float(pose.q[1]), 4),
                                 round(float(pose.q[2]), 4),
                                 round(float(pose.q[3]), 4)],
                "is_articulation": True
            }
            
            # Sub-components (links) - like handles
            for link in art.get_links():
                link_name = link.get_name()
                if "handle" in link_name.lower():
                    link_pose = link.get_pose()
                    objects[f"{name}_{link_name}"] = {
                        "position": [round(float(link_pose.p[0]), 4),
                                       round(float(link_pose.p[1]), 4),
                                       round(float(link_pose.p[2]), 4)],
                        "orientation": [round(float(link_pose.q[0]), 4),
                                         round(float(link_pose.q[1]), 4),
                                         round(float(link_pose.q[2]), 4),
                                         round(float(link_pose.q[3]), 4)],
                    }
    except Exception as e:
        print(f"[Perception] Warning: failed to read scene objects: {e}")

    return objects


def get_robot_state(TASK_ENV) -> dict:
    """
    Returns current robot end-effector poses and gripper states.
    """
    try:
        obs = TASK_ENV.now_obs if hasattr(TASK_ENV, 'now_obs') and TASK_ENV.now_obs else {}
        endpose = obs.get("endpose", {})

        left_ee = endpose.get("left_endpose", None)
        right_ee = endpose.get("right_endpose", None)
        left_grip = endpose.get("left_gripper", None)
        right_grip = endpose.get("right_gripper", None)

        return {
            "left_arm": {
                "endpose": [round(float(v), 4) for v in left_ee] if left_ee is not None else None,
                "gripper_val": round(float(left_grip), 4) if left_grip is not None else None,
            },
            "right_arm": {
                "endpose": [round(float(v), 4) for v in right_ee] if right_ee is not None else None,
                "gripper_val": round(float(right_grip), 4) if right_grip is not None else None,
            }
        }
    except Exception as e:
        print(f"[Perception] Warning: failed to read robot state: {e}")
        return {}


def build_scene_description(TASK_ENV) -> str:
    """
    Builds a natural-language scene description for the LLM prompt.

    Example output:
        Objects in scene:
        - roller_0: position=[0.12, -0.05, 0.82]
        - tray_1: position=[0.30, 0.10, 0.80]

        Robot state:
        - Right arm end-effector: [0.25, 0.00, 1.05, ...], gripper: open (0.04)
    """
    objects = get_scene_objects(TASK_ENV)
    robot = get_robot_state(TASK_ENV)

    lines = ["Objects in scene:"]
    if objects:
        for name, info in objects.items():
            pos = info["position"]
            lines.append(f"  - {name}: position={pos}")
    else:
        lines.append("  (no objects detected)")

    lines.append("\nRobot state:")
    for arm in ["left_arm", "right_arm"]:
        arm_data = robot.get(arm, {})
        ee = arm_data.get("endpose")
        grip = arm_data.get("gripper_val")
        grip_str = f"open ({grip:.3f})" if grip is not None and grip > 0.01 else "closed"
        ee_str = str(ee[:3]) if ee else "unknown"
        lines.append(f"  - {arm}: end-effector pos={ee_str}, gripper={grip_str}")

    return "\n".join(lines)


def find_object_by_keyword(TASK_ENV, keyword: str) -> Optional[tuple]:
    """
    Finds an object by partial name match. Returns (name, position) or None.

    Used by skill functions when LLM provides a semantic object name like "roller".
    """
    objects = get_scene_objects(TASK_ENV)
    keyword_lower = keyword.lower()

    # Exact match first
    if keyword in objects:
        return keyword, objects[keyword]["position"]

    # Partial match
    for name, info in objects.items():
        if keyword_lower in name.lower():
            return name, info["position"]

    return None
