"""
Atomic primitive layer for RoboTwin Your_Policy (Phase 2A).

Primitives are internal building blocks composed by safe skills. They are NOT
exposed to the LLM Executor in Phase 2A — only safe skills and the existing
Phase 1 API remain LLM-facing.

Layer responsibility (per docs/03_AGENT_SPEC.md and PHASE2_ATOMIC_PRIMITIVES_DESIGN.md):

    safe skill                 ← Phase 1 API (skill_library.py) + new skills/
       │
       ▼
    primitive                  ← THIS package
       │  motion / perception / gripper / verification / sequence
       ▼
    env.move(env.move_to_pose(...))  ← controller via RoboTwin Action system

Every motion primitive resets `TASK_ENV.plan_success = True` before calling
`env.move()`. Pose convention used everywhere here is **SAPIEN wxyz**:
[x, y, z, qw, qx, qy, qz]. See `pose_utils.py` for conversion helpers.

Imports are kept shallow so that `from primitives import pose_utils` does
not pull in the rest of the env stack. Submodules that need RoboTwin's env
import their dependencies lazily.
"""

# Lightweight modules — safe to import unconditionally.
from .result import (
    SUCCESS, FAILED,
    STAGE_MOTION, STAGE_GRASP_VERIFICATION, STAGE_TASK_VERIFICATION,
    make_primitive_result, make_skill_result, get_primitive_feedback,
    is_success,
)
from . import pose_utils

# Heavier modules (perception/motion/gripper/verification/sequence) are NOT
# re-exported here to avoid eagerly importing envs.utils.action and pulling
# in sapien. Callers should `from primitives.motion import ...` etc.

__all__ = [
    "SUCCESS", "FAILED",
    "STAGE_MOTION", "STAGE_GRASP_VERIFICATION", "STAGE_TASK_VERIFICATION",
    "make_primitive_result", "make_skill_result",
    "get_primitive_feedback", "is_success",
    "pose_utils",
]
