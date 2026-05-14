"""
Safe skills composed from primitives (Phase 2A).

These skills are intended to be called by safe-skill orchestration code
or by examples/primitive_lift_pot.py. They are NOT yet wired into
build_skill_namespace — Phase 1 skills remain the LLM-facing API.
"""

from .dual_arm_lift import dual_arm_lift_with_primitives

__all__ = ["dual_arm_lift_with_primitives"]
