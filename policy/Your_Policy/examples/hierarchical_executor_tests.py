"""
Pure-Python tests for hierarchical planner/executor boundaries.

Run:
    cd ~/Documents/GitHub/RoboTwin
    python policy/Your_Policy/examples/hierarchical_executor_tests.py
"""

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICY_DIR = _HERE.parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

import hierarchical_executor as he  # noqa: E402
from deploy_policy import HIERARCHICAL_PLANNER_SYSTEM, ALRMAgent  # noqa: E402


_RESULTS = []


def _record(name, ok, details=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}")
    if details:
        print(f"      {details}")
    _RESULTS.append((name, ok))
    return ok


class _FakeLLM:
    model = "fake"

    def call(self, system, user, temperature=None):
        return "[Final Answer]: done"


def case_planner_system_has_no_env_state():
    forbidden = ["position=", "end-effector", "contact_points", "scene_desc", "robot_state"]
    ok = all(token not in HIERARCHICAL_PLANNER_SYSTEM for token in forbidden)
    return _record("planner_system_has_no_env_state", ok, HIERARCHICAL_PLANNER_SYSTEM)


def case_hierarchical_plan_prompt_has_no_scene_dump():
    agent = ALRMAgent(_FakeLLM(), executor_mode="hierarchical_schema")
    agent.new_episode("lift_pot", "demo")
    action = agent._plan_hierarchical("Complete the lift pot task.")
    prompt = agent.logger.log["planner"]["prompt"]
    forbidden = ["Objects in scene", "060_kitchenpot", "position=", "Robot state"]
    ok = action.startswith("[Final Answer]") and all(token not in prompt for token in forbidden)
    return _record("hierarchical_plan_prompt_has_no_scene_dump", ok, prompt)


def case_find_action_resolves_object_without_llm():
    original = he.get_scene_objects
    he.get_scene_objects = lambda _env: {
        "060_kitchenpot": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]}
    }
    try:
        executor = he.HierarchicalSchemaExecutor(_FakeLLM())
        success, feedback = executor.execute("Find target object for: pot", object(), [])
        obs = json.loads(feedback)
        ok = success and obs["selected_object"] == "060_kitchenpot"
        return _record("find_action_resolves_object_without_llm", ok, feedback)
    finally:
        he.get_scene_objects = original


def case_parse_program_requires_program_key():
    executor = he.HierarchicalSchemaExecutor(_FakeLLM())
    program, err = executor._parse_program_response('{"not_program": []}')
    ok = program is None and "program" in err
    return _record("parse_program_requires_program_key", ok, err)


def case_parse_program_accepts_thought_action_shape():
    executor = he.HierarchicalSchemaExecutor(_FakeLLM())
    response = json.dumps({
        "thought": "Use the lift actor API and verify task success.",
        "action": {"program": [{"op": "is_task_success"}]},
    })
    program, err = executor._parse_program_response(response)
    ok = err is None and program == [{"op": "is_task_success"}]
    return _record("parse_program_accepts_thought_action_shape", ok, err)


def case_executor_prompt_describes_apis_not_order():
    executor = he.HierarchicalSchemaExecutor(_FakeLLM())
    prompt = executor._build_program_prompt(
        action="Execute robot subtask: Lift 060_kitchenpot with both arms",
        selected_object="060_kitchenpot",
        env_snapshot={
            "selected_object": "060_kitchenpot",
            "available_objects": ["060_kitchenpot"],
            "object_pose": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]},
            "robot_state": {},
            "contact_metadata": {},
        },
        planner_history=[],
        executor_attempts=[],
    )
    forbidden = [
        "required_content",
        "local_dependencies",
        "must happen before",
        "must happen after",
        "must be followed",
    ]
    ok = all(token not in prompt for token in forbidden)
    ok = ok and "primitive_api_docs" in prompt and "success_criteria" in prompt
    ok = ok and '"thought"' in prompt and '"action"' in prompt
    return _record("executor_prompt_describes_apis_not_order", ok, prompt[:1200])


def case_lift_subtask_requires_task_success():
    program = [
        {"op": "get_object_pose", "object_name": "060_kitchenpot", "save_as": "obj0"},
        {"op": "dual_grasp_actor", "object": "060_kitchenpot",
         "left_contact_point_id": 0, "right_contact_point_id": 1,
         "pre_grasp_dis": 0.035, "preclose_gripper_pos": 0.5, "gripper_pos": 0.0},
        {"op": "wait_steps", "n": 20},
        {"op": "move_both_delta", "left_delta": [0, 0, 0.1], "right_delta": [0, 0, 0.1]},
        {"op": "wait_steps", "n": 20},
        {"op": "is_lift_verified", "object_name": "060_kitchenpot",
         "z_before": "$obj0.data.position[2]", "min_dz": 0.05},
    ]
    err = he._validate_subtask_program(
        program,
        "Execute robot subtask: Lift 060_kitchenpot with both arms",
        "060_kitchenpot",
    )
    ok = err is not None and err.get("failure_type") == "executor_validation_error"
    return _record("lift_subtask_requires_task_success", ok, err)


def _valid_lift_program():
    return [
        {"op": "get_object_pose", "object_name": "060_kitchenpot", "save_as": "obj0"},
        {"op": "dual_grasp_actor", "object": "060_kitchenpot",
         "left_contact_point_id": 0, "right_contact_point_id": 1,
         "pre_grasp_dis": 0.035, "preclose_gripper_pos": 0.5, "gripper_pos": 0.0},
        {"op": "wait_steps", "n": 20},
        {"op": "move_both_delta", "left_delta": [0, 0, 0.1], "right_delta": [0, 0, 0.1]},
        {"op": "wait_steps", "n": 20},
        {"op": "is_lift_verified", "object_name": "060_kitchenpot",
         "z_before": "$obj0.data.position[2]", "min_dz": 0.05},
        {"op": "is_task_success"},
    ]


def case_full_args_lift_program_passes():
    err = he._validate_subtask_program(
        _valid_lift_program(),
        "Execute robot subtask: Lift 060_kitchenpot with both arms",
        "060_kitchenpot",
    )
    return _record("full_args_lift_program_passes", err is None, err)


def case_missing_grasp_config_fails():
    program = _valid_lift_program()
    del program[1]["pre_grasp_dis"]
    err = he._validate_subtask_program(
        program,
        "Execute robot subtask: Lift 060_kitchenpot with both arms",
        "060_kitchenpot",
    )
    ok = err and err.get("failure_type") == "executor_validation_error"
    return _record("missing_grasp_config_fails", ok, err)


def case_missing_two_waits_fails():
    program = [entry for entry in _valid_lift_program() if entry["op"] != "wait_steps"]
    err = he._validate_subtask_program(
        program,
        "Execute robot subtask: Lift 060_kitchenpot with both arms",
        "060_kitchenpot",
    )
    ok = err and err.get("failure_type") == "executor_validation_error"
    return _record("missing_two_waits_fails", ok, err)


def case_grasp_after_lift_fails():
    program = _valid_lift_program()
    program[1], program[3] = program[3], program[1]
    err = he._validate_subtask_program(
        program,
        "Execute robot subtask: Lift 060_kitchenpot with both arms",
        "060_kitchenpot",
    )
    ok = err and "before lift" in err.get("details", "")
    return _record("grasp_after_lift_fails", ok, err)


def case_verify_before_lift_fails():
    program = _valid_lift_program()
    program[3], program[5] = program[5], program[3]
    err = he._validate_subtask_program(
        program,
        "Execute robot subtask: Lift 060_kitchenpot with both arms",
        "060_kitchenpot",
    )
    ok = err and "after lift" in err.get("details", "")
    return _record("verify_before_lift_fails", ok, err)


def case_wait_position_flexible_passes():
    program = _valid_lift_program()
    program[4], program[5] = program[5], program[4]
    err = he._validate_subtask_program(
        program,
        "Execute robot subtask: Lift 060_kitchenpot with both arms",
        "060_kitchenpot",
    )
    return _record("wait_position_flexible_passes", err is None, err)


def main():
    cases = [
        case_planner_system_has_no_env_state,
        case_hierarchical_plan_prompt_has_no_scene_dump,
        case_find_action_resolves_object_without_llm,
        case_parse_program_requires_program_key,
        case_parse_program_accepts_thought_action_shape,
        case_executor_prompt_describes_apis_not_order,
        case_lift_subtask_requires_task_success,
        case_full_args_lift_program_passes,
        case_missing_grasp_config_fails,
        case_missing_two_waits_fails,
        case_grasp_after_lift_fails,
        case_verify_before_lift_fails,
        case_wait_position_flexible_passes,
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
    print(f"Hierarchical executor test summary: {passed}/{total} passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
