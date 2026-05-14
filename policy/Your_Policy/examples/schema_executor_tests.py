"""
Pure-Python tests for the schema executor.

Run:
    cd ~/Documents/GitHub/RoboTwin
    python policy/Your_Policy/examples/schema_executor_tests.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICY_DIR = _HERE.parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from schema_executor import (  # noqa: E402
    FAILED,
    SUCCESS,
    compile_dual_arm_lift_actor,
    _is_expert_unreachable_seed,
    _make_expert_unreachable_feedback,
    parse_json_response,
    validate_lift_schema,
)


_RESULTS = []


def _record(name, ok, details=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}")
    if details:
        print(f"      {details}")
    _RESULTS.append((name, ok))
    return ok


def _base_schema():
    return {
        "template": "dual_arm_lift_actor",
        "object": "060_kitchenpot",
        "left_contact_point_id": 0,
        "right_contact_point_id": 1,
        "preclose_gripper_pos": 0.5,
        "grasp_gripper_pos": 0.0,
        "pre_grasp_dis": 0.035,
        "lift_delta": [0.0, 0.0, 0.10],
        "wait_after_grasp": 20,
        "wait_after_lift": 20,
        "verify_min_dz": 0.05,
        "move_to_home_first": False,
    }


def case_parse_json():
    parsed = parse_json_response('```json\n{"template": "dual_arm_lift_actor"}\n```')
    return _record("parse_json_response_strips_fence",
                   parsed["template"] == "dual_arm_lift_actor")


def case_parse_rejects_python():
    try:
        parse_json_response('result = True')
    except ValueError:
        return _record("parse_json_response_rejects_python", True)
    return _record("parse_json_response_rejects_python", False)


def case_valid_schema_compiles():
    schema = _base_schema()
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    if r["status"] != SUCCESS:
        return _record("valid_schema_compiles", False, r["details"])
    program = compile_dual_arm_lift_actor(r["data"]["schema"])
    ok = (
        len(program) == 7 and
        [op["op"] for op in program] == [
            "get_object_pose",
            "dual_grasp_actor",
            "wait_steps",
            "move_both_delta",
            "wait_steps",
            "is_lift_verified",
            "is_task_success",
        ] and
        program[0]["save_as"] == "obj0" and
        program[5]["z_before"] == "$obj0.data.position[2]" and
        program[3]["left_delta"] == program[3]["right_delta"]
    )
    return _record("valid_schema_compiles", ok, str(program) if not ok else "")


def case_invalid_object_rejected():
    schema = _base_schema()
    schema["object"] = "missing"
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    return _record("invalid_object_rejected", r["status"] == FAILED, r["details"])


def case_bad_contact_type_rejected():
    schema = _base_schema()
    schema["left_contact_point_id"] = "0"
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    return _record("bad_contact_type_rejected", r["status"] == FAILED, r["details"])


def case_duplicate_contact_points_rejected():
    schema = _base_schema()
    schema["right_contact_point_id"] = schema["left_contact_point_id"]
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    ok = r["status"] == FAILED and "must be different" in r["details"]
    return _record("duplicate_contact_points_rejected", ok, r["details"])


def case_oversized_lift_rejected():
    schema = _base_schema()
    schema["lift_delta"] = [0.0, 0.0, 0.30]
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    return _record("oversized_lift_rejected", r["status"] == FAILED, r["details"])


def case_bad_wait_rejected():
    schema = _base_schema()
    schema["wait_after_grasp"] = 99
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    return _record("bad_wait_rejected", r["status"] == FAILED, r["details"])


def case_missing_required_field_rejected():
    schema = _base_schema()
    del schema["move_to_home_first"]
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    ok = r["status"] == FAILED and "move_to_home_first" in r["details"]
    return _record("missing_required_field_rejected", ok, r["details"])


def case_move_home_prefix():
    schema = _base_schema()
    schema["move_to_home_first"] = True
    r = validate_lift_schema(schema, refs=["060_kitchenpot"])
    if r["status"] != SUCCESS:
        return _record("move_home_prefix", False, r["details"])
    program = compile_dual_arm_lift_actor(r["data"]["schema"])
    return _record("move_home_prefix", program[0]["op"] == "move_to_home")


def case_expert_unreachable_feedback_structured():
    result = {
        "status": "FAILED",
        "details": "Sequence aborted.",
        "data": {
            "failed_op_index": 1,
            "primitive_results": [
                {"details": "ok"},
                {"details": "Motion planner found no valid path. The expert play_once() would also fail on this seed."},
            ],
        },
    }
    feedback = _make_expert_unreachable_feedback(result)
    ok = _is_expert_unreachable_seed(result) and '"reason": "expert_unreachable_seed"' in feedback
    return _record("expert_unreachable_feedback_structured", ok, feedback)


def main():
    cases = [
        case_parse_json,
        case_parse_rejects_python,
        case_valid_schema_compiles,
        case_invalid_object_rejected,
        case_bad_contact_type_rejected,
        case_duplicate_contact_points_rejected,
        case_oversized_lift_rejected,
        case_bad_wait_rejected,
        case_missing_required_field_rejected,
        case_move_home_prefix,
        case_expert_unreachable_feedback_structured,
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
    print(f"Schema executor test summary: {passed}/{total} passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
