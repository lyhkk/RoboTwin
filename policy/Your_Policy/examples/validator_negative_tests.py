"""
Validator negative tests (Phase 2B).

Runs the program validator against a battery of intentionally-broken
programs and asserts that each is rejected with the right validator rule.
No live env / no sapien required — this is pure Python.

Run:
    cd ~/Documents/GitHub/RoboTwin
    python policy/Your_Policy/examples/validator_negative_tests.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICY_DIR = _HERE.parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from primitives.program_validator import validate_program  # noqa: E402


# ── Test runner ──────────────────────────────────────────────────────────

_RESULTS = []


def _check(name: str, program, expected_rule: str = None,
           refs=None, expected_status: str = "FAILED") -> bool:
    """Run validator. Print outcome. Return True iff result matches."""
    r = validate_program(program, refs=refs)
    got_status = r.get("status")
    got_rule = r.get("data", {}).get("validator_rule")
    ok_status = (got_status == expected_status)
    ok_rule = (expected_rule is None) or (got_rule == expected_rule)
    ok = ok_status and ok_rule
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}")
    print(f"        expected status={expected_status} "
          f"rule={expected_rule}")
    print(f"        got      status={got_status} rule={got_rule}")
    print(f"        details: {r.get('details')}")
    _RESULTS.append((name, ok))
    return ok


# ── Cases ────────────────────────────────────────────────────────────────

def case_unknown_op():
    program = [
        {"op": "teleport_robot", "args": {"to": "moon"}},
    ]
    return _check("unknown_op_rejected", program, expected_rule="V3")


def case_too_large_delta():
    program = [
        {"op": "move_delta", "arm": "right", "delta": [0.5, 0.0, 0.0]},
    ]
    return _check("too_large_delta_rejected", program, expected_rule="V5")


def case_too_many_steps():
    program = [{"op": "wait_steps", "n": 1}] * 50
    return _check("too_many_steps_rejected", program, expected_rule="V2")


def case_wait_too_large():
    program = [{"op": "wait_steps", "n": 500}]
    return _check("wait_too_large_rejected", program, expected_rule="V4")


def case_missing_verification_after_lift():
    # close_gripper followed by positive-z move_both_delta but no verification.
    program = [
        {"op": "close_gripper", "arm": "left"},
        {"op": "close_gripper", "arm": "right"},
        {"op": "move_both_delta",
         "left_delta": [0, 0, 0.12], "right_delta": [0, 0, 0.12]},
    ]
    # Both V8 and V9 trigger; whichever runs first is fine — both indicate
    # the same family of error.
    r = validate_program(program)
    rule = r.get("data", {}).get("validator_rule")
    ok = r.get("status") == "FAILED" and rule in ("V8", "V9")
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] missing_verification_after_lift_rejected")
    print(f"        expected V8 or V9, got {rule}")
    print(f"        details: {r.get('details')}")
    _RESULTS.append(("missing_verification_after_lift_rejected", ok))
    return ok


def case_forbidden_op_name():
    program = [
        {"op": "import", "args": {"module": "os"}},
    ]
    return _check("forbidden_op_name_rejected", program, expected_rule="V3")


def case_forbidden_substring_in_args():
    program = [
        {"op": "get_object_pose", "object_name": "TASK_ENV.scene"},
    ]
    return _check("forbidden_substring_rejected", program, expected_rule="V10")


def case_invalid_variable_reference():
    program = [
        # Forward reference: "grasp" not yet defined.
        {"op": "move_to_pose", "arm": "right", "pose": "$grasp.data.left_pose"},
    ]
    return _check("invalid_variable_reference_rejected",
                  program, expected_rule="V7")


def case_unsupported_op():
    # compute_grasp is on the allowlist but UNSUPPORTED; should fail with V11.
    program = [
        {"op": "compute_grasp", "object": "060_kitchenpot"},
    ]
    return _check("unsupported_op_rejected", program, expected_rule="V11")


def case_duplicate_dual_grasp_contacts():
    program = [
        {
            "op": "dual_grasp_actor",
            "object": "060_kitchenpot",
            "left_contact_point_id": 0,
            "right_contact_point_id": 0,
        },
    ]
    return _check("duplicate_dual_grasp_contacts_rejected",
                  program, expected_rule="V12",
                  refs=["060_kitchenpot"])


def case_get_object_pose_missing_object_name():
    program = [
        {"op": "get_object_pose", "save_as": "obj0"},
    ]
    return _check("get_object_pose_missing_object_name_rejected",
                  program, expected_rule="V12")


def case_save_as_reference_name():
    program = [
        {"op": "get_object_pose", "object_name": "060_kitchenpot", "save_as": "$obj0"},
    ]
    return _check("save_as_reference_name_rejected",
                  program, expected_rule="V1")


def case_is_lift_verified_unknown_arg():
    program = [
        {"op": "get_object_pose", "object_name": "060_kitchenpot", "save_as": "obj0"},
        {"op": "is_lift_verified",
         "object_name": "060_kitchenpot",
         "z_before": "$obj0.data.position[2]",
         "verify_min_dz": 0.05},
    ]
    return _check("is_lift_verified_unknown_arg_rejected",
                  program, expected_rule="V12")


def case_is_lift_verified_missing_required_arg():
    program = [
        {"op": "get_object_pose", "object_name": "060_kitchenpot", "save_as": "obj0"},
        {"op": "is_lift_verified",
         "object_name": "060_kitchenpot",
         "min_dz": 0.05},
    ]
    return _check("is_lift_verified_missing_required_arg_rejected",
                  program, expected_rule="V12")


def case_workspace_out_of_bounds():
    program = [
        {"op": "move_to_pose", "arm": "right",
         "pose": [3.0, 0.0, 1.0, 1, 0, 0, 0]},
    ]
    return _check("workspace_out_of_bounds_rejected", program, expected_rule="V6")


# ── Valid programs that should pass ──────────────────────────────────────

def case_v1_none_program():
    return _check("v1_none_program_rejected", None, expected_rule="V1")


def case_v1_string_program():
    return _check("v1_string_program_rejected", "not a list", expected_rule="V1")


def case_v1_int_program():
    return _check("v1_int_program_rejected", 42, expected_rule="V1")


def case_v1_list_of_non_dicts():
    return _check("v1_list_of_non_dicts_rejected", ["a", 1, None], expected_rule="V1")


def case_v1_entry_missing_op():
    program = [{"arm": "left", "pos": 0.5}]
    return _check("v1_entry_missing_op_rejected", program, expected_rule="V1")


# ── Dedicated forbidden-substring tests (V10) ──────────────────────────

def _case_forbidden(token: str):
    """Generate a test case for one FORBIDDEN_SUBSTRINGS token."""
    program = [
        {"op": "get_object_pose", "object_name": f"x{token}x", "save_as": "a"},
    ]
    name = f"forbidden_substring_{token.replace('(', '_paren').replace('.', '_dot')}"
    return _check(name, program, expected_rule="V10")


def case_forbidden_TASK_ENV():
    return _case_forbidden("TASK_ENV")


def case_forbidden_env_move():
    return _case_forbidden("env.move")


def case_forbidden_take_action():
    return _case_forbidden("take_action")


def case_forbidden_scene_step():
    return _case_forbidden("scene.step")


def case_forbidden___import__():
    return _case_forbidden("__import__")


def case_forbidden_eval():
    return _case_forbidden("eval(")


def case_forbidden_exec():
    return _case_forbidden("exec(")


def case_forbidden_os_system():
    return _case_forbidden("os.system")


def case_forbidden_subprocess():
    return _case_forbidden("subprocess")


def case_forbidden_socket():
    return _case_forbidden("socket.")


def case_forbidden_raw_action():
    return _case_forbidden("raw_action")


# ── Valid programs that should pass ──────────────────────────────────────

def case_valid_simple():
    program = [
        {"op": "get_object_pose", "object_name": "060_kitchenpot", "save_as": "obj"},
        {"op": "wait_steps", "n": 5},
        {"op": "is_task_success"},
    ]
    return _check("valid_simple_passes", program,
                  expected_rule=None, expected_status="SUCCESS")


def case_valid_lift_with_verification():
    program = [
        {"op": "get_object_pose", "object_name": "060_kitchenpot", "save_as": "obj0"},
        {"op": "close_gripper", "arm": "left", "pos": 0.5},
        {"op": "close_gripper", "arm": "right", "pos": 0.5},
        {"op": "move_both_delta",
         "left_delta": [0, 0, 0.10], "right_delta": [0, 0, 0.10]},
        {"op": "wait_steps", "n": 10},
        {"op": "is_lift_verified",
         "object_name": "060_kitchenpot",
         "z_before": "$obj0.data.position[2]",
         "min_dz": 0.05},
        {"op": "is_task_success"},
    ]
    return _check("valid_lift_with_verification_passes", program,
                  expected_rule=None, expected_status="SUCCESS")


def case_dual_grasp_actor_valid():
    """dual_grasp_actor in allowlist, followed by lift + verification = OK."""
    return _check("dual_grasp_actor_valid", [
        {"op": "get_object_pose", "object_name": "pot", "save_as": "obj0"},
        {"op": "dual_grasp_actor", "object": "pot",
         "left_contact_point_id": 0, "right_contact_point_id": 1},
        {"op": "wait_steps", "n": 20},
        {"op": "move_both_delta", "left_delta": [0,0,0.1], "right_delta": [0,0,0.1]},
        {"op": "is_lift_verified", "object_name": "pot",
         "z_before": "$obj0.data.position[2]", "min_dz": 0.05},
    ], expected_status="SUCCESS")

def case_v8_dual_grasp_then_lift_no_verify():
    """V8: dual_grasp_actor + lift without verification → FAILED."""
    return _check("v8_dual_grasp_lift_no_verify", [
        {"op": "dual_grasp_actor", "object": "pot",
         "left_contact_point_id": 0, "right_contact_point_id": 1},
        {"op": "move_both_delta", "left_delta": [0,0,0.1], "right_delta": [0,0,0.1]},
    ], expected_rule="V8")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    cases = [
        # Original tests
        case_unknown_op,
        case_too_large_delta,
        case_too_many_steps,
        case_wait_too_large,
        case_missing_verification_after_lift,
        case_forbidden_op_name,
        case_forbidden_substring_in_args,
        case_invalid_variable_reference,
        case_unsupported_op,
        case_duplicate_dual_grasp_contacts,
        case_get_object_pose_missing_object_name,
        case_save_as_reference_name,
        case_is_lift_verified_unknown_arg,
        case_is_lift_verified_missing_required_arg,
        case_workspace_out_of_bounds,
        # V1 malformed-program tests
        case_v1_none_program,
        case_v1_string_program,
        case_v1_int_program,
        case_v1_list_of_non_dicts,
        case_v1_entry_missing_op,
        # V10 all forbidden substrings
        case_forbidden_TASK_ENV,
        case_forbidden_env_move,
        case_forbidden_take_action,
        case_forbidden_scene_step,
        case_forbidden___import__,
        case_forbidden_eval,
        case_forbidden_exec,
        case_forbidden_os_system,
        case_forbidden_subprocess,
        case_forbidden_socket,
        case_forbidden_raw_action,
        # Valid programs
        case_valid_simple,
        case_valid_lift_with_verification,
        case_dual_grasp_actor_valid,
        case_v8_dual_grasp_then_lift_no_verify,
    ]
    for c in cases:
        try:
            c()
        except Exception as e:
            print(f"[FAIL] {c.__name__} raised: {e}")
            _RESULTS.append((c.__name__, False))
        print()

    n_pass = sum(1 for _, ok in _RESULTS if ok)
    n_total = len(_RESULTS)
    print("=" * 60)
    print(f"Validator negative-test summary: {n_pass}/{n_total} passed")
    print("=" * 60)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
