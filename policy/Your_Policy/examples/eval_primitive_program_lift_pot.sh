#!/bin/bash
# Phase 2B no-LLM smoke test for the primitive-program runtime.
#
# Usage:
#   bash policy/Your_Policy/examples/eval_primitive_program_lift_pot.sh [task_config] [seed] [gpu_id]
set -e

task_name=lift_pot
task_config=${1:-demo_clean}
ckpt_setting=primitive_program_smoke
seed=${2:-0}
gpu_id=${3:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${ROBOTWIN_ROOT}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYOPENGL_PLATFORM=egl

echo -e "\033[33m[Phase 2B Smoke] Task: ${task_name} | Config: ${task_config} | seed=${seed} | GPU=${gpu_id}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config policy/Your_Policy/examples/primitive_program_lift_pot.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name Your_Policy.examples.primitive_program_lift_pot
