#!/bin/bash
# LLM Agent (ALRM-CaP) Evaluation Script
# Usage: bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
# Example: bash eval.sh grab_roller arx-x5_randomized_500 phase1 0 0

policy_name=Your_Policy
task_name=${1}
task_config=${2}
ckpt_setting=${3:-phase1}
seed=${4:-0}
gpu_id=${5:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYOPENGL_PLATFORM=egl  # headless GPU rendering

echo -e "\033[33m[LLM Agent] Task: ${task_name} | Config: ${task_config} | GPU: ${gpu_id}\033[0m"
echo -e "\033[33m[LLM Agent] Logs → policy/Your_Policy/logs/\033[0m"


# Resolve paths relative to this script, not the caller's cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROBOTWIN_ROOT}" || { echo "[ERROR] Cannot cd to RoboTwin root"; exit 1; }

echo -e "\033[33m[LLM Agent] Root: ${ROBOTWIN_ROOT}\033[0m"

# Ensure .env exists
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    echo "[ERROR] ${SCRIPT_DIR}/.env not found!"
    echo "Copy policy/Your_Policy/.env.example to .env and fill in QWEN_API_KEY"
    exit 1
fi

# Install dependencies if needed
pip install python-dotenv openai -q

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}

echo -e "\033[32m[Done] Logs saved to policy/Your_Policy/logs/\033[0m"
echo "To sync logs locally, run:"
echo "  rsync -avP ubuntu:~/RoboTwin-release/policy/Your_Policy/logs/ ./policy/Your_Policy/logs/"
