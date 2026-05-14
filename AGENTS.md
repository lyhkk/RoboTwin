# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.
Don't respond in Korean.

## Project Documentation Pointers

For detailed guidelines, please refer to the following documents:

- **[01_ENV_SETUP.md](file:///Users/lyhkk/Documents/GitHub/RoboTwin/docs/01_ENV_SETUP.md)**: Environment installation and dependency management.
- **[02_DEV_WORKFLOW.md](file:///Users/lyhkk/Documents/GitHub/RoboTwin/docs/02_DEV_WORKFLOW.md)**: **Standard Evaluation Workflow (Start Here for Testing).**
- **[03_AGENT_SPEC.md](file:///Users/lyhkk/Documents/GitHub/RoboTwin/docs/03_AGENT_SPEC.md)**: Agent architecture and ALRM compliance.
- **[04_FUTURE_ROADMAP.md](file:///Users/lyhkk/Documents/GitHub/RoboTwin/docs/04_FUTURE_ROADMAP.md)**: Vision and future milestones.

RoboTwin is a bimanual robotic manipulation benchmark and data generation platform built on [SAPIEN](https://sapien.ucsd.edu/). It supports 50+ tasks, 5+ robot embodiments, and 10+ policy baselines for training and evaluating dual-arm robot policies.

## Common Commands

### Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example:
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Policy Training (example: Diffusion Policy)

```bash
bash policy/DP/train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${action_dim} ${gpu_id}
```

### Policy Evaluation

```bash
python script/eval_policy.py --task-name ${task_name} --policy-name ${policy_name} --task-config ${task_config} --checkpoint ${ckpt_path}
```

### Remote Policy Inference (for large models)

```bash
# On inference server:
python script/policy_model_server.py --policy-name ${policy_name} --port 5000
# On eval client:
python script/eval_policy_client.py --task-name ${task_name} --server-url http://host:5000
```

### Task Code Generation (LLM-assisted)

```bash
python code_gen/task_generation.py --task-name ${task_name}
```

### Install core dependencies

```bash
pip install -r script/requirements.txt
```

## Architecture

### Key Paths (defined in `envs/_GLOBAL_CONFIGS.py`)

- `./assets/` — robot URDFs, meshes, textures
- `./task_config/` — YAML configs for tasks, cameras, embodiments
- `./data/` — collected trajectory data (zarr format)
- `./description/` — language instruction templates per task

### Task Environment Layer (`envs/`)

- `_base_task.py` — `Base_Task(gym.Env)`: all tasks inherit this. Manages scene setup, robot loading, camera setup, physics stepping, and data recording.
- Each task file (e.g., `envs/beat_block_hammer.py`) implements three methods:
  - `load_actors()` — place task-specific objects in scene
  - `play_once()` — scripted expert demonstration using motion planner

## Remote Environment Hygiene & Debugging Rules

To maintain simulation stability and prevent OpenGL/CUDA context corruption:

### 1. Environment Isolation

- **RoboTwin Execution**: Must always run inside the `robotwin` Conda environment.
- **VNC Services**: `x11vnc` and `websockify` should run in the `base` environment or a clean system shell to avoid library conflicts (e.g., `libffi` or `ncurses` ABI mismatches).

### 2. noVNC Debugging (Short-term Only)

- **Debugging Only**: noVNC is for quick visual inspections. Do not keep it open for long durations.
- **Process Count Rule**: Before starting a new noVNC session, verify that there is exactly **one** `x11vnc` and **one** `websockify` process. Kill any redundant or zombie processes.
- **Cleanup Command**: `pkill -u $USER -f x11vnc && pkill -u $USER -f websockify`.

### 3. Long-term Observation

- **Standard Practice**: Use Headless Mode (`export PYOPENGL_PLATFORM=egl`).
- **Data-Driven Review**: Rely on saved `.mp4` videos, JSON logs, and episode statistics for analysis rather than live viewing.

### Robot Control (`envs/robot/`)

- `robot.py` — `Robot` class: loads URDF, wraps dual-arm kinematics
- `planner.py` — two planners:
  - `MPlibPlanner`: RRT-based (mplib)
  - `CuroboPlanner`: differentiable collision-free (cuRobo, GPU-required)
- `ik.py` — inverse kinematics utilities

### Configuration System (`task_config/`)

- `_config_template.yml` — canonical reference for all config parameters
- `demo_randomized.yml` / `demo_clean.yml` — standard task configs
- `_embodiment_config.yml` — maps embodiment names to URDF paths
- `_camera_config.yml` — camera sensor specs (D435, RealSense, etc.)
- `_eval_step_limit.yml` — per-task max steps for evaluation

### Data Pipeline

1. **Collect**: `script/collect_data.py` runs episodes, saves zarr to `data/{task}/{config}/`
2. **Preprocess**: `script/process_data.py` normalizes and packages for training
3. **Train**: Policy-specific scripts under `policy/{POLICY}/train.sh`
4. **Evaluate**: `script/eval_policy.py` loads checkpoint, rolls out in sim

### Policy Interface (all policies follow this contract)

Each `policy/{POLICY}/deploy_policy.py` exposes:

```python
def get_model(usr_args): ...    # load checkpoint
def encode_obs(observation): ... # normalize inputs
def eval(TASK_ENV, model, obs): ... # inference loop
def reset_model(model): ...      # reset recurrent state
```

### Observation Structure

```python
{
  "observation": {
    "head_camera": {"rgb": (H,W,3)},
    "left_camera": {"rgb": (H,W,3)},
    "right_camera": {"rgb": (H,W,3)},
    # optional: depth, segmentation, point_cloud
  },
  "joint_action": {
    "vector": [left_qpos(6), right_qpos(6), left_gripper(1), right_gripper(1)]
  },
  "instruction": "language description"
}
```

### Supported Embodiments

`aloha-agilex`, `piper`, `franka-panda`, `ARX-X5`, `ur5-wsg` — configured via `_embodiment_config.yml`

### Supported Policies

`DP`, `ACT`, `DP3`, `RDT`, `pi0`, `pi05`, `openvla-oft`, `TinyVLA`, `DexVLA`, `LLaVA-VLA`, `GO1`

## Adding a New Task

1. Create `envs/{task_name}.py` inheriting `Base_Task`
2. Implement `load_actors()`, `play_once()`, `check_success()`
3. Add language instructions to `description/{task_name}.json`
4. Add eval step limit to `task_config/_eval_step_limit.yml`
5. Optionally use `code_gen/task_generation.py` for LLM-assisted scaffolding

## Domain Randomization

Controlled via task config YAML (`domain_randomization` key). Key knobs:

- `background_type`: texture variation
- `table_height_variation`: ±0.03m
- `head_camera_distance_variation`: camera pose jitter
- `lighting`: random intensity range
- `cluttered_table`: add distractor objects

Set `demo_clean.yml` for no randomization baseline.
