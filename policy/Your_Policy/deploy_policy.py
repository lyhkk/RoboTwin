"""
ALRM-Style LLM Agent for RoboTwin (Phase 1: Privileged Perception + CaP)
==========================================================================

Architecture (from ALRM paper):
  Task Planner (ReAct) → Task Executor (CaP) → Skill Library → TASK_ENV

API: Qwen via OpenAI-compatible endpoint
Mode: CaP (Code-as-Policy) — LLM generates Python, we exec() it

To run:
  cd ~/RoboTwin-release/policy/Your_Policy
  cp .env.example .env
  # manually fill QWEN_API_KEY in .env
  cd ../..
  PYOPENGL_PLATFORM=egl bash policy/Your_Policy/eval.sh grab_roller arx-x5_randomized_500 phase1 0 0

Logs: policy/Your_Policy/logs/  (rsync to local for inspection)
"""

import os
import sys
import time
import traceback
from pathlib import Path

# ── ensure local imports work ────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# ── load .env (API key never appears in logs) ─────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
except ImportError:
    print("[Warning] python-dotenv not installed. Reading env vars directly.")

from openai import OpenAI

from llm_logger import EpisodeLogger, RunSummaryLogger
from privileged_perception import build_scene_description, get_scene_objects
from hierarchical_executor import HierarchicalSchemaExecutor
from schema_executor import SchemaExecutor
from skill_library import build_skill_namespace


# ── Config (fallback defaults — overridden by YAML config via get_model) ──
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.6-plus")


# ── Prompts ───────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are the high-level Task Planner Agent for a robot manipulation system.

You operate in a ReAct loop:
Thought -> Action -> Observation -> Thought -> Action -> ...

Rules:
1. Produce exactly one semantic action per turn.
2. Before any manipulation, use check_object_sanity(object_name) to ensure the object is reachable.
3. For dual-arm lift tasks (like lift_pot), produce one semantic action:
   "Lift <object_name> with both arms." The executor chooses the concrete
   implementation.
4. If an observation reports failure (e.g., grasp failed), choose a recovery action (e.g., reset home, or try a different approach).
5. If check_object_sanity reports the object has fallen, use [Final Answer] to report failure.

Output format:
[Thought]: ...
[Action]: ...
"""

SCHEMA_PLANNER_SYSTEM = """You are the high-level Task Planner Agent for a schema-based RoboTwin executor.

You operate in a ReAct loop:
Thought -> Action -> Observation -> Thought -> Action -> ...

Rules:
1. Produce exactly one semantic action per turn.
2. For lift_pot, if the scene contains 060_kitchenpot and there is no prior failure, your first action must be:
   Lift 060_kitchenpot with both arms.
3. Do not ask for object sanity checks before the first lift action in schema mode; the executor receives privileged scene metadata and validates the schema.
4. If an observation reports failure, choose one semantic recovery action based on that observation.
5. If repeated recovery fails or the object is no longer usable, use [Final Answer] to report failure.

Output format:
[Thought]: ...
[Action]: ...
"""

HIERARCHICAL_PLANNER_SYSTEM = """You are the high-level task Planner for a robot system.

You receive:
- the original user goal
- a history of completed Executor subtasks
- whether each subtask succeeded or failed

You do not see robot environment state directly. Do not ask for object poses,
robot state, contact points, or scene dumps. Only the Executor can inspect the
environment and execute robot actions.

Your job:
1. Understand the original user goal.
2. Decompose it into high-level subtasks.
3. After every Executor observation, decide whether the original goal is fully
   complete or whether another subtask is needed.

Allowed actions:
- Find target object for: <natural language object description>
- Execute robot subtask: <natural language manipulation instruction using a resolved object name>
- [Final Answer]: <success/failure summary>

For lift_pot-like goals, first find the target object for "pot". After the
Executor resolves the true object name, ask the Executor to lift that object
with both arms. Do not finish merely because env_task_success is true; finish
only when the original user goal is complete.

Output format:
[Thought]: ...
[Action]: ...
or
[Final Answer]: ...
"""

EXECUTOR_SYSTEM = """You are the Code-as-Policy Task Executor Agent.
You convert one high-level planner action into executable Python code.

Available Functions (Primary):
- get_reference_names() -> list[str]
- resolve_reference(query, refs) -> str
- check_object_sanity(object_name) -> dict
- dual_arm_grasp(object_name) -> dict (Composite: moves to handles, closes, and verifies micro-lift)
- dual_arm_lift(object_name, height) -> dict (Only call AFTER dual_arm_grasp succeeds. Only moves UP.)
- move_to_home_pos() -> dict
- get_feedback(result_dict) -> str (Use this to extract error messages)
- verify_task_success() -> dict

Rules:
1. Output ONLY Python code. No markdown fences.
2. YOU MUST ALWAYS ASSIGN BOTH `result` (bool) AND `feedback` (str) at the end of your code.
3. Use clean Code-as-Policy structures. Example for lifting:
    r = check_object_sanity("obj")
    if r["status"] == "SUCCESS":
        g = dual_arm_grasp("obj")
        if g["status"] == "SUCCESS":
            l = dual_arm_lift("obj", 0.15)
            # handle l["status"]
4. If a skill succeeds, set `result = True` and a descriptive `feedback`.
5. If any skill fails (status != "SUCCESS"), set `result = False`, use `get_feedback(r)` for `feedback`, and call `move_to_home_pos()`.
6. For compute_dual_grasp (Advanced), NEVER index it as a list like r[0]. It returns a dict. Prefer dual_arm_grasp().
"""


# ── LLM Client ───────────────────────────────────────────────────────────

class QwenClient:
    def __init__(self, model: str = None, base_url: str = None,
                 temperature: float = 0.1, max_tokens: int = 1024):
        api_key = os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "QWEN_API_KEY not found. Set it in policy/Your_Policy/.env"
            )
        self.base_url = base_url or QWEN_BASE_URL
        self.model = model or QWEN_MODEL
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        print(f"[QwenClient] Model: {self.model} | Base URL: {self.base_url}")

    def call(self, system: str, user: str, temperature: float = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=self.default_max_tokens,
        )
        return response.choices[0].message.content.strip()


# ── Agent ─────────────────────────────────────────────────────────────────

class ALRMAgent:
    def __init__(self, llm_client: QwenClient, max_turns: int = 10,
                 executor_mode: str = "cap", schema_max_retries: int = 1,
                 hierarchical_executor_max_attempts: int = 10):
        self.llm = llm_client
        self.episode_idx = 0
        self.logger: EpisodeLogger = None
        self.run_summary: RunSummaryLogger = None
        self._history = []
        self._episode_done = False
        self._step_limit = max_turns
        self._turn = 0
        self._task_name = "unknown"
        self._config_name = "unknown"
        self._executor_mode = executor_mode
        self._schema_max_retries = schema_max_retries
        self._hierarchical_executor_max_attempts = int(hierarchical_executor_max_attempts)
        self.continue_after_env_success = executor_mode == "hierarchical_schema"

    def new_episode(self, task_name: str, config_name: str):
        self._task_name = task_name
        self._config_name = config_name
        self.logger = EpisodeLogger(task_name, self.episode_idx)
        if self.run_summary is None:
            self.run_summary = RunSummaryLogger(task_name, config_name, self.llm.model)
        self._history = []
        self._turn = 0
        self._episode_done = False

    def end_episode(self, success: bool, steps: int, failure_reason: str = None):
        if self.logger:
            self.logger.finalize(success, steps, failure_reason)
            if self.run_summary:
                self.run_summary.add_episode(
                    self.episode_idx, success, self.logger._log_path
                )
        self.episode_idx += 1

    def step(self, TASK_ENV, observation) -> bool:
        if self._episode_done:
            return True

        instruction = TASK_ENV.get_instruction()
        if self.logger.log["instruction"] is None:
            self.logger.log_instruction(instruction)

        if self._executor_mode == "hierarchical_schema":
            scene_desc = None
        else:
            for _ in range(10):
                scene_objects = get_scene_objects(TASK_ENV)
                if scene_objects:
                    break
                if hasattr(TASK_ENV, 'delay'):
                    TASK_ENV.delay(10)
            scene_desc = build_scene_description(TASK_ENV)
            self.logger.log_scene(get_scene_objects(TASK_ENV))

        if self._turn >= self._step_limit:
            self._episode_done = True
            return True

        if self._executor_mode == "hierarchical_schema":
            action = self._plan_hierarchical(instruction)
        else:
            action = self._plan(instruction, scene_desc)
        if action is None or action.lower().startswith("final answer") or "[final answer]" in action.lower():
            self._episode_done = True
            return True

        print(f"\n[Agent] Turn {self._turn + 1}/{self._step_limit}: {action}")
        success, feedback = self._execute(action, scene_desc, TASK_ENV)
        
        self._history.append({"action": action, "observation": feedback})
        self._turn += 1

        if not success:
            print(f"[Agent] Action failed. Feedback: {feedback}")
        else:
            print(f"[Agent] Action succeeded. Feedback: {feedback}")

        if '"reason": "expert_unreachable_seed"' in feedback:
            self._episode_done = True
        if self._executor_mode == "hierarchical_schema" and not success:
            self._episode_done = True

        env_success_done = TASK_ENV.eval_success and self._executor_mode != "hierarchical_schema"
        if env_success_done or self._turn >= self._step_limit or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            self._episode_done = True

        if self._episode_done:
            # Proactively finalize if not already done, to avoid missing the last call
            global _episode_finalized
            if not _episode_finalized:
                self.end_episode(
                    success=TASK_ENV.eval_success,
                    steps=TASK_ENV.take_action_cnt,
                    failure_reason="Timeout or step limit reached" if not TASK_ENV.eval_success else None
                )
                _episode_finalized = True

        return self._episode_done

    def _plan(self, instruction: str, scene_desc: str):
        history_str = "\n".join([f"Action: {h['action']}\nObservation: {h['observation']}" for h in self._history])
        if not history_str:
            history_str = "None"
        
        user_msg = (
            f"Task instruction: {instruction}\n\n"
            f"Scene:\n{scene_desc}\n\n"
            f"Execution History:\n{history_str}\n\n"
            f"What is your next action?"
        )
        start = time.time()
        try:
            planner_system = SCHEMA_PLANNER_SYSTEM if self._executor_mode == "schema" else PLANNER_SYSTEM
            response = self.llm.call(planner_system, user_msg)
            latency = time.time() - start
            
            action = None
            for line in response.splitlines():
                if line.startswith("[Action]:"):
                    action = line.replace("[Action]:", "").strip()
                elif line.startswith("[Final Answer]:"):
                    action = line.strip()
            
            if not action:
                action = response.strip()

            self.logger.log_planner_call(
                prompt=user_msg, response=response, subtasks=[action],
                latency_s=latency, model=self.llm.model
            )
            return action

        except Exception as e:
            print(f"[Planner] ERROR: {e}")
            return None

    def _plan_hierarchical(self, instruction: str):
        history_str = "\n".join([
            f"Planner action: {h['action']}\nExecutor observation: {h['observation']}"
            for h in self._history
        ])
        if not history_str:
            history_str = "None"

        user_msg = (
            f"Original user goal:\n{instruction}\n\n"
            f"Subtask history:\n{history_str}\n\n"
            f"Decide the next high-level subtask or final answer."
        )
        start = time.time()
        try:
            response = self.llm.call(HIERARCHICAL_PLANNER_SYSTEM, user_msg)
            latency = time.time() - start

            action = None
            for line in response.splitlines():
                if line.startswith("[Action]:"):
                    action = line.replace("[Action]:", "").strip()
                elif line.startswith("[Final Answer]:"):
                    action = line.strip()

            if not action:
                action = response.strip()

            self.logger.log_planner_call(
                prompt=user_msg, response=response, subtasks=[action],
                latency_s=latency, model=self.llm.model
            )
            return action
        except Exception as e:
            print(f"[Planner] ERROR: {e}")
            return None

    def _execute(self, action: str, scene_desc: str, TASK_ENV) -> tuple:
        self.logger.start_step(action)

        if self._executor_mode == "hierarchical_schema":
            executor = HierarchicalSchemaExecutor(
                self.llm,
                logger=self.logger,
                max_attempts=self._hierarchical_executor_max_attempts,
            )
            success, feedback = executor.execute(
                action=action,
                TASK_ENV=TASK_ENV,
                planner_history=self._history,
            )
            self.logger.log_step_result(success, feedback)
            return success, feedback

        if self._executor_mode == "schema":
            executor = SchemaExecutor(
                self.llm,
                logger=self.logger,
                max_retries=self._schema_max_retries,
            )
            success, feedback = executor.execute(
                action=action,
                scene_desc=scene_desc,
                TASK_ENV=TASK_ENV,
                history=self._history,
            )
            self.logger.log_step_result(success, feedback)
            return success, feedback
        
        user_msg = (
            f"Current scene:\n{scene_desc}\n\n"
            f"Action to execute: {action}\n\n"
            f"Write Python code to complete this action using the available skills:"
        )

        start = time.time()
        try:
            code = self.llm.call(EXECUTOR_SYSTEM, user_msg)
            latency = time.time() - start
            
            code = code.replace("```python", "").replace("```", "").strip()
            
            self.logger.log_executor_call(
                prompt=user_msg, code=code,
                latency_s=latency, model=self.llm.model
            )

        except Exception as e:
            latency = time.time() - start
            error_msg = f"LLM call failed: {e}"
            self.logger.log_step_result(False, error_msg, error=error_msg)
            return False, error_msg

        skill_ns = build_skill_namespace(TASK_ENV, logger=self.logger)
        exec_locals = {"result": False, "feedback": "Code executed but result/feedback not set.", **skill_ns}

        try:
            exec(code, {}, exec_locals)
            success = bool(exec_locals.get("result", False))
            feedback = str(exec_locals.get("feedback", ""))
            
            # Fallback for missing feedback on success
            if success and feedback == "Code executed but result/feedback not set.":
                feedback = "Success"
                
            self.logger.log_step_result(success, feedback)
            return success, feedback

        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Code exec error: {e}"
            self.logger.log_step_result(False, error_msg, error=tb)
            return False, error_msg


# ── RoboTwin Policy Interface ─────────────────────────────────────────────
#
# Required: get_model, eval, reset_model  (called by script/eval_policy.py)
# Optional: encode_obs                    (internal convention, not called by harness)
#
# get_action() is NOT implemented. The ALRM agent's "action" is the full
# ReAct loop (plan -> generate code -> exec skills -> observe) spanning
# multiple env steps. There is no single action tensor to return.
#
# update_obs() is NOT implemented. The agent reads fresh scene state from
# TASK_ENV via privileged perception on every turn; there is no
# observation history buffer to maintain.

_agent: ALRMAgent = None
_episode_started = False
_episode_finalized = False

def get_model(usr_args: dict) -> ALRMAgent:
    """Initialize the ALRM agent from YAML config (deploy_policy.yml).

    Config keys read from usr_args (with env-var / hardcoded fallbacks):
      llm_model, llm_base_url, temperature, max_tokens, max_turns, verbose
    The API key is always read from the QWEN_API_KEY environment variable.
    """
    global _agent
    model_name = usr_args.get("llm_model") or QWEN_MODEL
    base_url = usr_args.get("llm_base_url") or QWEN_BASE_URL
    temperature = float(usr_args.get("temperature", 0.1))
    max_tokens = int(usr_args.get("max_tokens", 1024))
    max_turns = int(usr_args.get("max_turns", 10))
    executor_mode = str(usr_args.get("executor_mode", "cap")).lower()
    schema_max_retries = int(usr_args.get("schema_max_retries", 1))
    hierarchical_executor_max_attempts = int(
        usr_args.get("hierarchical_executor_max_attempts", 10)
    )
    verbose = usr_args.get("verbose", True)
    if executor_mode not in ("cap", "schema", "hierarchical_schema"):
        raise ValueError(
            "executor_mode must be 'cap', 'schema', or "
            f"'hierarchical_schema', got {executor_mode!r}"
        )

    llm = QwenClient(model=model_name, base_url=base_url,
                     temperature=temperature, max_tokens=max_tokens)
    _agent = ALRMAgent(
        llm,
        max_turns=max_turns,
        executor_mode=executor_mode,
        schema_max_retries=schema_max_retries,
        hierarchical_executor_max_attempts=hierarchical_executor_max_attempts,
    )
    if verbose:
        print(f"[LLMAgent] Initialized. Model={model_name} | MaxTurns={max_turns}")
        print("[ALRMAgent config]", {
            "llm_model": usr_args.get("llm_model"),
            "max_turns": usr_args.get("max_turns"),
            "executor_mode": executor_mode,
            "schema_max_retries": schema_max_retries,
            "hierarchical_executor_max_attempts": hierarchical_executor_max_attempts,
        })
    return _agent

def encode_obs(observation: dict) -> dict:
    """Observation preprocessing hook (pass-through for ALRM agent).

    Neural-net policies (ACT, DP, pi0) use this to normalize camera
    images and extract joint states. The ALRM agent reads scene state
    via privileged perception (build_scene_description) inside the
    ReAct loop, so raw observation preprocessing is unnecessary.
    """
    return observation

def eval(TASK_ENV, model: ALRMAgent, observation: dict):
    global _episode_started, _episode_finalized

    if not _episode_started:
        # Ensure environment is fully settled before capturing start state
        if hasattr(TASK_ENV, 'delay'):
            TASK_ENV.delay(10)
            
        task_name = getattr(TASK_ENV, 'task_name', 'unknown')
        config = getattr(TASK_ENV, 'task_config', 'unknown')
        
        # Proactively get fresh state for the logger
        model.new_episode(task_name, config)
        _episode_started = True
        _episode_finalized = False

    done = model.step(TASK_ENV, observation)

    if done and not _episode_finalized:
        model.end_episode(
            success=TASK_ENV.eval_success,
            steps=TASK_ENV.take_action_cnt,
            failure_reason="Task ended before success" if not TASK_ENV.eval_success else None
        )
        _episode_finalized = True

def reset_model(model: ALRMAgent):
    global _episode_started, _episode_finalized
    _episode_started = False
    _episode_finalized = False
