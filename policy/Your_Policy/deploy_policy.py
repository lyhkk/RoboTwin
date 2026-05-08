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
from skill_library import build_skill_namespace


# ── Config ────────────────────────────────────────────────────────────────
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
3. For dual-arm tasks (like lifting a pot), the sequence MUST be:
   a. dual_arm_grasp(object_name)
   b. dual_arm_lift(object_name, height)
4. If an observation reports failure (e.g., grasp failed), choose a recovery action (e.g., reset home, or try a different approach).
5. If check_object_sanity reports the object has fallen, use [Final Answer] to report failure.

Output format:
[Thought]: ...
[Action]: ...
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
    def __init__(self):
        api_key = os.environ.get("QWEN_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "QWEN_API_KEY not found. Set it in policy/Your_Policy/.env"
            )
        self.client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
        self.model = QWEN_MODEL
        print(f"[QwenClient] Model: {self.model} | Base URL: {QWEN_BASE_URL}")

    def call(self, system: str, user: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()


# ── Agent ─────────────────────────────────────────────────────────────────

class ALRMAgent:
    def __init__(self, llm_client: QwenClient):
        self.llm = llm_client
        self.episode_idx = 0
        self.logger: EpisodeLogger = None
        self.run_summary: RunSummaryLogger = None
        self._history = []
        self._episode_done = False
        self._step_limit = 10
        self._turn = 0
        self._task_name = "unknown"
        self._config_name = "unknown"

    def new_episode(self, task_name: str, config_name: str):
        self._task_name = task_name
        self._config_name = config_name
        self.logger = EpisodeLogger(task_name, self.episode_idx)
        if self.run_summary is None:
            self.run_summary = RunSummaryLogger(task_name, config_name, QWEN_MODEL)
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

        if TASK_ENV.eval_success or self._turn >= self._step_limit or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
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
            response = self.llm.call(PLANNER_SYSTEM, user_msg)
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
                latency_s=latency, model=QWEN_MODEL
            )
            return action

        except Exception as e:
            print(f"[Planner] ERROR: {e}")
            return None

    def _execute(self, action: str, scene_desc: str, TASK_ENV) -> tuple:
        self.logger.start_step(action)
        
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
                latency_s=latency, model=QWEN_MODEL
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

_agent: ALRMAgent = None
_episode_started = False
_episode_finalized = False

def get_model(usr_args: dict) -> ALRMAgent:
    global _agent
    llm = QwenClient()
    _agent = ALRMAgent(llm)
    print(f"[LLMAgent] Initialized. Model={QWEN_MODEL}")
    return _agent

def encode_obs(observation: dict) -> dict:
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
