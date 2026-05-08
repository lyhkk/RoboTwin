"""
LLM Pipeline Logger
Logs all LLM API calls (prompt/response pairs), skill executions,
and episode feedback to timestamped JSON files.

Remote log dir: policy/Your_Policy/logs/
Sync to local: rsync -avP ubuntu:~/RoboTwin-release/policy/Your_Policy/logs/ ./policy/Your_Policy/logs/
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class EpisodeLogger:
    """
    One logger instance per eval episode.
    Records the full trace: instruction → LLM calls → skill executions → outcome.
    """

    def __init__(self, task_name: str, episode_idx: int):
        self.task_name = task_name
        self.episode_idx = episode_idx
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")

        self.log = {
            "run_id": self.run_id,
            "task_name": task_name,
            "episode_idx": episode_idx,
            "start_time": self.start_time.isoformat(),
            "instruction": None,
            "scene_info": None,
            "planner": {
                "prompt": None,
                "response": None,
                "subtasks": [],
                "latency_s": 0,
            },
            "steps": [], # Each step: {subtask, executor: {prompt, code, latency}, skills: [], result, feedback}
            "outcome": None,
            "failure_reason": None,
            "total_steps": 0,
            "duration_s": None,
        }

        self._log_path = LOG_DIR / f"{task_name}_ep{episode_idx:03d}_{self.run_id}.json"
        print(f"[Logger] Episode log → {self._log_path}")

    def log_instruction(self, instruction: str):
        self.log["instruction"] = instruction

    def log_scene(self, scene_info: dict):
        self.log["scene_info"] = scene_info

    def log_planner_call(self, prompt: str, response: str, subtasks: list, latency_s: float, model: str):
        self.log["planner"] = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "subtasks": subtasks,
            "latency_s": round(latency_s, 3),
        }
        self._flush()
        print(f"[LLM:Planner] {latency_s:.1f}s ✓ | Plan: {subtasks}")

    def start_step(self, subtask: str):
        """Initialize a new execution step."""
        step_entry = {
            "idx": len(self.log["steps"]),
            "subtask": subtask,
            "executor": None,
            "skills": [],
            "success": False,
            "feedback": None,
            "timestamp": datetime.now().isoformat(),
        }
        self.log["steps"].append(step_entry)
        self._flush()

    def log_executor_call(self, prompt: str, code: str, latency_s: float, model: str):
        if not self.log["steps"]: return
        step = self.log["steps"][-1]
        step["executor"] = {
            "model": model,
            "prompt": prompt,
            "code": code,
            "latency_s": round(latency_s, 3),
            "timestamp": datetime.now().isoformat(),
        }
        self._flush()
        print(f"[LLM:Executor] {latency_s:.1f}s ✓")

    def log_step_result(self, success: bool, feedback: str, error: str = None):
        if not self.log["steps"]: return
        step = self.log["steps"][-1]
        step["success"] = success
        step["feedback"] = feedback
        if error:
            step["error"] = error
        self._flush()
        icon = "✓" if success else "✗"
        print(f"[Agent] Step {step['idx']} {icon} | Feedback: {feedback}")

    def log_skill(self, skill_name: str, args: dict, result: str,
                  feedback: str, step_num: int, success: bool, data: dict = None):
        # Find current step and append skill
        if not self.log["steps"]: return
        step = self.log["steps"][-1]
        entry = {
            "skill_name": skill_name,
            "args": args,
            "result": result,
            "feedback": feedback,
            "success": success,
            "sim_step": step_num,
            "timestamp": datetime.now().isoformat(),
        }
        if data:
            entry["data"] = data
        step["skills"].append(entry)
        self._flush()
        icon = "✓" if success else "✗"
        print(f"  [Skill:{skill_name}] {icon} | {feedback}")

    def finalize(self, success: bool, total_steps: int, failure_reason: str = None):
        self.log["outcome"] = "success" if success else "failure"
        self.log["total_steps"] = total_steps
        self.log["failure_reason"] = failure_reason
        self.log["duration_s"] = round(
            (datetime.now() - self.start_time).total_seconds(), 2
        )
        self._flush()

        icon = "✅" if success else "❌"
        print(f"\n[Episode {self.episode_idx}] {icon} {self.log['outcome']} | "
              f"{total_steps} steps | {self.log['duration_s']}s")
        if failure_reason:
            print(f"  Failure reason: {failure_reason}")
        print(f"  Full log: {self._log_path}\n")

    def _flush(self):
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False, default=str)


class RunSummaryLogger:
    """
    Aggregates across all episodes in one eval run.
    Written to logs/summary_<run_id>.json
    """

    def __init__(self, task_name: str, config_name: str, model: str):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary = {
            "run_id": self.run_id,
            "task_name": task_name,
            "config_name": config_name,
            "model": model,
            "start_time": datetime.now().isoformat(),
            "episodes": [],
            "success_count": 0,
            "total_episodes": 0,
            "success_rate": 0.0,
        }
        self._path = LOG_DIR / f"summary_{task_name}_{self.run_id}.json"

    def add_episode(self, episode_idx: int, success: bool, log_path: str):
        self.summary["episodes"].append({
            "episode_idx": episode_idx,
            "success": success,
            "log_path": str(log_path),
        })
        self.summary["total_episodes"] += 1
        if success:
            self.summary["success_count"] += 1
        self.summary["success_rate"] = round(
            self.summary["success_count"] / self.summary["total_episodes"] * 100, 1
        )
        self._flush()
        print(f"[Summary] SR: {self.summary['success_count']}/{self.summary['total_episodes']} "
              f"= {self.summary['success_rate']}%")

    def _flush(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
