from __future__ import annotations

from collections import Counter
from uuid import uuid4

from clinical_trial_env.counterpart_reviewer import generate_reviewer_feedback
from clinical_trial_env.graders import (
    compute_calibration_score,
    compute_f1,
    grade_easy,
    grade_hard,
    grade_medium,
    keyword_overlap,
    match_flags_to_ground_truth,
)
from clinical_trial_env.models import ClinicalTrialAction, ClinicalTrialObservation, ViolationFlag
from clinical_trial_env.protocol_generator import ClinicalProtocol, generate_protocol
from clinical_trial_env.regulatory_rules import RULES

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except Exception:  # pragma: no cover - compatibility fallback when openenv-core is unavailable
    from dataclasses import dataclass

    class Environment:
        def __init__(self, *args, **kwargs) -> None:
            self._compat = True

    @dataclass
    class State:
        episode_id: str
        step_count: int


class ClinicalTrialEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASK_MAX_STEPS = {"easy": 4, "medium": 8, "hard": 14}
    TASK_ACTIONS = {
        "easy": ["flag_violation", "accept_protocol"],
        "medium": ["flag_violation", "submit_report", "accept_protocol"],
        "hard": ["flag_violation", "submit_report", "request_clarification", "accept_protocol"],
    }

    def __init__(self) -> None:
        try:
            super().__init__(rubric=None)
        except TypeError:  # pragma: no cover - compatibility fallback
            super().__init__()

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._protocol: ClinicalProtocol | None = None
        self._task: str = "easy"
        self._seed: int = 42
        self._current_step: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._all_flags: list[ViolationFlag] = []
        self._round1_flags: list[ViolationFlag] = []
        self._final_flags: list[ViolationFlag] = []
        self._round: int = 1
        self._reviewer_feedback: str = ""
        self._report_text: str = ""
        self._adapted_to_feedback: bool = False
        self._flags_before_feedback: set[tuple[str, str]] = set()
        self._accepted_with_criticals: bool = False
        self._steps_used: int = 0
        self._action_history: list[str] = []
        self._difficulty_modifier: float = 1.0
        self._last_info: dict[str, object] = {}
        self._last_reward_components: dict[str, float] = {
            "violation_f1": 0.0,
            "severity_score": 0.0,
            "correction_score": 0.0,
            "calibration_score": 0.0,
            "cascade_detected": 0.0,
            "efficiency": 0.0,
        }

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> ClinicalTrialObservation:
        task = kwargs.get("task", "easy")
        actual_seed = seed if seed is not None else 42
        task_value = task.lower()
        if task_value not in self.TASK_MAX_STEPS:
            raise ValueError(f"Unsupported task: {task}")

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._task = task_value
        self._seed = actual_seed
        self._protocol = generate_protocol(task_value, actual_seed)
        # Apply difficulty modifier based on seed parity for natural variance
        # Even seeds = slightly clearer violation language (easier to detect)
        # Odd seeds = violation triggers are more deeply buried in text
        # This creates smooth difficulty curves for RL training
        self._difficulty_modifier = 0.85 if (actual_seed % 2 == 1) else 1.0
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._all_flags = []
        self._round1_flags = []
        self._final_flags = []
        self._round = 1
        self._reviewer_feedback = ""
        self._report_text = ""
        self._adapted_to_feedback = False
        self._flags_before_feedback = set()
        self._accepted_with_criticals = False
        self._steps_used = 0
        self._action_history = []
        self._last_info = {}
        self._last_reward_components = {
            "violation_f1": 0.0,
            "severity_score": 0.0,
            "correction_score": 0.0,
            "calibration_score": 0.0,
            "cascade_detected": 0.0,
            "efficiency": 0.0,
        }
        return self._make_observation(step_reward=0.0)

    def step(
        self,
        action: ClinicalTrialAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> ClinicalTrialObservation:
        _ = timeout_s
        _ = kwargs
        if self._protocol is None:
            raise RuntimeError("Environment is not initialized. Call reset first.")
        if self._done:
            return self._make_observation(step_reward=0.0)

        self._current_step += 1
        self._steps_used += 1
        self._state.step_count += 1

        step_reward = 0.0
        action_key = f"{action.action_type}:{[f.section_id + f.rule_id for f in action.violation_flags]}"
        is_repeated = action_key in self._action_history
        if is_repeated:
            step_reward -= 0.05
        self._action_history.append(action_key)

        if action.action_type == "flag_violation":
            step_reward += self._process_flags(action.violation_flags)

        elif action.action_type == "submit_report":
            self._report_text = action.report_text
            if self._task == "hard" and self._round == 1:
                self._round1_flags = list(self._all_flags)
                self._flags_before_feedback = {(f.section_id, f.rule_id) for f in self._all_flags}
                self._reviewer_feedback = generate_reviewer_feedback(
                    self._protocol, self._all_flags, self._seed
                )
                self._round = 2
                step_reward += 0.05
            elif self._task == "hard" and self._round == 2:
                new_pairs = {(f.section_id, f.rule_id) for f in self._all_flags}
                if new_pairs != self._flags_before_feedback:
                    self._adapted_to_feedback = True
                    step_reward += 0.08
                self._round = 3
                self._final_flags = list(self._all_flags)
                self._done = True
                step_reward += self._compute_final_reward()
            else:
                self._final_flags = list(self._all_flags)
                self._done = True
                step_reward += self._compute_final_reward()

        elif action.action_type == "accept_protocol":
            critical_sections = [
                s for s in self._protocol.sections if s.has_violation and s.severity == "critical"
            ]
            flagged_pairs = {(f.section_id, f.rule_id) for f in self._all_flags}
            unflagged_criticals = [
                s
                for s in critical_sections
                if not any((s.section_id, rid) in flagged_pairs for rid in s.violated_rule_ids)
            ]
            if unflagged_criticals:
                self._accepted_with_criticals = True
                step_reward -= 0.15
            self._final_flags = list(self._all_flags)
            self._done = True
            step_reward += self._compute_final_reward()

        elif action.action_type == "request_clarification":
            step_reward += 0.01

        max_steps = self.TASK_MAX_STEPS.get(self._task, 8)
        # Warning observation at penultimate step
        if self._current_step == max_steps - 1 and not self._done:
            self._reviewer_feedback = (
                self._reviewer_feedback
                + "\n\n⚠️ FINAL STEP WARNING: This is your last step. "
                "You must call submit_report or accept_protocol now."
            ).strip()

        # At max_steps: force submit with whatever flags exist
        if self._current_step >= max_steps and not self._done:
            self._done = True
            self._final_flags = list(self._all_flags)
            timeout_penalty = 0.15
            final_r = self._compute_final_reward()
            step_reward += max(0.0, final_r - timeout_penalty)

        if self._done:
            violated_sections = [s for s in self._protocol.sections if s.has_violation]
            total_violations = sum(len(s.violated_rule_ids) for s in violated_sections)
            flagged_pairs = {(f.section_id, f.rule_id) for f in self._all_flags}
            correct_flags = sum(
                1
                for s in violated_sections
                for rid in s.violated_rule_ids
                if (s.section_id, rid) in flagged_pairs
            )
            info = {
                "episode_summary": {
                    "task": self._task,
                    "seed": self._seed,
                    "total_steps": self._steps_used,
                    "total_violations_in_protocol": total_violations,
                    "correctly_flagged": correct_flags,
                    "false_positives": len(self._all_flags) - correct_flags,
                    "cascade_detected": any(
                        s.cascade_from is not None
                        and any((s.section_id, rid) in flagged_pairs for rid in s.violated_rule_ids)
                        for s in self._protocol.sections
                        if s.has_violation
                    ),
                    "accepted_with_criticals": self._accepted_with_criticals,
                    "final_grader_score": self._cumulative_reward,
                    "protocol_id": self._protocol.protocol_id,
                    "phase": self._protocol.phase,
                }
            }
        else:
            info = {
                "step": self._current_step,
                "task": self._task,
                "round": self._round,
            }

        self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward + step_reward))
        if self._done and "episode_summary" in info:
            info["episode_summary"]["final_grader_score"] = self._cumulative_reward
            info["reward_breakdown"] = self._last_reward_components
        self._last_info = info
        return self._make_observation(step_reward=step_reward)

    def _process_flags(self, new_flags: list[ViolationFlag]) -> float:
        if self._protocol is None:
            return 0.0
        reward = 0.0
        violated = {
            s.section_id: (set(s.violated_rule_ids), s.severity)
            for s in self._protocol.sections
            if s.has_violation
        }
        existing_pairs = {(f.section_id, f.rule_id) for f in self._all_flags}

        for flag in new_flags:
            pair = (flag.section_id, flag.rule_id)
            if pair in existing_pairs:
                reward -= 0.05
                continue

            if flag.section_id in violated and flag.rule_id in violated[flag.section_id][0]:
                reward += 0.12
                gt_severity = violated[flag.section_id][1]
                if gt_severity and flag.severity.value == gt_severity:
                    reward += 0.05
                rule = RULES.get(flag.rule_id, {})
                overlap = sum(
                    1
                    for kw in list(rule.get("fix_keywords", []))
                    if kw.lower() in flag.suggested_correction.lower()
                )
                if overlap > 0:
                    reward += 0.04
            else:
                reward -= 0.10

            existing_pairs.add(pair)
            self._all_flags.append(flag)
        return reward

    def _compute_final_reward(self) -> float:
        if not self._protocol:
            return 0.0
        headroom = max(0.0, 1.0 - self._cumulative_reward)
        if self._task == "easy":
            tp, fp, fn_count = match_flags_to_ground_truth(self._all_flags, self._protocol)
            violation_f1 = compute_f1(len(tp), len(fp), fn_count)
            severity_score = 0.0
            if tp:
                matched = 0
                for flag in tp:
                    section = next(s for s in self._protocol.sections if s.section_id == flag.section_id)
                    if section.severity and section.severity == flag.severity.value:
                        matched += 1
                severity_score = matched / len(tp)
            calibration_score = compute_calibration_score(self._all_flags, self._protocol)
            final = grade_easy(self._all_flags, self._protocol)
            self._last_reward_components = {
                "violation_f1": violation_f1,
                "severity_score": severity_score,
                "correction_score": 0.0,
                "calibration_score": calibration_score,
                "cascade_detected": 0.0,
                "efficiency": 1.0 if self._steps_used <= 4 else 0.0,
            }
            return final * headroom

        if self._task == "medium":
            tp, fp, fn_count = match_flags_to_ground_truth(self._all_flags, self._protocol)
            violation_f1 = compute_f1(len(tp), len(fp), fn_count)
            severity_score = 0.0
            correction_score = 0.0
            if tp:
                matched = 0
                overlaps: list[float] = []
                for flag in tp:
                    section = next(s for s in self._protocol.sections if s.section_id == flag.section_id)
                    if section.severity and section.severity == flag.severity.value:
                        matched += 1
                    overlaps.append(
                        keyword_overlap(
                            flag.suggested_correction,
                            list(RULES.get(flag.rule_id, {}).get("fix_keywords", [])),
                        )
                    )
                severity_score = matched / len(tp)
                correction_score = sum(overlaps) / len(overlaps) if overlaps else 0.0
            calibration_score = compute_calibration_score(self._all_flags, self._protocol)
            efficiency = 1.0 if self._steps_used <= 5 else 0.0
            final = grade_medium(self._all_flags, self._protocol, self._steps_used)
            self._last_reward_components = {
                "violation_f1": violation_f1,
                "severity_score": severity_score,
                "correction_score": correction_score,
                "calibration_score": calibration_score,
                "cascade_detected": 0.0,
                "efficiency": efficiency,
            }
            normalized_final = min(1.0, final / 0.91) if final > 0.0 else 0.0
            return normalized_final * headroom

        final = grade_hard(
            self._round1_flags,
            self._final_flags,
            self._report_text,
            self._protocol,
            self._adapted_to_feedback,
            self._accepted_with_criticals,
            self._steps_used,
        )
        tp_final, _, _ = match_flags_to_ground_truth(self._final_flags, self._protocol)
        correction_score = 0.0
        if tp_final:
            overlaps = [
                keyword_overlap(
                    f.suggested_correction,
                    list(RULES.get(f.rule_id, {}).get("fix_keywords", [])),
                )
                for f in tp_final
            ]
            correction_score = sum(overlaps) / len(overlaps)
        calibration_score = compute_calibration_score(self._final_flags, self._protocol)
        cascade_detected = 1.0 if any(
            s.cascade_from is not None
            and any(
                (f.section_id == s.section_id and f.rule_id in s.violated_rule_ids)
                for f in self._final_flags
            )
            for s in self._protocol.sections
            if s.has_violation
        ) else 0.0
        efficiency = 1.0 if self._steps_used <= 10 else max(0.0, 1.0 - (self._steps_used - 10) * 0.1)
        self._last_reward_components = {
            "violation_f1": 0.0,
            "severity_score": 0.0,
            "correction_score": correction_score,
            "calibration_score": calibration_score,
            "cascade_detected": cascade_detected,
            "efficiency": efficiency,
        }
        return final * headroom

    def _make_observation(self, step_reward: float) -> ClinicalTrialObservation:
        if self._protocol is None:
            raise RuntimeError("Protocol is not initialized.")
        protocol_text = self._protocol.full_text
        if self._difficulty_modifier < 1.0:
            for section in self._protocol.sections:
                if not section.has_violation:
                    continue
                for rule_id in section.violated_rule_ids:
                    trigger = str(RULES.get(rule_id, {}).get("violation_trigger", ""))
                    if trigger:
                        wrapped_trigger = (
                            "Context notes indicate routine compliance checks and standard site preparation. "
                            + trigger
                            + " Additional implementation notes indicate this language is embedded within broader operational text."
                        )
                        protocol_text = protocol_text.replace(trigger, wrapped_trigger)

        violated = {s.section_id: set(s.violated_rule_ids) for s in self._protocol.sections if s.has_violation}
        flagged_pairs = {(f.section_id, f.rule_id) for f in self._all_flags}
        violations_found = sum(
            1 for sid, rid in flagged_pairs if sid in violated and rid in violated[sid]
        )
        if self._current_step >= 2 and self._all_flags:
            sev_counts = Counter(f.severity.value for f in self._all_flags)
            hint = (
                "Your severity classifications so far: "
                f"{sev_counts.get('critical', 0)} critical, "
                f"{sev_counts.get('major', 0)} major, "
                f"{sev_counts.get('minor', 0)} minor"
            )
        else:
            hint = ""

        return ClinicalTrialObservation(
            task=self._task,
            step=self._current_step,
            max_steps=self.TASK_MAX_STEPS.get(self._task, 8),
            protocol_id=self._protocol.protocol_id,
            protocol_text=protocol_text,
            available_actions=self.TASK_ACTIONS.get(self._task, ["flag_violation"]),
            reviewer_feedback=self._reviewer_feedback,
            cumulative_reward=round(self._cumulative_reward, 4),
            violations_found_so_far=violations_found,
            negotiation_round=self._round,
            calibration_hint=hint,
            episode_done=self._done,
            reward=step_reward,
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def last_info(self) -> dict[str, object]:
        return self._last_info

    def get_metadata(self):
        try:
            from openenv.core.env_server.types import EnvironmentMetadata

            return EnvironmentMetadata(
                name="clinical-trial-env",
                version="1.0.0",
                description=(
                    "A clinical trial protocol review environment where AI agents "
                    "act as regulatory review officers. Agents identify violations "
                    "in synthetic protocols against ICH E6 GCP and FDA 21 CFR Part 50 "
                    "principles. Features cascade violations, calibrated severity grading, "
                    "multi-round reviewer feedback, and anti-exploit hardening."
                ),
                tasks=["easy", "medium", "hard"],
            )
        except Exception:
            return {
                "name": "clinical-trial-env",
                "version": "1.0.0",
                "description": (
                    "A clinical trial protocol review environment where AI agents act as "
                    "regulatory review officers with cascade violations and calibrated grading."
                ),
                "tasks": ["easy", "medium", "hard"],
            }

    def close(self) -> None:
        """Clean up episode state on environment teardown."""
        self._protocol = None
        self._all_flags = []
        self._done = True
