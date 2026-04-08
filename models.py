from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import Action, Observation
except Exception:  # pragma: no cover - compatibility fallback when openenv-core is unavailable
    class Action(BaseModel):
        model_config = {"extra": "ignore"}

    class Observation(BaseModel):
        model_config = {"extra": "ignore"}


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ViolationFlag(BaseModel):
    model_config = ConfigDict(extra="ignore")

    section_id: str = Field(..., description="Which protocol section")
    rule_id: str = Field(..., description="Which regulatory rule e.g. RULE_001")
    severity: Severity = Field(..., description="Agent severity classification")
    explanation: str = Field(default="", description="Agent reasoning")
    suggested_correction: str = Field(default="", description="Agent fix")


class ClinicalTrialAction(Action):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
        validate_assignment=True,
    )

    action_type: str = Field(
        ...,
        description="One of: flag_violation | submit_report | accept_protocol | request_clarification",
    )
    violation_flags: list[ViolationFlag] = Field(
        default_factory=list,
        description="Violations being flagged in this action",
    )
    report_text: str = Field(
        default="",
        description="Full report text when action_type is submit_report",
    )
    explanation: str = Field(
        default="",
        description="Agent's overall explanation",
    )


class ClinicalTrialObservation(Observation):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
        validate_assignment=True,
    )

    task: str = Field(..., description="easy | medium | hard")
    step: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps for this task")
    protocol_id: str = Field(..., description="Unique protocol identifier")
    protocol_text: str = Field(..., description="Full protocol text visible to agent")
    available_actions: list[str] = Field(..., description="Valid action_types this step")
    reviewer_feedback: str = Field(default="", description="Feedback from regulatory reviewer (hard task only)")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated this episode")
    violations_found_so_far: int = Field(default=0, description="Correctly flagged violations so far")
    negotiation_round: int = Field(default=1, description="Round number for hard task")
    calibration_hint: str = Field(
        default="",
        description="Hint shown after step 2: 'Your severity classifications so far: X critical, Y major, Z minor'",
    )
    episode_done: bool = Field(default=False, description="Whether episode is complete")
    reward: float = Field(default=0.0, description="Most recent step reward")
