from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .models import ClinicalTrialAction, ClinicalTrialObservation

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
except Exception:  # pragma: no cover - compatibility fallback when openenv-core is unavailable
    TAction = TypeVar("TAction")
    TObs = TypeVar("TObs")
    TState = TypeVar("TState")

    @dataclass
    class State:
        episode_id: str
        step_count: int

    @dataclass
    class StepResult(Generic[TObs]):
        observation: TObs
        reward: float
        done: bool

    class EnvClient(Generic[TAction, TObs, TState]):
        def __init__(self, *args, **kwargs) -> None:
            self._args = args
            self._kwargs = kwargs


class ClinicalTrialEnv(EnvClient[ClinicalTrialAction, ClinicalTrialObservation, State]):
    def _step_payload(self, action: ClinicalTrialAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[ClinicalTrialObservation]:
        obs_data = payload.get("observation", {})
        obs = ClinicalTrialObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=str(payload.get("episode_id", "")),
            step_count=int(payload.get("step_count", 0)),
        )
