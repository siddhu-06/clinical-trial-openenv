from __future__ import annotations

from typing import Any

try:
    from openenv.core.env_server import create_app
except Exception:  # pragma: no cover
    from fastapi import FastAPI, HTTPException

    def create_app(
        env_cls: type,
        action_model: type,
        observation_model: type,
        env_name: str,
        gradio_builder=None,
    ) -> FastAPI:
        app_instance = FastAPI(title=env_name, version="1.0.0")
        env = env_cls()

        class _WebManager:
            def __init__(self, env_obj: Any, action_cls: type):
                self._env = env_obj
                self._action_cls = action_cls

            def reset_environment(self, payload: dict[str, object]) -> dict[str, object]:
                task = str(payload.get("task_id") or payload.get("task") or "easy")
                seed_raw = payload.get("seed", 42)
                seed = int(seed_raw) if seed_raw is not None else 42
                obs = self._env.reset(seed=seed, task_id=task)
                return obs.model_dump()

            def step_environment(self, payload: dict[str, object]) -> dict[str, object]:
                action = self._action_cls.model_validate(payload)
                obs = self._env.step(action)
                return {
                    "observation": obs.model_dump(),
                    "reward": obs.reward,
                    "done": obs.episode_done,
                    "info": getattr(self._env, "last_info", {}),
                }

        web_manager = _WebManager(env, action_model)

        @app_instance.post("/reset")
        async def reset(payload: dict[str, object] | None = None):
            payload = payload or {}
            return web_manager.reset_environment(payload)

        @app_instance.post("/step")
        async def step(payload: dict[str, object]):
            try:
                return web_manager.step_environment(payload)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @app_instance.get("/state")
        async def state():
            state_obj = env.state
            if hasattr(state_obj, "model_dump"):
                return state_obj.model_dump()
            return {
                "episode_id": getattr(state_obj, "episode_id", ""),
                "step_count": getattr(state_obj, "step_count", 0),
            }

        @app_instance.get("/health")
        async def health():
            return {"status": "ok", "version": "1.0.0"}

        return app_instance

from clinical_trial_env.models import ClinicalTrialAction, ClinicalTrialObservation

from .environment import ClinicalTrialEnvironment


def _gradio_builder(*args, **kwargs):
    from .gradio_ui import build_clinical_trial_ui

    return build_clinical_trial_ui(*args, **kwargs)


_env_instance = ClinicalTrialEnvironment()


def _env_factory() -> ClinicalTrialEnvironment:
    return _env_instance


app = create_app(
    _env_factory,
    ClinicalTrialAction,
    ClinicalTrialObservation,
    env_name="clinical_trial_env",
    gradio_builder=_gradio_builder,
)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "clinical_trial_env",
        "message": "Clinical Trial Protocol Review Environment",
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "clinical_trial_env.server.app:app",
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
