from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from lead_scoring.models import LeadScoringAction, LeadScoringObservation


class LeadScoringEnv(EnvClient[LeadScoringAction, LeadScoringObservation, State]):
    """Typed WebSocket client for the lead scoring OpenEnv server."""

    def _step_payload(self, action: LeadScoringAction) -> Dict[str, Any]:
        return action.model_dump(by_alias=True, exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LeadScoringObservation]:
        obs_raw = payload.get("observation", {})
        obs = LeadScoringObservation.model_validate(obs_raw)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State.model_validate(payload)
