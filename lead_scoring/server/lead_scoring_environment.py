from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from lead_scoring.lead_data import BusinessLead, SAMPLE_LEADS
from lead_scoring.models import LeadScoringAction, LeadScoringObservation
from lead_scoring.scoring import score_business


def _reward_from_priority(priority: float) -> float:
    # Rough upper bound from pain<=3, expansion<=~8, fit<=3 -> ~37
    return max(0.0, min(1.0, priority / 40.0))


class LeadScoringEnvironment(Environment[LeadScoringAction, LeadScoringObservation, State]):
    """
    One-step episode: reset samples a lead; step applies signals + features and returns scores.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        leads: list[BusinessLead] | None = None,
        transform: Any = None,
        rubric: Any = None,
    ) -> None:
        super().__init__(transform=transform, rubric=rubric)
        self._catalog = list(leads) if leads is not None else list(SAMPLE_LEADS)
        if not self._catalog:
            raise ValueError("Lead catalog must be non-empty")
        self._rng = random.Random()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current: BusinessLead | None = None
        self._awaiting_action = False

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> LeadScoringObservation:
        if seed is not None:
            self._rng.seed(seed)
        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)
        idx = self._rng.randrange(len(self._catalog))
        self._current = self._catalog[idx]
        self._awaiting_action = True
        return self._apply_transform(
            LeadScoringObservation(
                business=dict(self._current),
                scores=None,
                done=False,
                reward=0.0,
            )
        )

    def step(
        self,
        action: LeadScoringAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> LeadScoringObservation:
        if not self._awaiting_action or self._current is None:
            return self._apply_transform(
                LeadScoringObservation(
                    business={},
                    scores=None,
                    done=True,
                    reward=0.0,
                    metadata={"error": "step_called_without_active_episode"},
                )
            )

        inp = {
            "categories": self._current["categories"],
            "phone": self._current["phone"],
            "review_count": self._current["review_count"],
            "hasForm": action.has_form,
            "hasChat": action.has_chat,
            "hasBooking": action.has_booking,
            "hasPhone": action.has_phone,
            "missing_features": list(action.missing_features),
            "scrapeError": action.scrape_error,
        }
        scores = score_business(inp)
        reward = _reward_from_priority(float(scores["priority"]))

        self._state = self._state.model_copy(update={"step_count": self._state.step_count + 1})
        self._awaiting_action = False

        return self._apply_transform(
            LeadScoringObservation(
                business=dict(self._current),
                scores=dict(scores),
                done=True,
                reward=reward,
            )
        )

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="lead_scoring",
            description="Sample a business lead, submit website signals, receive priority scores.",
            version="0.1.0",
        )
