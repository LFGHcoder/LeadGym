from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field
from openenv.core.env_server.types import Action, Observation


class LeadScoringAction(Action):
    """Website signals + optional scrape error + LLM-style gap list (feeds score_business)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    has_form: bool = Field(alias="hasForm")
    has_chat: bool = Field(alias="hasChat")
    has_booking: bool = Field(alias="hasBooking")
    has_phone: bool = Field(alias="hasPhone")
    missing_features: list[str] = Field(default_factory=list)
    scrape_error: str | None = Field(default=None, alias="scrapeError")


class LeadScoringObservation(Observation):
    """After reset: business only. After step: includes scores and terminal reward."""

    business: dict[str, Any] = Field(description="Current lead (BusinessLead-shaped dict)")
    scores: dict[str, float] | None = Field(
        default=None, description="score_business output once an action is applied"
    )
