"""
Deterministic sales lead qualification simulator.

Uses ``lead_scoring.scoring.score_business`` for each lead's ``true_priority``,
then stores ``true_label`` via: score >= 20 → hot, >= 12 → warm, else cold.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, TypedDict

from lead_scoring.scoring import ScoreInput, score_business


class LeadSpec(TypedDict):
    id: str
    category: str
    has_website: bool
    has_contact_form: bool
    has_chat: bool
    rating: float


class ClassifyAction(TypedDict):
    type: Literal["classify"]
    value: Literal["hot", "warm", "cold"]


class PrioritizeAction(TypedDict):
    type: Literal["prioritize"]
    value: int


class SkipAction(TypedDict):
    type: Literal["skip"]


Action = ClassifyAction | PrioritizeAction | SkipAction


def _default_leads() -> list[LeadSpec]:
    """Fixed catalog (12 leads). No randomness."""
    return [
        {
            "id": "L01",
            "category": "plumbing",
            "has_website": True,
            "has_contact_form": False,
            "has_chat": False,
            "rating": 4.2,
        },
        {
            "id": "L02",
            "category": "hvac",
            "has_website": True,
            "has_contact_form": True,
            "has_chat": False,
            "rating": 3.8,
        },
        {
            "id": "L03",
            "category": "dental",
            "has_website": False,
            "has_contact_form": False,
            "has_chat": False,
            "rating": 4.5,
        },
        {
            "id": "L04",
            "category": "plumbing",
            "has_website": True,
            "has_contact_form": True,
            "has_chat": True,
            "rating": 4.9,
        },
        {
            "id": "L05",
            "category": "hvac",
            "has_website": True,
            "has_contact_form": False,
            "has_chat": True,
            "rating": 3.2,
        },
        {
            "id": "L06",
            "category": "dental",
            "has_website": True,
            "has_contact_form": False,
            "has_chat": False,
            "rating": 4.0,
        },
        {
            "id": "L07",
            "category": "plumbing",
            "has_website": True,
            "has_contact_form": True,
            "has_chat": False,
            "rating": 2.8,
        },
        {
            "id": "L08",
            "category": "hvac",
            "has_website": False,
            "has_contact_form": False,
            "has_chat": False,
            "rating": 3.5,
        },
        {
            "id": "L09",
            "category": "dental",
            "has_website": True,
            "has_contact_form": True,
            "has_chat": True,
            "rating": 4.6,
        },
        {
            "id": "L10",
            "category": "plumbing",
            "has_website": True,
            "has_contact_form": False,
            "has_chat": False,
            "rating": 3.0,
        },
        {
            "id": "L11",
            "category": "hvac",
            "has_website": True,
            "has_contact_form": True,
            "has_chat": False,
            "rating": 4.1,
        },
        {
            "id": "L12",
            "category": "dental",
            "has_website": True,
            "has_contact_form": False,
            "has_chat": True,
            "rating": 3.6,
        },
    ]


def _phone_for_lead_id(lead_id: str) -> str:
    s = sum((i + 1) * ord(c) for i, c in enumerate(lead_id))
    return f"330{s % 10_000_000:07d}"


def _review_count_from_rating(rating: float) -> int:
    return max(0, int(round(rating * 20.0)))


def _missing_features(lead: LeadSpec) -> list[str]:
    out: list[str] = []
    if not lead["has_contact_form"]:
        out.append("contact form")
    if not lead["has_chat"]:
        out.append("live chat")
    out.append("online booking")
    return out


def lead_to_score_input(lead: LeadSpec) -> ScoreInput:
    """Map public lead fields to the existing scorer (deterministic)."""
    cat = lead["category"]
    categories = [{"alias": cat, "title": cat}]
    if not lead["has_website"]:
        return {
            "categories": categories,
            "phone": _phone_for_lead_id(lead["id"]),
            "review_count": _review_count_from_rating(lead["rating"]),
            "hasForm": False,
            "hasChat": False,
            "hasBooking": False,
            "hasPhone": False,
            "missing_features": [],
            "scrapeError": "no_website",
        }

    return {
        "categories": categories,
        "phone": _phone_for_lead_id(lead["id"]),
        "review_count": _review_count_from_rating(lead["rating"]),
        "hasForm": lead["has_contact_form"],
        "hasChat": lead["has_chat"],
        "hasBooking": False,
        "hasPhone": lead["has_website"],
        "missing_features": _missing_features(lead),
        "scrapeError": None,
    }


def _label_from_score(priority: float) -> Literal["hot", "warm", "cold"]:
    """Derive qualification label from numeric priority (same thresholds as the JS pipeline)."""
    if priority >= 20.0:
        return "hot"
    if priority >= 12.0:
        return "warm"
    return "cold"


def _compute_ground_truth(
    leads: list[LeadSpec],
) -> tuple[dict[str, float], dict[str, int], dict[str, Literal["hot", "warm", "cold"]]]:
    """
    For each lead: run score_business, store true_priority and true_label.
    Also compute global rank (1 = highest priority). Ties: review_count, then id.
    """
    priorities: dict[str, float] = {}
    labels: dict[str, Literal["hot", "warm", "cold"]] = {}
    review_by_id: dict[str, int] = {}
    for lead in leads:
        scores = score_business(lead_to_score_input(lead))
        p = float(scores["priority"])
        lid = lead["id"]
        priorities[lid] = p
        labels[lid] = _label_from_score(p)
        review_by_id[lid] = _review_count_from_rating(lead["rating"])

    ordered = sorted(
        leads,
        key=lambda L: (
            -priorities[L["id"]],
            -review_by_id[L["id"]],
            L["id"],
        ),
    )
    ranks = {L["id"]: i + 1 for i, L in enumerate(ordered)}
    return priorities, ranks, labels


class LeadQualificationEnv:
    """
    Gym-style API: ``reset()``, ``step(action)``, ``state()``.

    One action is consumed per lead; the cursor then advances. ``done`` is True
    after the last lead has been processed.

    ``reset()``, ``step()`` (observation), and ``state()`` share the same nested
    structure: ``current_lead``, ``ground_truth``, ``remaining_leads``,
    ``history``, ``progress``, ``previous_reward``.
    """

    def __init__(self, leads: list[LeadSpec] | None = None) -> None:
        self._catalog = [copy.deepcopy(x) for x in (leads if leads is not None else _default_leads())]
        if not (10 <= len(self._catalog) <= 15):
            raise ValueError("Lead catalog must contain between 10 and 15 leads")
        self.leads = self._catalog
        self.current_index: int = 0
        self.actions_taken: list[Action] = []
        self.total_reward: float = 0.0
        self._previous_reward: float | None = None
        self._priorities, self._ranks, self._labels = _compute_ground_truth(self._catalog)

    def reset(self) -> dict[str, Any]:
        self.current_index = 0
        self.actions_taken = []
        self.total_reward = 0.0
        self._previous_reward = None
        self._priorities, self._ranks, self._labels = _compute_ground_truth(self._catalog)
        return self._observation()

    def state(self) -> dict[str, Any]:
        return self._rich_observation()

    def _lead_as_dict(self, lead: LeadSpec) -> dict[str, Any]:
        return {
            "id": lead["id"],
            "category": lead["category"],
            "has_website": lead["has_website"],
            "has_contact_form": lead["has_contact_form"],
            "has_chat": lead["has_chat"],
            "rating": lead["rating"],
        }

    def _classifications_from_actions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for step_i, act in enumerate(self.actions_taken):
            if act["type"] != "classify":
                continue
            # Lead this action applied to was at index step_i when taken
            if step_i >= len(self.leads):
                continue
            lid = self.leads[step_i]["id"]
            out.append(
                {
                    "step_index": step_i,
                    "lead_id": lid,
                    "label": act["value"],
                }
            )
        return out

    def _rich_observation(self) -> dict[str, Any]:
        n = len(self.leads)
        idx = self.current_index
        finished = idx >= n

        if finished:
            current_lead = None
            ground_truth = None
            upcoming: list[dict[str, Any]] = []
            remaining_count = 0
        else:
            L = self.leads[idx]
            current_lead = self._lead_as_dict(L)
            lid = L["id"]
            ground_truth = {
                "true_priority": self._priorities[lid],
                "true_label": self._labels[lid],
            }
            remaining_count = n - idx
            peek_end = min(n, idx + 1 + 3)
            upcoming = [self._lead_as_dict(self.leads[j]) for j in range(idx + 1, peek_end)]

        return {
            "current_lead": current_lead,
            "ground_truth": ground_truth,
            "remaining_leads": {
                "count": remaining_count,
                "upcoming": upcoming,
            },
            "history": {
                "actions": copy.deepcopy(self.actions_taken),
                "classifications": self._classifications_from_actions(),
            },
            "progress": {
                "current_index": idx,
                "total_leads": n,
                "finished": finished,
            },
            "previous_reward": self._previous_reward,
        }
    def step(self, action: Action) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.current_index >= len(self.leads):
            obs = self._observation()
            info = {"error": "episode_finished"}
            return obs, 0.0, True, info

        lead = self.leads[self.current_index]
        lid = lead["id"]
        true_priority = self._priorities[lid]
        true_label = self._labels[lid]
        true_rank = self._ranks[lid]

        reward = 0.0
        info: dict[str, Any] = {
            "lead_id": lid,
            "true_priority": true_priority,
            "true_label": true_label,
            "true_rank": true_rank,
        }

        kind = action["type"]
        if kind == "classify":
            predicted = action["value"]
            info["predicted_label"] = predicted
            correct = predicted == true_label
            info["correct"] = correct
            reward = 1.0 if correct else -1.0
        elif kind == "prioritize":
            predicted_rank = int(action["value"])
            info["predicted_rank"] = predicted_rank
            correct = predicted_rank == true_rank
            info["correct"] = correct
            reward = 0.5 if correct else -0.2
        elif kind == "skip":
            reward = -0.2
            info["skipped"] = True
        else:
            reward = -0.2
            info["error"] = "unknown_action_type"
            info["unnecessary"] = True

        self.actions_taken.append(copy.deepcopy(action))
        self.total_reward += reward
        self.current_index += 1
        self._previous_reward = reward

        done = self.current_index >= len(self.leads)
        info["total_reward"] = self.total_reward
        obs = self._observation()
        return obs, reward, done, info

    def _observation(self) -> dict[str, Any]:
        return self._rich_observation()
