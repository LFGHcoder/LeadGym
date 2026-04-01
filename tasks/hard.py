"""
Hard task: shuffled, similar-priority leads; multi-step qualify workflow; step budget.

- ``max_steps = n + 5`` (physical environment steps).
- Agent should **classify** and **prioritize** every lead (single ``qualify`` action
  does both; or two separate actions across steps).
- Unnecessary actions (skip, invalid id, redundant work) are penalized via ``tasks.grader``.
- Final score = full ``grade_episode`` (0.5·classification + 0.3·ranking + 0.2·efficiency).
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from environment import LeadSpec, _compute_ground_truth  # noqa: E402

_SHUFFLE_SEED = 137


def _load_grader():
    path = Path(__file__).resolve().parent / "grader.py"
    spec = importlib.util.spec_from_file_location("tasks_grader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grade_episode


grade_episode = _load_grader()


class QualifyAction(TypedDict):
    type: Literal["qualify"]
    lead_id: str
    label: Literal["hot", "warm", "cold"]
    rank: int


class TargetClassifyAction(TypedDict):
    type: Literal["classify"]
    lead_id: str
    value: Literal["hot", "warm", "cold"]


class TargetPrioritizeAction(TypedDict):
    type: Literal["prioritize"]
    lead_id: str
    value: int


class SkipActionHard(TypedDict):
    type: Literal["skip"]


HardAction = QualifyAction | TargetClassifyAction | TargetPrioritizeAction | SkipActionHard


def hard_lead_catalog() -> list[LeadSpec]:
    """
    Twelve leads with **tight rating clusters** so many priorities sit in a narrow band
    (harder global ranking); categories repeat so the agent cannot rely on coarse cues alone.
    """
    rows: list[tuple[str, str, bool, bool, bool, float]] = [
        ("H01", "plumbing", True, False, False, 4.06),
        ("H02", "plumbing", True, False, False, 4.08),
        ("H03", "plumbing", True, True, False, 4.07),
        ("H04", "plumbing", True, False, True, 4.09),
        ("H05", "hvac", True, False, False, 4.05),
        ("H06", "hvac", True, True, False, 4.10),
        ("H07", "hvac", True, False, True, 4.04),
        ("H08", "dental", True, False, False, 4.11),
        ("H09", "dental", True, True, True, 4.12),
        ("H10", "plumbing", True, False, False, 4.05),
        ("H11", "dental", False, False, False, 4.13),
        ("H12", "hvac", True, True, True, 4.03),
    ]
    out: list[LeadSpec] = []
    for lid, cat, web, form, chat, rating in rows:
        out.append(
            {
                "id": lid,
                "category": cat,
                "has_website": web,
                "has_contact_form": form,
                "has_chat": chat,
                "rating": rating,
            }
        )
    return out


def _shuffle_leads(leads: list[LeadSpec], seed: int) -> list[LeadSpec]:
    idx = list(range(len(leads)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    return [leads[i] for i in idx]


def _ground_truth_map(leads: list[LeadSpec]) -> dict[str, dict[str, Any]]:
    priorities, ranks, labels = _compute_ground_truth(leads)
    return {
        lid: {
            "label": labels[lid],
            "priority": priorities[lid],
            "rank": ranks[lid],
        }
        for lid in priorities
    }


def _lead_public(L: LeadSpec) -> dict[str, Any]:
    return {
        "id": L["id"],
        "category": L["category"],
        "has_website": L["has_website"],
        "has_contact_form": L["has_contact_form"],
        "has_chat": L["has_chat"],
        "rating": L["rating"],
    }


class HardQualificationRunner:
    """
    Physical step budget ``n + 5``. Agent targets leads by id (shuffled presentation order).

    Observations expose **all** leads and per-lead completion state so policies must
    track progress over multiple steps.
    """

    def __init__(self, leads: list[LeadSpec], *, shuffle_seed: int = _SHUFFLE_SEED) -> None:
        base = [dict(x) for x in leads]  # type: ignore[arg-type]
        self.leads: list[LeadSpec] = [x for x in base]  # type: ignore[assignment]
        self.n = len(self.leads)
        if self.n < 10:
            raise ValueError("hard task expects at least 10 leads")
        self.lead_ids_shuffled: list[str] = [L["id"] for L in _shuffle_leads(self.leads, shuffle_seed)]
        self.gt = _ground_truth_map(self.leads)
        self.max_steps = self.n + 5
        self.physical_steps = 0
        self.classified: set[str] = set()
        self.prioritized: set[str] = set()
        self.grader_decisions: list[dict[str, Any]] = []
        self.physical_trace: list[dict[str, Any]] = []
        self.done_early = False

    def _all_required_done(self) -> bool:
        ids = {L["id"] for L in self.leads}
        return ids <= self.classified and ids <= self.prioritized

    def reset(self) -> dict[str, Any]:
        self.physical_steps = 0
        self.classified.clear()
        self.prioritized.clear()
        self.grader_decisions.clear()
        self.physical_trace.clear()
        self.done_early = False
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        by_id = {L["id"]: L for L in self.leads}
        per_lead = {}
        for lid in self.lead_ids_shuffled:
            L = by_id[lid]
            per_lead[lid] = {
                **_lead_public(L),
                "needs_classify": lid not in self.classified,
                "needs_prioritize": lid not in self.prioritized,
            }
        incomplete_cls = [lid for lid in self.lead_ids_shuffled if lid not in self.classified]
        incomplete_pri = [lid for lid in self.lead_ids_shuffled if lid not in self.prioritized]
        return {
            "task": "hard",
            "presentation_order": list(self.lead_ids_shuffled),
            "per_lead": per_lead,
            "incomplete_classify": incomplete_cls,
            "incomplete_prioritize": incomplete_pri,
            "physical_step": self.physical_steps,
            "max_steps": self.max_steps,
            "num_leads": self.n,
            "all_complete": self._all_required_done(),
        }

    def _append_grader_classify(self, lid: str, pred_label: str) -> None:
        t = self.gt[lid]
        correct = pred_label == t["label"]
        self.grader_decisions.append(
            {
                "action": {"type": "classify", "value": pred_label},
                "info": {
                    "lead_id": lid,
                    "true_label": t["label"],
                    "true_priority": t["priority"],
                    "true_rank": t["rank"],
                    "correct": correct,
                },
            }
        )

    def _append_grader_prioritize(self, lid: str, pred_rank: int) -> None:
        t = self.gt[lid]
        self.grader_decisions.append(
            {
                "action": {"type": "prioritize", "value": int(pred_rank)},
                "info": {
                    "lead_id": lid,
                    "true_label": t["label"],
                    "true_priority": t["priority"],
                    "true_rank": t["rank"],
                },
            }
        )

    def _append_grader_skip(self) -> None:
        self.grader_decisions.append({"action": {"type": "skip"}, "info": {}})

    def step(self, action: HardAction) -> tuple[dict[str, Any], bool]:
        """
        One physical step. May append one or two grader rows (qualify → classify + prioritize).
        Returns (observation, episode_over).
        """
        if self.physical_steps >= self.max_steps or self.done_early:
            obs = self._observation()
            return obs, True

        self.physical_steps += 1
        episode_over = self.physical_steps >= self.max_steps
        kind = action["type"]

        if kind == "skip":
            self._append_grader_skip()
        elif kind == "qualify":
            lid = action["lead_id"]
            if lid not in self.gt:
                self._append_grader_skip()
            else:
                progressed = False
                if lid not in self.classified:
                    self._append_grader_classify(lid, action["label"])
                    self.classified.add(lid)
                    progressed = True
                if lid not in self.prioritized:
                    self._append_grader_prioritize(lid, action["rank"])
                    self.prioritized.add(lid)
                    progressed = True
                if not progressed:
                    self._append_grader_skip()
        elif kind == "classify":
            lid = action["lead_id"]
            if lid not in self.gt or lid in self.classified:
                self._append_grader_skip()
            else:
                self._append_grader_classify(lid, action["value"])
                self.classified.add(lid)
        elif kind == "prioritize":
            lid = action["lead_id"]
            if lid not in self.gt or lid in self.prioritized:
                self._append_grader_skip()
            else:
                self._append_grader_prioritize(lid, action["value"])
                self.prioritized.add(lid)
        else:
            self._append_grader_skip()

        self.physical_trace.append({"physical_step": self.physical_steps, "action": dict(action)})

        if self._all_required_done():
            self.done_early = True
            episode_over = True

        obs = self._observation()
        return obs, episode_over


def oracle_hard_policy(obs: dict[str, Any], runner: HardQualificationRunner) -> HardAction:
    """One ``qualify`` per lead in presentation order (optimal under step budget)."""
    for lid in runner.lead_ids_shuffled:
        if lid not in runner.classified or lid not in runner.prioritized:
            t = runner.gt[lid]
            return {
                "type": "qualify",
                "lead_id": lid,
                "label": t["label"],  # type: ignore[arg-type]
                "rank": int(t["rank"]),
            }
    return {"type": "skip"}


def naive_hard_policy(obs: dict[str, Any], runner: HardQualificationRunner) -> HardAction:
    """Heuristic qualify: rough label from rating; rank from proxy (often wrong on tight cluster)."""
    for lid in runner.lead_ids_shuffled:
        if lid not in runner.classified or lid not in runner.prioritized:
            L = next(x for x in runner.leads if x["id"] == lid)
            if not L["has_website"]:
                label: Literal["hot", "warm", "cold"] = "warm"
            elif L["rating"] >= 4.11:
                label = "hot"
            elif L["rating"] >= 4.07:
                label = "warm"
            else:
                label = "cold"
            proxy = int(L["rating"] * 100) % runner.n + 1
            return {"type": "qualify", "lead_id": lid, "label": label, "rank": proxy}
    return {"type": "skip"}


def run_task(
    policy: Callable[[dict[str, Any], HardQualificationRunner], HardAction] | None = None,
    *,
    shuffle_seed: int = _SHUFFLE_SEED,
) -> dict[str, Any]:
    """
    Run until **physical** step budget is exhausted or every lead is classified **and**
    prioritized.

    Grader input is built from expanded rows (each ``qualify`` → classify + prioritize).
    """
    policy = policy or oracle_hard_policy
    leads = hard_lead_catalog()
    runner = HardQualificationRunner(leads, shuffle_seed=shuffle_seed)
    obs = runner.reset()
    done = False

    while not done:
        act = policy(obs, runner)
        obs, reward, done, info = runner.step(act)

    gt = {lid: {"label": v["label"], "priority": v["priority"], "rank": v["rank"]} for lid, v in runner.gt.items()}
    graded = grade_episode(runner.grader_decisions, gt)

    req_ok = runner._all_required_done()
    if not req_ok:
        graded = {
            "score": 0.0,
            "metrics": {
                "classification_accuracy": 0.0,
                "ranking_quality": 0.0,
                "efficiency": 0.0,
            },
        }

    return {
        "task": "hard",
        "score": graded["score"],
        "final_score": graded["score"],
        "metrics": graded["metrics"],
        "requirement_all_classified_and_prioritized": req_ok,
        "num_leads": runner.n,
        "max_steps": runner.max_steps,
        "physical_steps_used": runner.physical_steps,
        "shuffle_seed": shuffle_seed,
        "presentation_order": list(runner.lead_ids_shuffled),
        "grader_decisions": runner.grader_decisions,
        "physical_trace": runner.physical_trace,
    }


if __name__ == "__main__":
    r = run_task()
    print(r["task"], "score=", r["score"], "metrics=", r["metrics"])
    r2 = run_task(naive_hard_policy)
    print("naive", "score=", r2["score"], "req_ok=", r2["requirement_all_classified_and_prioritized"])
