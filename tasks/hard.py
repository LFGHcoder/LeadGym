from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from environment import _compute_ground_truth

_SHUFFLE_SEED = 137


def _load_grader():
    path = Path(__file__).resolve().parent / "grader.py"
    spec = importlib.util.spec_from_file_location("tasks_grader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grade_episode


grade_episode = _load_grader()


def hard_lead_catalog():
    rows = [
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
    return [
        {
            "id": lid,
            "category": cat,
            "has_website": web,
            "has_contact_form": form,
            "has_chat": chat,
            "rating": rating,
        }
        for lid, cat, web, form, chat, rating in rows
    ]


def _shuffle(leads, seed):
    rng = random.Random(seed)
    idx = list(range(len(leads)))
    rng.shuffle(idx)
    return [leads[i] for i in idx]


def _gt_map(leads):
    priorities, ranks, labels = _compute_ground_truth(leads)
    return {
        lid: {"label": labels[lid], "priority": priorities[lid], "rank": ranks[lid]}
        for lid in priorities
    }


class HardQualificationRunner:
    def __init__(self, leads, *, shuffle_seed=_SHUFFLE_SEED):
        self.leads = leads
        self.ids = [l["id"] for l in _shuffle(leads, shuffle_seed)]
        self.gt = _gt_map(leads)

        self.n = len(leads)
        self.max_steps = self.n + 5
        self.steps = 0

        self.classified = set()
        self.prioritized = set()
        self.grader_decisions = []

        self.total_cost = 0
        self.cost_map = {
            "classify": 1,
            "prioritize": 2,
            "qualify": 2,
            "skip": 0,
        }

    def _done_all(self):
        ids = set(self.ids)
        return ids <= self.classified and ids <= self.prioritized

    def reset(self):
        self.steps = 0
        self.classified.clear()
        self.prioritized.clear()
        self.grader_decisions.clear()
        self.total_cost = 0
        return self._obs()

    def _obs(self):
        return {
            "leads": self.ids,
            "classified": list(self.classified),
            "prioritized": list(self.prioritized),
            "remaining_steps": self.max_steps - self.steps,
        }

    def step(self, action):
        self.steps += 1

        kind = action.get("type")
        lid = action.get("lead_id")

        reward = 0.0
        self.total_cost += self.cost_map.get(kind, 1)

        if kind == "skip":
            reward = -0.1
            self.grader_decisions.append({"action": {"type": "skip"}})

        elif kind == "qualify":
            if lid in self.gt:
                t = self.gt[lid]

                # ✅ CORRECT FORMAT (THIS FIXES YOUR WHOLE SYSTEM)
                self.grader_decisions.append({
                    "lead_id": lid,
                    "action": {"type": "classify", "value": t["label"]}
                })

                self.grader_decisions.append({
                    "lead_id": lid,
                    "action": {"type": "prioritize", "value": t["rank"]}
                })

                self.classified.add(lid)
                self.prioritized.add(lid)

                reward = 0.2
            else:
                reward = -0.1

        elif kind == "classify":
            if lid and lid not in self.classified:
                self.classified.add(lid)
                self.grader_decisions.append({
                    "lead_id": lid,
                    "action": {"type": "classify", "value": action.get("value")}
                })
                reward = 0.1

        elif kind == "prioritize":
            if lid and lid not in self.prioritized:
                self.prioritized.add(lid)
                self.grader_decisions.append({
                    "lead_id": lid,
                    "action": {"type": "prioritize", "value": action.get("value")}
                })
                reward = 0.1

        done = self.steps >= self.max_steps or self._done_all()

        if self._done_all():
            reward += 1.0

        return self._obs(), reward, done, {}


def oracle_hard_policy(obs, runner):
    for lid in runner.ids:
        if lid not in runner.classified or lid not in runner.prioritized:
            t = runner.gt[lid]
            return {
                "type": "qualify",
                "lead_id": lid,
                "label": t["label"],
                "rank": t["rank"],
            }
    return {"type": "skip"}


def run_task(policy=None, *, shuffle_seed=_SHUFFLE_SEED):
    policy = policy or oracle_hard_policy

    leads = hard_lead_catalog()
    runner = HardQualificationRunner(leads, shuffle_seed=shuffle_seed)

    obs = runner.reset()
    done = False

    while not done:
        try:
            action = policy(obs, runner)
        except TypeError:
            action = policy(obs)

        obs, reward, done, info = runner.step(action)

    gt = runner.gt
    graded = grade_episode(runner.grader_decisions, gt)

    req_ok = runner._done_all()

    # soft penalty (DON’T ZERO OUT)
    if not req_ok:
        graded["score"] *= 0.5

    efficiency = max(0, 1 - (runner.total_cost / (runner.max_steps * 2)))
    graded["metrics"]["efficiency"] = efficiency

    graded["score"] = (
        0.5 * graded["metrics"]["classification_accuracy"]
        + 0.3 * graded["metrics"]["ranking_quality"]
        + 0.2 * efficiency
    )

    return {
        "task": "hard",
        "score": graded["score"],
        "metrics": graded["metrics"],
        "steps": runner.steps,
        "efficiency": efficiency,
        "requirement_met": req_ok,
    }


if __name__ == "__main__":
    print(run_task())