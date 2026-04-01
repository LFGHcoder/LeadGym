"""
Medium task: run until done with prioritize-only actions; final score = ranking quality [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from environment import (  # noqa: E402
    Action,
    LeadQualificationEnv,
    LeadSpec,
    _compute_ground_truth,
)


def _true_ranks(leads: list[LeadSpec]) -> dict[str, int]:
    priorities, ranks, _ = _compute_ground_truth(leads)
    return ranks


def _normalized_rank_score(predicted: int, true_rank: int, n: int) -> float:
    """1.0 if exact; linear decay to 0.0 at maximal rank error (n - 1)."""
    if n <= 1:
        return 1.0 if predicted == true_rank else 0.0
    max_err = n - 1
    err = abs(int(predicted) - int(true_rank))
    return max(0.0, 1.0 - err / max_err)


def oracle_prioritize_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    """Deterministic perfect rank (matches env ground truth)."""
    lead = obs["current_lead"]
    assert lead is not None
    ranks = _true_ranks(env.leads)
    r = ranks[lead["id"]]
    return {"type": "prioritize", "value": r}


def naive_prioritize_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    """Guess rank from observable features only (deterministic, imperfect)."""
    lead = obs["current_lead"]
    assert lead is not None
    n = len(env.leads)
    # Crude score proxy: higher rating and more engagement -> assume better rank (lower number)
    score_proxy = lead["rating"] * 10.0
    if lead["has_website"]:
        score_proxy += 2.0
    if lead["has_contact_form"]:
        score_proxy += 1.5
    if lead["has_chat"]:
        score_proxy += 1.5
    if lead["category"] == "plumbing":
        score_proxy += 1.0
    # Map proxy to 1..n deterministically
    bucket = int(score_proxy) % n + 1
    return {"type": "prioritize", "value": bucket}


def run_task(
    policy: Callable[[dict[str, Any], LeadQualificationEnv], Action] | None = None,
) -> dict[str, Any]:
    """
    Run one episode with prioritize actions.

    Per-step quality = 1 - |pred - true_rank| / (n - 1), clipped to [0, 1].
    Final score = mean per-step quality over all n leads (deterministic).
    """
    policy = policy or oracle_prioritize_policy
    env = LeadQualificationEnv()
    obs = env.reset()
    decisions: list[dict[str, Any]] = []
    n = len(env.leads)
    qualities: list[float] = []

    while True:
        action = policy(obs, env)
        obs, reward, done, info = env.step(action)
        tr = int(info["true_rank"])
        if action["type"] == "prioritize":
            pred = int(action["value"])
            q = _normalized_rank_score(pred, tr, n)
        else:
            q = 0.0
        qualities.append(q)
        decisions.append(
            {
                "action": action,
                "reward": reward,
                "done": done,
                "info": dict(info),
                "rank_quality": q,
            }
        )
        if done:
            break

    final_score = sum(qualities) / n if n else 0.0
    final_score = max(0.0, min(1.0, float(final_score)))

    return {
        "task": "medium",
        "metric": "ranking_quality_mean",
        "final_score": final_score,
        "num_leads": n,
        "per_step_rank_quality": qualities,
        "decisions": decisions,
    }


if __name__ == "__main__":
    r = run_task()
    print(r["task"], "final_score=", r["final_score"])
