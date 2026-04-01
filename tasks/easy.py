"""
Easy task: run until done with classify-only actions; final score = classification accuracy [0, 1].
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
    _label_from_score,
    lead_to_score_input,
    score_business,
)


def _true_label_for_lead(lead: dict[str, Any]) -> str:
    inp = lead_to_score_input(lead)  # type: ignore[arg-type]
    p = float(score_business(inp)["priority"])
    return _label_from_score(p)


def oracle_classify_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    """Deterministic perfect classifier (matches env ground truth)."""
    lead = obs["current_lead"]
    assert lead is not None
    label = _true_label_for_lead(lead)
    return {"type": "classify", "value": label}  # type: ignore[return-value]


def naive_classify_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    """Simple heuristic (deterministic, imperfect)."""
    lead = obs["current_lead"]
    assert lead is not None
    if not lead["has_website"]:
        return {"type": "classify", "value": "warm"}
    if lead["category"] == "plumbing" and lead["rating"] >= 4.0:
        return {"type": "classify", "value": "hot"}
    if lead["rating"] >= 4.0:
        return {"type": "classify", "value": "warm"}
    return {"type": "classify", "value": "cold"}


def run_task(
    policy: Callable[[dict[str, Any], LeadQualificationEnv], Action] | None = None,
) -> dict[str, Any]:
    """
    Run one episode. Every step must be ``classify`` for a well-defined accuracy.

    Final score = (number of correct classifications) / (number of leads), in [0.0, 1.0].
    """
    policy = policy or oracle_classify_policy
    env = LeadQualificationEnv()
    obs = env.reset()
    decisions: list[dict[str, Any]] = []
    n = len(env.leads)

    while True:
        action = policy(obs, env)
        obs, reward, done, info = env.step(action)
        decisions.append(
            {
                "action": action,
                "reward": reward,
                "done": done,
                "info": dict(info),
            }
        )
        if done:
            break

    correct = sum(
        1
        for d in decisions
        if d["action"]["type"] == "classify" and d["info"].get("correct") is True
    )
    # If policy always classifies, correct <= n; score caps at 1.0
    final_score = correct / n if n else 0.0
    final_score = max(0.0, min(1.0, float(final_score)))

    return {
        "task": "easy",
        "metric": "classification_accuracy",
        "final_score": final_score,
        "num_leads": n,
        "correct_classifications": correct,
        "decisions": decisions,
    }


if __name__ == "__main__":
    r = run_task()
    print(r["task"], "final_score=", r["final_score"])
