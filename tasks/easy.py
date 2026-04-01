"""
Easy task: classification-only episode and scoring.

Only ``classify`` actions are sent to the environment (or used for accuracy).
``prioritize`` and ``skip`` policy outputs are ignored for scoring; the episode
still advances using a deterministic oracle classify for those steps.
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
    Run until ``done``. Only ``classify`` actions from the policy are graded.

    ``prioritize`` / ``skip`` (or unknown types) are ignored for accuracy: they do
    not add to the denominator. The environment still advances using
    ``oracle_classify_policy`` on those steps.

    **final_score** = (correct policy classifications) / (total policy classify
    steps), in ``[0.0, 1.0]``. If the policy never classifies, score is ``0.0``.
    """
    policy = policy or oracle_classify_policy
    env = LeadQualificationEnv()
    obs = env.reset()
    decisions: list[dict[str, Any]] = []
    n = len(env.leads)

    classify_attempts = 0
    classify_correct = 0

    while True:
        raw = policy(obs, env)
        if raw["type"] == "classify":
            action: Action = raw
            used_for_score = True
        else:
            action = oracle_classify_policy(obs, env)
            used_for_score = False

        obs, reward, done, info = env.step(action)
        info = dict(info)

        if used_for_score:
            classify_attempts += 1
            if info.get("correct") is True:
                classify_correct += 1

        decisions.append(
            {
                "raw_action": raw,
                "action": action,
                "used_policy_classify": used_for_score,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        if done:
            break

    if classify_attempts > 0:
        final_score = classify_correct / classify_attempts
    else:
        final_score = 0.0
    final_score = max(0.0, min(1.0, float(final_score)))

    return {
        "task": "easy",
        "metric": "classification_accuracy",
        "classification_accuracy": final_score,
        "final_score": final_score,
        "num_leads": n,
        "classify_attempts": classify_attempts,
        "correct_classifications": classify_correct,
        "decisions": decisions,
    }


if __name__ == "__main__":
    r = run_task()
    print(r["task"], "final_score=", r["final_score"])
