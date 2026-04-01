"""
Hard task: two deterministic episodes (classify all leads, then prioritize all leads),
then combine classification accuracy, mean rank quality, and efficiency.

Efficiency penalizes skip and unknown action types (unnecessary actions).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from environment import Action, LeadQualificationEnv  # noqa: E402


def _load_sibling(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tasks_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_easy = _load_sibling("easy")
_medium = _load_sibling("medium")
oracle_classify_policy = _easy.oracle_classify_policy
oracle_prioritize_policy = _medium.oracle_prioritize_policy
_normalized_rank_score = _medium._normalized_rank_score


def _is_unnecessary(action: Action) -> bool:
    return action["type"] == "skip"


def _run_classify_episode(
    policy: Callable[[dict[str, Any], LeadQualificationEnv], Action],
) -> tuple[list[dict[str, Any]], float, int, int]:
    env = LeadQualificationEnv()
    obs = env.reset()
    decisions: list[dict[str, Any]] = []
    n = len(env.leads)
    correct = 0
    unnecessary = 0

    while True:
        action = policy(obs, env)
        if _is_unnecessary(action) or action["type"] != "classify":
            unnecessary += 1
        obs, reward, done, info = env.step(action)
        if action["type"] == "classify" and info.get("correct") is True:
            correct += 1
        decisions.append({"action": action, "reward": reward, "info": dict(info), "done": done})
        if done:
            break

    cls_score = correct / n if n else 0.0
    return decisions, max(0.0, min(1.0, cls_score)), unnecessary, n


def _run_prioritize_episode(
    policy: Callable[[dict[str, Any], LeadQualificationEnv], Action],
) -> tuple[list[dict[str, Any]], float, int, int]:
    env = LeadQualificationEnv()
    obs = env.reset()
    decisions: list[dict[str, Any]] = []
    n = len(env.leads)
    qualities: list[float] = []
    unnecessary = 0

    while True:
        action = policy(obs, env)
        if _is_unnecessary(action) or action["type"] != "prioritize":
            unnecessary += 1
        obs, reward, done, info = env.step(action)
        tr = int(info["true_rank"])
        if action["type"] == "prioritize":
            q = _normalized_rank_score(int(action["value"]), tr, n)
        else:
            q = 0.0
        qualities.append(q)
        decisions.append(
            {
                "action": action,
                "reward": reward,
                "info": dict(info),
                "done": done,
                "rank_quality": q,
            }
        )
        if done:
            break

    rnk_score = sum(qualities) / n if n else 0.0
    return decisions, max(0.0, min(1.0, rnk_score)), unnecessary, n


def run_task(
    classify_policy: Callable[[dict[str, Any], LeadQualificationEnv], Action] | None = None,
    prioritize_policy: Callable[[dict[str, Any], LeadQualificationEnv], Action] | None = None,
) -> dict[str, Any]:
    """
    Two full passes over the same deterministic catalog.

    - classification_accuracy: correct classifies / n (episode 1)
    - prioritization_quality: mean normalized rank score (episode 2)
    - efficiency: 1 - (unnecessary_actions / (2 * n)), where unnecessary = skip or wrong action type

    final_score = 0.4 * classification_accuracy + 0.4 * prioritization_quality + 0.2 * efficiency
    """
    classify_policy = classify_policy or oracle_classify_policy
    prioritize_policy = prioritize_policy or oracle_prioritize_policy

    d_cls, cls_acc, un_cls, n = _run_classify_episode(classify_policy)
    d_rnk, rnk_qual, un_rnk, n2 = _run_prioritize_episode(prioritize_policy)
    assert n == n2

    total_steps = 2 * n
    unnecessary = un_cls + un_rnk
    efficiency = 1.0 - (unnecessary / total_steps) if total_steps else 1.0
    efficiency = max(0.0, min(1.0, efficiency))

    w_cls, w_rnk, w_eff = 0.4, 0.4, 0.2
    final_score = w_cls * cls_acc + w_rnk * rnk_qual + w_eff * efficiency
    final_score = max(0.0, min(1.0, float(final_score)))

    return {
        "task": "hard",
        "final_score": final_score,
        "classification_accuracy": cls_acc,
        "prioritization_quality": rnk_qual,
        "efficiency": efficiency,
        "weights": {"classification": w_cls, "prioritization": w_rnk, "efficiency": w_eff},
        "num_leads": n,
        "unnecessary_actions": unnecessary,
        "total_steps": total_steps,
        "classification_episode": {"decisions": d_cls},
        "prioritization_episode": {"decisions": d_rnk},
    }


if __name__ == "__main__":
    r = run_task()
    print(r["task"], "final_score=", r["final_score"])
