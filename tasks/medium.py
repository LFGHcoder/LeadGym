"""
Medium task: agent must run a full classify pass, then a full prioritize pass.

Final score is **ranking_quality only** (top-3 overlap). Efficiency is not used.
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
    _label_from_score,
    lead_to_score_input,
    score_business,
)

TOP_K = 3


def _true_label_for_lead(lead: dict[str, Any]) -> str:
    inp = lead_to_score_input(lead)  # type: ignore[arg-type]
    p = float(score_business(inp)["priority"])
    return _label_from_score(p)


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


def oracle_classify_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    lead = obs["current_lead"]
    assert lead is not None
    label = _true_label_for_lead(lead)
    return {"type": "classify", "value": label}  # type: ignore[return-value]


def oracle_prioritize_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    lead = obs["current_lead"]
    assert lead is not None
    ranks = _true_ranks(env.leads)
    r = ranks[lead["id"]]
    return {"type": "prioritize", "value": r}


def naive_classify_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    lead = obs["current_lead"]
    assert lead is not None
    if not lead["has_website"]:
        return {"type": "classify", "value": "warm"}
    if lead["category"] == "plumbing" and lead["rating"] >= 4.0:
        return {"type": "classify", "value": "hot"}
    if lead["rating"] >= 4.0:
        return {"type": "classify", "value": "warm"}
    return {"type": "classify", "value": "cold"}


def naive_prioritize_policy(obs: dict[str, Any], env: LeadQualificationEnv) -> Action:
    lead = obs["current_lead"]
    assert lead is not None
    n = len(env.leads)
    score_proxy = lead["rating"] * 10.0
    if lead["has_website"]:
        score_proxy += 2.0
    if lead["has_contact_form"]:
        score_proxy += 1.5
    if lead["has_chat"]:
        score_proxy += 1.5
    if lead["category"] == "plumbing":
        score_proxy += 1.0
    bucket = int(score_proxy) % n + 1
    return {"type": "prioritize", "value": bucket}


def _run_classify_pass(
    policy: Callable[[dict[str, Any], LeadQualificationEnv], Action],
) -> tuple[list[dict[str, Any]], bool]:
    """Returns decisions and whether every policy output was ``classify``."""
    env = LeadQualificationEnv()
    obs = env.reset()
    decisions: list[dict[str, Any]] = []
    ok = True

    while True:
        raw = policy(obs, env)
        if raw["type"] != "classify":
            ok = False
            action = oracle_classify_policy(obs, env)
        else:
            action = raw

        obs, reward, done, info = env.step(action)
        decisions.append(
            {
                "phase": "classify",
                "raw_action": raw,
                "action": action,
                "reward": reward,
                "done": done,
                "info": dict(info),
            }
        )
        if done:
            break

    return decisions, ok


def _run_prioritize_pass(
    policy: Callable[[dict[str, Any], LeadQualificationEnv], Action],
) -> tuple[list[dict[str, Any]], bool]:
    """Returns decisions and whether every policy output was ``prioritize``."""
    env = LeadQualificationEnv()
    obs = env.reset()
    decisions: list[dict[str, Any]] = []
    ok = True

    while True:
        raw = policy(obs, env)
        if raw["type"] != "prioritize":
            ok = False
            action = oracle_prioritize_policy(obs, env)
        else:
            action = raw

        obs, reward, done, info = env.step(action)
        tr = int(info["true_rank"])
        pred = int(action["value"]) if action["type"] == "prioritize" else 0
        n = len(env.leads)
        q = _normalized_rank_score(pred, tr, n) if action["type"] == "prioritize" else 0.0

        decisions.append(
            {
                "phase": "prioritize",
                "raw_action": raw,
                "action": action,
                "reward": reward,
                "done": done,
                "info": dict(info),
                "rank_quality": q,
            }
        )
        if done:
            break

    return decisions, ok


def _top_k_lead_ids_by_rank(
    lead_ids: list[str],
    rank_by_id: dict[str, int],
    *,
    predicted: bool,
    pred_by_id: dict[str, int] | None = None,
    k: int,
) -> list[str]:
    """Deterministic ordering: by (rank, lead_id)."""
    if predicted:
        assert pred_by_id is not None

        def key(lid: str) -> tuple[int, str]:
            return (pred_by_id[lid], lid)

    else:

        def key(lid: str) -> tuple[int, str]:
            return (rank_by_id[lid], lid)

    ordered = sorted(lead_ids, key=key)
    kk = min(k, len(ordered))
    return ordered[:kk]


def _top_k_ranking_quality(
    env: LeadQualificationEnv,
    prioritize_decisions: list[dict[str, Any]],
    k: int,
) -> float:
    """
    Compare predicted top-k lead set to true top-k (by true global rank).
    Score = |intersection| / k.
    """
    n = len(env.leads)
    ranks = _true_ranks(env.leads)
    lead_ids = [L["id"] for L in env.leads]

    pred_by_id: dict[str, int] = {}
    for d in prioritize_decisions:
        if d["phase"] != "prioritize":
            continue
        lid = str(d["info"]["lead_id"])
        pred_by_id[lid] = int(d["action"]["value"])

    if len(pred_by_id) != n:
        return 0.0

    kk = min(k, n)
    true_top = set(_top_k_lead_ids_by_rank(lead_ids, ranks, predicted=False, k=kk, pred_by_id=None))
    pred_top = set(
        _top_k_lead_ids_by_rank(lead_ids, ranks, predicted=True, pred_by_id=pred_by_id, k=kk)
    )

    overlap = len(true_top & pred_top)
    return overlap / kk if kk > 0 else 0.0


def run_task(
    classify_policy: Callable[[dict[str, Any], LeadQualificationEnv], Action] | None = None,
    prioritize_policy: Callable[[dict[str, Any], LeadQualificationEnv], Action] | None = None,
) -> dict[str, Any]:
    """
    Two full passes over the same deterministic catalog (fresh env each time).

    - **Classify pass**: policy must return ``classify`` each step (oracle substitute
      otherwise; pass marked invalid).
    - **Prioritize pass**: policy must return ``prioritize`` each step (oracle substitute
      otherwise; pass marked invalid).

    **final_score** = top-``TOP_K`` overlap only: fraction of the true top-k lead ids
    that appear in the predicted top-k (sorted by predicted rank, ties broken by
    ``lead_id``). If either pass is invalid, score is ``0.0``.
    """
    classify_policy = classify_policy or oracle_classify_policy
    prioritize_policy = prioritize_policy or oracle_prioritize_policy

    cls_decisions, cls_ok = _run_classify_pass(classify_policy)
    pri_decisions, pri_ok = _run_prioritize_pass(prioritize_policy)

    env_ref = LeadQualificationEnv()
    n = len(env_ref.leads)

    if not cls_ok or not pri_ok:
        ranking_quality = 0.0
    else:
        ranking_quality = _top_k_ranking_quality(env_ref, pri_decisions, TOP_K)

    final_score = max(0.0, min(1.0, float(ranking_quality)))

    return {
        "task": "medium",
        "metric": "ranking_quality_top_k",
        "top_k": min(TOP_K, n),
        "ranking_quality": final_score,
        "final_score": final_score,
        "classify_pass_ok": cls_ok,
        "prioritize_pass_ok": pri_ok,
        "num_leads": n,
        "classify_decisions": cls_decisions,
        "prioritize_decisions": pri_decisions,
    }


if __name__ == "__main__":
    r = run_task()
    print(r["task"], "final_score=", r["final_score"])
