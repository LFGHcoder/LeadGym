"""
Centralized deterministic grader for a completed episode.

Expects each decision to include at least ``action`` and typically ``info`` from
``LeadQualificationEnv.step`` (``lead_id``, ``true_label``, ``true_rank``, ...).
Ground truth may supply ``rank`` or it will be derived from ``priority`` (and ``label`` is unused for ranking).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

Label = Literal["hot", "warm", "cold"]


def _normalize_gt(
    ground_truth: dict[str, dict[str, Any]],
) -> tuple[dict[str, Label], dict[str, float], dict[str, int], int]:
    """Returns labels, priorities, ranks, n. Ranks derived deterministically if missing."""
    n = len(ground_truth)
    if n == 0:
        return {}, {}, {}, 0

    labels: dict[str, Label] = {}
    priorities: dict[str, float] = {}
    ranks: dict[str, int] = {}

    for lid, row in ground_truth.items():
        labels[lid] = row["label"]  # type: ignore[assignment]
        priorities[lid] = float(row["priority"])

    explicit = all("rank" in ground_truth[lid] for lid in ground_truth)
    if explicit:
        for lid in ground_truth:
            ranks[lid] = int(ground_truth[lid]["rank"])
    else:
        order = sorted(
            ground_truth.keys(),
            key=lambda x: (-priorities[x], x),
        )
        ranks = {lid: i + 1 for i, lid in enumerate(order)}

    return labels, priorities, ranks, n


def _lead_id_for_step(
    decisions: list[dict[str, Any]],
    step_index: int,
    lead_order: list[str] | None,
) -> str | None:
    d = decisions[step_index]
    info = d.get("info") or {}
    lid = info.get("lead_id")
    if lid is not None:
        return str(lid)
    if lead_order is not None and step_index < len(lead_order):
        return lead_order[step_index]
    return None


def _max_sum_abs_rank_error(true_ranks: list[int], n: int) -> float:
    """
    Worst-case sum_i |p_i - t_i| when each predicted rank may be any integer in [1, n],
    chosen independently per step (upper bound achieved by p in {1, n}).
    """
    if not true_ranks or n <= 0:
        return 0.0
    return float(sum(max(tr - 1, n - tr) for tr in true_ranks))


def grade_episode(
    decisions: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    *,
    lead_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    Grade a completed episode.

    Parameters
    ----------
    decisions:
        Ordered list of per-step records (e.g. ``{"action": ..., "info": {...}}``).
    ground_truth:
        ``lead_id -> {"label": "hot"|"warm"|"cold", "priority": float [, "rank": int]}``.
        If ``rank`` is omitted for any lead, ranks are derived by sorting
        ``(-priority, lead_id)``.
    lead_order:
        Fallback when a decision's ``info`` lacks ``lead_id`` (index aligns with step).

    Returns
    -------
    ``{"score": float, "metrics": {...}}`` with all values in [0.0, 1.0].
    """
    labels, _priorities, ranks, n = _normalize_gt(ground_truth)
    if n == 0:
        out = {
            "classification_accuracy": 0.0,
            "ranking_quality": 0.0,
            "efficiency": 0.0,
        }
        return {"score": 0.0, "metrics": out}

    total_actions = len(decisions)
    wasted = 0

    class_correct = 0
    class_total = 0

    rank_abs_sum = 0.0
    true_ranks_used: list[int] = []

    for i, d in enumerate(decisions):
        action = d.get("action") or {}
        kind = action.get("type")

        if kind not in ("classify", "prioritize", "skip"):
            wasted += 1
            continue
        if kind == "skip":
            wasted += 1
            continue

        lid = _lead_id_for_step(decisions, i, lead_order)
        if lid is None or lid not in ranks:
            wasted += 1
            continue

        if kind == "classify":
            class_total += 1
            pred = action.get("value")
            if pred == labels[lid]:
                class_correct += 1
        elif kind == "prioritize":
            tr = ranks[lid]
            pr = int(action["value"])
            true_ranks_used.append(tr)
            rank_abs_sum += abs(pr - tr)

    classification_accuracy = (class_correct / class_total) if class_total > 0 else 0.0

    s_max = _max_sum_abs_rank_error(true_ranks_used, n)
    if s_max > 0.0:
        ranking_quality = max(0.0, min(1.0, 1.0 - rank_abs_sum / s_max))
    elif rank_abs_sum == 0.0:
        ranking_quality = 0.0
    else:
        ranking_quality = 0.0

    if total_actions > 0:
        efficiency = max(0.0, min(1.0, 1.0 - wasted / total_actions))
    else:
        efficiency = 0.0

    w_cls, w_rank, w_eff = 0.5, 0.3, 0.2
    score = (
        w_cls * classification_accuracy
        + w_rank * ranking_quality
        + w_eff * efficiency
    )
    score = max(0.0, min(1.0, float(score)))

    metrics = {
        "classification_accuracy": max(0.0, min(1.0, float(classification_accuracy))),
        "ranking_quality": max(0.0, min(1.0, float(ranking_quality))),
        "efficiency": max(0.0, min(1.0, float(efficiency))),
    }

    return {
        "score": score,
        "metrics": metrics,
    }


def ground_truth_from_env(env: Any) -> dict[str, dict[str, Any]]:
    """Build grader input from a ``LeadQualificationEnv`` (optional helper)."""
    out: dict[str, dict[str, Any]] = {}
    for L in env.leads:
        lid = L["id"]
        out[lid] = {
            "label": env._labels[lid],
            "priority": env._priorities[lid],
            "rank": env._ranks[lid],
        }
    return out


if __name__ == "__main__":
    import importlib.util

    path = Path(__file__).resolve().parent / "easy.py"
    spec = importlib.util.spec_from_file_location("task_easy", path)
    assert spec and spec.loader
    easy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(easy)

    result = easy.run_task()
    from environment import LeadQualificationEnv

    env = LeadQualificationEnv()
    gt = ground_truth_from_env(env)
    graded = grade_episode(result["decisions"], gt)
    print("grader score", graded["score"], graded["metrics"])
