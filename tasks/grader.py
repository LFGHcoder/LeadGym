"""
Deterministic grader for lead qualification tasks.
Strict format enforcement: invalid decisions are counted as wasted.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

Label = Literal["hot", "warm", "cold"]


# ---------- GROUND TRUTH ----------

def _normalize_gt(ground_truth):
    n = len(ground_truth)
    if n == 0:
        return {}, {}, {}, 0

    labels = {}
    priorities = {}
    ranks = {}

    for lid, row in ground_truth.items():
        labels[lid] = row["label"]
        priorities[lid] = float(row["priority"])
        ranks[lid] = int(row["rank"])

    return labels, priorities, ranks, n


# ---------- HELPERS ----------

def _max_sum_abs_rank_error(true_ranks, n):
    if not true_ranks or n <= 0:
        return 0.0
    return float(sum(max(tr - 1, n - tr) for tr in true_ranks))


# ---------- GRADER ----------

def grade_episode(decisions, ground_truth):
    labels, _, ranks, n = _normalize_gt(ground_truth)

    if n == 0:
        return {
            "score": 0.0,
            "metrics": {
                "classification_accuracy": 0.0,
                "ranking_quality": 0.0,
                "efficiency": 0.0,
            },
        }

    total_actions = len(decisions)
    wasted = 0

    class_correct = 0
    class_total = 0

    rank_abs_sum = 0.0
    true_ranks_used = []

    for d in decisions:

        # ✅ STRICT FORMAT CHECK
        if not isinstance(d, dict) or "action" not in d or "info" not in d:
            wasted += 1
            continue

        action = d["action"]
        info = d["info"]

        kind = action.get("type")
        lid = info.get("lead_id")

        if kind not in ("classify", "prioritize", "skip"):
            wasted += 1
            continue

        if kind == "skip":
            wasted += 1
            continue

        if lid is None or lid not in ranks:
            wasted += 1
            continue

        # ---------- CLASSIFICATION ----------
        if kind == "classify":
            class_total += 1
            if action.get("value") == labels[lid]:
                class_correct += 1

        # ---------- PRIORITIZATION ----------
        elif kind == "prioritize":
            try:
                pr = int(action.get("value"))
            except (TypeError, ValueError):
                wasted += 1
                continue

            tr = ranks[lid]

            true_ranks_used.append(tr)
            rank_abs_sum += abs(pr - tr)

    # ---------- METRICS ----------

    classification_accuracy = (
        class_correct / class_total if class_total > 0 else 0.0
    )

    s_max = _max_sum_abs_rank_error(true_ranks_used, n)

    if s_max > 0:
        ranking_quality = max(0.0, min(1.0, 1.0 - rank_abs_sum / s_max))
    else:
        ranking_quality = 0.0

    if total_actions > 0:
        efficiency = max(0.0, min(1.0, 1.0 - wasted / total_actions))
    else:
        efficiency = 0.0

    score = (
        0.5 * classification_accuracy
        + 0.3 * ranking_quality
        + 0.2 * efficiency
    )

    return {
        "score": max(0.0, min(1.0, score)),
        "metrics": {
            "classification_accuracy": classification_accuracy,
            "ranking_quality": ranking_quality,
            "efficiency": efficiency,
        },
    }


# ---------- OPTIONAL UTILITY ----------

def ground_truth_from_env(env):
    return {
        L["id"]: {
            "label": env._labels[L["id"]],
            "priority": env._priorities[L["id"]],
            "rank": env._ranks[L["id"]],
        }
        for L in env.leads
    }