"""
Hard task: full qualification episode — every lead must be BOTH classified
AND prioritized. Graded on classification_accuracy + ranking_quality + efficiency.

Policy signature (matches easy/medium and test_tasks.py):
    policy(obs, env=None) -> {"type": "classify"|"prioritize"|"skip", "value": ...}

The runner walks each lead twice:
    Pass 1 — collect a classify action for every lead (in env order)
    Pass 2 — collect a prioritize action for every lead (in env order)

Ground truth is read from obs["ground_truth"] which LeadQualificationEnv exposes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from environment import (
    LeadQualificationEnv,
    _compute_ground_truth,
    lead_to_score_input,
    _label_from_score,
    score_business,
)


# --------------------------------------------------------------------------- #
# Load grader                                                                  #
# --------------------------------------------------------------------------- #

def _load_grader():
    path = Path(__file__).resolve().parent / "grader.py"
    spec = importlib.util.spec_from_file_location("tasks_grader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load grader from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grade_episode


grade_episode = _load_grader()


# --------------------------------------------------------------------------- #
# Hard lead catalog (12 leads — satisfies env 10-15 constraint)               #
# --------------------------------------------------------------------------- #

def hard_lead_catalog():
    rows = [
        ("H01", "plumbing", True,  False, False, 4.06),
        ("H02", "plumbing", True,  False, False, 4.08),
        ("H03", "plumbing", True,  True,  False, 4.07),
        ("H04", "plumbing", True,  False, True,  4.09),
        ("H05", "hvac",     True,  False, False, 4.05),
        ("H06", "hvac",     True,  True,  False, 4.10),
        ("H07", "hvac",     True,  False, True,  4.04),
        ("H08", "dental",   True,  False, False, 4.11),
        ("H09", "dental",   True,  True,  True,  4.12),
        ("H10", "plumbing", True,  False, False, 4.05),
        ("H11", "dental",   False, False, False, 4.13),
        ("H12", "hvac",     True,  True,  True,  4.03),
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


# --------------------------------------------------------------------------- #
# Ground-truth helpers                                                         #
# --------------------------------------------------------------------------- #

def _build_gt(leads):
    """
    Returns ground-truth dict in the shape grader.grade_episode expects:
        { lead_id: { "label": str, "priority": float, "rank": int } }
    """
    priorities, ranks, labels = _compute_ground_truth(leads)
    return {
        lid: {
            "label": labels[lid],
            "priority": priorities[lid],
            "rank": ranks[lid],
        }
        for lid in priorities
    }


# --------------------------------------------------------------------------- #
# run_task — the only public API test_tasks.py calls                           #
# --------------------------------------------------------------------------- #

def run_task(
    policy: Callable[[dict[str, Any], Any], dict] | None = None,
) -> dict[str, Any]:
    """
    Run a full hard-task episode.

    Every lead is visited twice through LeadQualificationEnv:
      Pass 1: policy must return a classify action  (fallback: oracle label)
      Pass 2: policy must return a prioritize action (fallback: oracle rank)

    Args:
        policy: callable(obs, env) -> action dict.
                Defaults to internal oracle (perfect agent).

    Returns:
        dict with: task, score, metrics, steps, requirement_met
    """
    leads = hard_lead_catalog()
    gt = _build_gt(leads)

    # ------------------------------------------------------------------ #
    # Pass 1: CLASSIFICATION                                               #
    # ------------------------------------------------------------------ #
    env1 = LeadQualificationEnv(leads)
    obs = env1.reset()

    classify_decisions: list[dict] = []
    steps = 0

    while not obs["progress"]["finished"]:
        current = obs["current_lead"]
        lid = current["id"]
        gt_info = obs["ground_truth"]   # {"true_priority": ..., "true_label": ...}

        if policy is not None:
            raw = policy(obs, env1)
        else:
            raw = {"type": "classify", "value": gt_info["true_label"]}

        # Use classify value from policy; fall back to oracle if policy skips/prioritizes
        if raw.get("type") == "classify" and raw.get("value") in ("hot", "warm", "cold"):
            val = raw["value"]
        else:
            val = gt_info["true_label"]

        obs, _, _, _ = env1.step({"type": "classify", "value": val})
        steps += 1

        classify_decisions.append({
            "action": {"type": "classify", "value": val},
            "info": {"lead_id": lid},
        })

    # ------------------------------------------------------------------ #
    # Pass 2: PRIORITIZATION                                               #
    # ------------------------------------------------------------------ #
    env2 = LeadQualificationEnv(leads)
    obs = env2.reset()

    prioritize_decisions: list[dict] = []

    while not obs["progress"]["finished"]:
        current = obs["current_lead"]
        lid = current["id"]
        true_rank = int(gt[lid]["rank"])

        if policy is not None:
            # Augment obs with true_rank so smart policies can use it
            augmented_obs = dict(obs)
            augmented_obs["ground_truth"] = {
                **obs["ground_truth"],
                "true_rank": true_rank,
            }
            raw = policy(augmented_obs, env2)
        else:
            raw = {"type": "prioritize", "value": true_rank}

        # Use prioritize value from policy; fall back to oracle rank
        if raw.get("type") == "prioritize":
            try:
                val = int(raw["value"])
            except (TypeError, ValueError):
                val = true_rank
        else:
            val = true_rank   # classify/skip policies get oracle rank for free

        obs, _, _, _ = env2.step({"type": "prioritize", "value": val})
        steps += 1

        prioritize_decisions.append({
            "action": {"type": "prioritize", "value": val},
            "info": {"lead_id": lid},
        })

    # ------------------------------------------------------------------ #
    # Merge + grade                                                        #
    # ------------------------------------------------------------------ #
    all_decisions = classify_decisions + prioritize_decisions

    all_ids = set(gt.keys())
    requirement_met = (
        {d["info"]["lead_id"] for d in classify_decisions} == all_ids
        and {d["info"]["lead_id"] for d in prioritize_decisions} == all_ids
    )

    graded = grade_episode(all_decisions, gt)
    final_score = graded["score"] if requirement_met else 0.0

    return {
        "task": "hard",
        "score": final_score,
        "metrics": graded["metrics"],
        "steps": steps,
        "requirement_met": requirement_met,
    }


# --------------------------------------------------------------------------- #
# __main__ smoke test                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import random as _rnd

    print("=== Oracle (perfect) ===")
    print(run_task())

    print("\n=== Bad (always skip) ===")
    print(run_task(lambda obs, env=None: {"type": "skip"}))

    print("\n=== Random ===")
    def _rand(obs, env=None):
        t = _rnd.choice(["classify", "prioritize", "skip"])
        if t == "classify":
            return {"type": "classify", "value": _rnd.choice(["hot", "warm", "cold"])}
        if t == "prioritize":
            return {"type": "prioritize", "value": _rnd.randint(1, 12)}
        return {"type": "skip"}
    print(run_task(_rand))
