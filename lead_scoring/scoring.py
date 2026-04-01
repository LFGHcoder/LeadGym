"""
Scoring logic ported from score.js (scoreBusiness, sortResults).
Input keys match the original JS object (camelCase).
"""

from __future__ import annotations

from typing import TypedDict

from lead_scoring.lead_data import LeadScores, PipelineRow


class ScoreInput(TypedDict, total=False):
    categories: list[dict[str, str]]
    phone: str
    review_count: int
    hasForm: bool
    hasChat: bool
    hasBooking: bool
    hasPhone: bool
    missing_features: list[str]
    scrapeError: str | None


def score_business(inp: ScoreInput) -> LeadScores:
    scrape_error = inp.get("scrapeError")
    if scrape_error:
        missing = inp.get("missing_features") or []
        exp = float(len(missing)) if isinstance(missing, list) else 2.0
        return {
            "pain": 3.0,
            "fit": 1.0,
            "expansion": exp,
            "priority": 15.0,
        }

    categories = inp.get("categories") or []

    def plumbing_hit() -> bool:
        for c in categories:
            a = str(c.get("alias", "")).lower()
            t = str(c.get("title", "")).lower()
            if "plumb" in a or "plumb" in t:
                return True
        return False

    digits = "".join(ch for ch in str(inp.get("phone") or "") if ch.isdigit())
    has_yelp_phone = len(digits) >= 10

    pain = 0.0
    if not inp.get("hasForm"):
        pain += 1.0
    if not inp.get("hasChat"):
        pain += 1.0
    if not inp.get("hasBooking"):
        pain += 1.0

    fit = 0.0
    if plumbing_hit():
        fit += 1.0
    if has_yelp_phone or inp.get("hasPhone"):
        fit += 1.0
    review_count = int(inp.get("review_count") or 0)
    if review_count >= 20:
        fit += 1.0
    elif review_count >= 5:
        fit += 0.5

    missing_features = inp.get("missing_features")
    expansion = float(len(missing_features)) if isinstance(missing_features, list) else 0.0

    priority = pain * 4.0 + expansion * 2.0 + fit * 3.0

    return {
        "pain": pain,
        "fit": round(fit * 10.0) / 10.0,
        "expansion": expansion,
        "priority": round(priority * 10.0) / 10.0,
    }


def sort_results(rows: list[PipelineRow]) -> list[PipelineRow]:
    def sort_key(r: PipelineRow) -> tuple[float, float, str]:
        scores = r.get("scores") or {"priority": 0.0}
        biz = r.get("business") or {"review_count": 0, "name": ""}
        pri = float(scores.get("priority", 0.0))
        rev = float(biz.get("review_count") or 0)
        name = str(biz.get("name") or "")
        return (-pri, -rev, name)

    return sorted(rows, key=sort_key)
