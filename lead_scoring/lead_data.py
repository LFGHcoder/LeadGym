"""
Lead / business record shapes aligned with the existing Node pipeline
(fetchBusinesses mapping + mock pipeline rows in run.js).
"""

from __future__ import annotations

from typing import Any, TypedDict


class Category(TypedDict):
    alias: str
    title: str


class BusinessLead(TypedDict):
    id: str
    name: str
    phone: str
    review_count: int
    rating: float
    url: str
    categories: list[Category]
    location: dict[str, Any]


class LeadScores(TypedDict):
    pain: float
    fit: float
    expansion: float
    priority: float


class WebsiteSignals(TypedDict):
    hasForm: bool
    hasChat: bool
    hasBooking: bool
    hasPhone: bool


class PipelineRow(TypedDict, total=False):
    """Row shape used by sort_results (mirrors run.js pipeline.push structure)."""

    business: BusinessLead
    websiteUrl: str | None
    scrapeError: str | None
    signals: WebsiteSignals
    scores: LeadScores


# Same businesses as mockBusinesses.js + the run.js map (reviews -> review_count, website -> url, etc.)
_SAMPLE_SOURCE: list[dict[str, Any]] = [
    {
        "name": "Ohio Plumbing & Boiler",
        "website": "https://www.ohioplumbing.com",
        "phone": "330-555-1111",
        "reviews": 40,
        "category": "plumbing",
    },
    {
        "name": "Apollo Heating & Cooling",
        "website": "https://www.apolloheatingandcooling.com",
        "phone": "330-555-2222",
        "reviews": 80,
        "category": "hvac",
    },
    {
        "name": "Aspen Dental",
        "website": "https://www.aspendental.com",
        "phone": "330-555-3333",
        "reviews": 60,
        "category": "dental",
    },
    {
        "name": "Mr. Rooter Plumbing",
        "website": "https://www.mrrooter.com",
        "phone": "330-555-4444",
        "reviews": 100,
        "category": "plumbing",
    },
    {
        "name": "One Hour Heating & Air",
        "website": "https://www.onehourheatandair.com",
        "phone": "330-555-5555",
        "reviews": 120,
        "category": "hvac",
    },
]


def _to_business_lead(row: dict[str, Any], index: int) -> BusinessLead:
    cat = str(row["category"])
    return {
        "id": f"mock-{index}",
        "name": str(row["name"]),
        "phone": str(row["phone"]),
        "review_count": int(row["reviews"]),
        "rating": 4.5,
        "url": str(row["website"]),
        "categories": [{"alias": cat, "title": cat}],
        "location": {},
    }


SAMPLE_LEADS: list[BusinessLead] = [
    _to_business_lead(r, i) for i, r in enumerate(_SAMPLE_SOURCE)
]
