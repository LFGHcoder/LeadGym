# LeadGym — Prospect Intelligence OpenEnv

A real-world AI environment for sales lead qualification and prioritization.
An agent learns to evaluate business leads across three difficulty levels.

## Environment

**Observation space**
- `current_lead`: id, category, has_website, has_contact_form, has_chat, rating
- `ground_truth`: true_priority (float), true_label (hot/warm/cold)
- `progress`: current_index, total_leads, finished
- `history`: actions taken so far

**Action space**
| Type | Value | Description |
|------|-------|-------------|
| `classify` | `"hot"` \| `"warm"` \| `"cold"` | Label the current lead |
| `prioritize` | integer rank | Assign a priority rank (1 = highest) |
| `skip` | — | Skip the current lead (penalty) |

**Reward** — per step: +1.0 correct classify, -1.0 wrong classify, +0.5 correct rank, -0.2 wrong/skip

## Tasks
| Task | Description | Scoring |
|------|-------------|---------|
| easy | Classification only | accuracy |
| medium | Classification + prioritization | accuracy + ranking quality |
| hard | Full qualification, shuffled leads | accuracy + ranking + efficiency |

## Setup
```bash
pip install -r requirements.txt
pip install -e .
python server.py          # starts on port 7860
python inference.py       # runs all three tasks
python test_tasks.py      # baseline agents
```

## API Endpoints
- `POST /reset` — start new episode
- `POST /step` — `{"action": {"type": "classify", "value": "hot"}}`
- `GET /state` — current observation
- `GET /health` — liveness check

## Environment Variables
| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM endpoint (default: HF router) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | HuggingFace API key |