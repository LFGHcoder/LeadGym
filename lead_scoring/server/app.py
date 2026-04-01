from openenv.core.env_server import create_app

from lead_scoring.models import LeadScoringAction, LeadScoringObservation
from lead_scoring.server.lead_scoring_environment import LeadScoringEnvironment

app = create_app(
    LeadScoringEnvironment,
    LeadScoringAction,
    LeadScoringObservation,
    env_name="lead_scoring",
)
