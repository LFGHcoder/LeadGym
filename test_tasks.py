from tasks.easy import run_task as run_easy
from tasks.medium import run_task as run_medium
from tasks.hard import run_task as run_hard

import random

# 🔴 BAD AGENT (always skip)
def bad_policy(obs, env=None):
    return {"type": "skip"}


# 🟡 RANDOM AGENT
def random_policy(obs, env=None):
    actions = ["classify", "prioritize", "skip"]
    action = random.choice(actions)

    if action == "classify":
        return {
            "type": "classify",
            "value": random.choice(["hot", "warm", "cold"])
        }
    elif action == "prioritize":
        return {
            "type": "prioritize",
            "value": random.randint(1, 5)
        }
    else:
        return {"type": "skip"}


# 🟢 PERFECT AGENT (uses ground truth)
def perfect_policy(obs, env=None):
    lead = obs.get("current_lead", {})

    if "true_label" in lead:
        return {
            "type": "classify",
            "value": lead["true_label"]
        }

    return {"type": "skip"}


# 🚀 Run tests
print("\n--- EASY TASK ---")
print("Bad:", run_easy(bad_policy))
print("Random:", run_easy(random_policy))
print("Perfect:", run_easy(perfect_policy))

print("\n--- MEDIUM TASK ---")
print("Bad:", run_medium(bad_policy))
print("Random:", run_medium(random_policy))
print("Perfect:", run_medium(perfect_policy))

print("\n--- HARD TASK ---")
print("Bad:", run_hard(bad_policy))
print("Random:", run_hard(random_policy))
print("Perfect:", run_hard(perfect_policy))