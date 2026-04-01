from environment import LeadQualificationEnv

env = LeadQualificationEnv()
obs = env.reset()

print("Initial:", obs)

done = False

while not done:
    action = {
        "type": "skip"
    }

    obs, reward, done, info = env.step(action)
    print("Reward:", reward, "| Done:", done)

print("Finished")