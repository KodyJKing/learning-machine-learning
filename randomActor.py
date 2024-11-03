import os
from time import time
import numpy as np
import gym
import matplotlib.pyplot as plt

# env = gym.make("ALE/Breakout-v5", render_mode="human")
env = gym.make("ALE/Breakout-v5")
# env = gym.make("ALE/Skiing-v5")
env.action_space.seed()

datapath = "data/breakout_random_actor_episodes_2/"
numruns = 100
numActions = env.action_space.n
actionVecs = np.identity(numActions, dtype="uint8")
nullAction = 0

for run in range(numruns):
    observations = []
    actions = []
    observations.append(env.reset())

    print("\nRun", run, "/", numruns)
    for step in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # plt.figure()
        # plt.imshow(observation)
        # plt.show()

        actionVec = actionVecs[action]

        observations.append(observation)
        actions.append(actionVec)
        if done:
            break
    env.close()

    actions.append(actionVecs[nullAction])

    print("\nSaving episode.")

    if not os.path.exists(datapath):
        os.makedirs(datapath)
    filename = datapath + str(int(time())) + "_" + str(run)
    np.savez_compressed(filename, np.array(observations), np.array(actions))

    print("\nData saved to:", filename)