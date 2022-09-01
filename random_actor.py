import os
from time import time
import numpy as np
import gym
import matplotlib.pyplot as plt

# env = gym.make("ALE/Breakout-v5", render_mode="human")
env = gym.make("ALE/Breakout-v5")
env.action_space.seed()

numruns = 1000

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
        observations.append(observation)
        actions.append(action)
        if done:
            break
    env.close()

    print("\nSaving episode.")

    datapath = "data/breakout_random_actor_episodes/"
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    filename = datapath + str(int(time()))
    np.savez_compressed(filename, np.array(observations), np.array(actions))

    print("\nData saved to:", filename)