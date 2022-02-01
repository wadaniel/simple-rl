#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('../../') # path to vracer

from vracer import *

numEpisodes = 1000

# Choose Gym environment
import gym
env = gym.envs.make("Pendulum-v1")  
#env = gym.envs.make("MountainCarContinuous-v0") 
env.seed(42)

# Dimension of state and action space
stateSpace = len(env.observation_space.high)
actionSpace = len(env.action_space.high)

# Initialize Vracer
vracer = Vracer(stateSpace, actionSpace, miniBatchSize=32, hiddenLayers=[32,32])

rewardHistory = []

# Training loop
for episodeId in range(numEpisodes):
  
    # Initialize episode
    steps = 0
    cumulativeReward = 0
    done = False
    
    # Receive initial state from Environment
    state = env.reset() 
  
    episode = []

    while (not done):
 
            # Evaluate policy on current state
            action = vracer.getAction(state)
            
            # Execute action and observe reward & next state from Environment
            nextState, reward, done, _ = env.step(action)
            
            # Collect state-action-reward tuple
            episode.append((state, action, reward))

            # Update variables
            steps += 1
            state = nextState
            cumulativeReward += reward

    
    # Traing agent
    vracer.train(episode)

    # Statistics
    rewardHistory.append(cumulativeReward)
    rollingAvg = np.mean(rewardHistory[-100:])
    print("\nEpisode: {}, Number of Steps : {}, Cumulative reward: {:0.3f} (Avg. {:0.3f})".format(episodeId, steps, cumulativeReward, rollingAvg))
