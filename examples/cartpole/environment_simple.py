#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('../../') # path to simpleagent.py

from simpleagent import *

numEpisodes = 100000

# Init cartpole
from cartpole import *
cart = CartPole()

# Dimension of state and action space
stateSpace = cart.stateSpace
actionSpace = cart.actionSpace

# Initialize Vracer
sagent = SAgent(stateSpace, actionSpace, learningRate=0.0001, sigma=0.1)

# Statistics init
maxEpisode = -1
maxReward = -np.inf
rewardHistory = []

# Training loop
for episodeId in range(numEpisodes):
  
    # Reset env
    cart.reset(episodeId)
    
    # Initialize episode
    steps = 0
    cumulativeReward = 0
    done = False
    
    # Receive initial state from Environment
    state = cart.getState()
  
    rewards = []

    while (not done and steps < 500):
 
            # Evaluate policy on current state
            action = sagent.getAction(state)
            
            # Execute action and observe reward & next state from Environment
            done = cart.advance(action)
            reward = cart.getReward()
            
            # Collect state-action-reward tuple
            rewards.append(reward)

            # Update variables
            steps += 1
            state = cart.getState()
            cumulativeReward += reward
    
    # Traing agent
    sagent.train(rewards)

    # Statistics
    if cumulativeReward > maxReward:
        maxEpisode = episodeId
        maxReward = cumulativeReward 
    
    rewardHistory.append(cumulativeReward)
    rollingAvg = np.mean(rewardHistory[-100:])
    print("\nEpisode: {}, Number of Steps : {}, Cumulative reward: {:0.1f} (Avg. {:0.2f} / Max {:0.1f} at {})".format(episodeId, steps, cumulativeReward, rollingAvg, maxReward, maxEpisode))

    if cumulativeReward == 500.:
        print("*********************Solved********************")
        sys.exit()
