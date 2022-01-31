#!/usr/bin/env python3
import sys
import numpy as np
sys.path.append('../../') # path to vracer

from vracer import *

numEpisodes = 1000

# Init cartpole
from cartpole import *
cart = CartPole()

# Dimension of state and action space
stateSpace = cart.stateSpace
actionSpace = cart.actionSpace

# Initialize Vracer
vracer = Vracer(stateSpace, actionSpace, learningRate=0.0001, miniBatchSize=32, experienceReplaySize=10000, hiddenLayers=[32, 32])

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
  
    episode = []

    while (not done and steps < 500):
 
            # Evaluate policy on current state
            action = vracer.getAction(state)
            
            # Execute action and observe reward & next state from Environment
            done = cart.advance(action)
            reward = cart.getReward()
            
            # Collect state-action-reward tuple
            episode.append((state, action, reward))

            # Update variables
            steps += 1
            state = cart.getState()
            cumulativeReward += reward
    
    # Traing agent
    vracer.train(episode)

    # Statistics
    rewardHistory.append(cumulativeReward)
    rollingAvg = np.mean(rewardHistory[-100:])
    print("\nEpisode: {}, Number of Steps : {}, Cumulative reward: {:0.3f} (Avg. {:0.3f})".format(episodeId, steps, cumulativeReward, rollingAvg))

    if rollingAvg > 490:
        print("*********************Solved********************")
