#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('../../') # path to gracer

# Init argparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--maxgen', type=int, default=1000)
parser.add_argument('--maxreward', type=float, default=1e6)
parser.add_argument('--maxavgreward', type=float, default=1e6)
args = parser.parse_args()

# Init cartpole
from cartpole import *
cart = CartPole()

# Dimension of state and action space
stateSpace = cart.stateSpace
actionSpace = cart.actionSpace

# Initialize Gracer
from gracer import *
gracer = Gracer(stateSpace, actionSpace, learningRate=0.001, miniBatchSize=32, experienceReplaySize=8192, hiddenLayers=[32,32])

# Statistics init
maxEpisode = -1
maxReward = -np.inf
rewardHistory = []

# Training loop
for episodeId in range(args.maxgen):
  
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
            action = gracer.getAction(state)
            
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
    gracer.train(episode)

    # Statistics
    if cumulativeReward > maxReward:
        maxEpisode = episodeId
        maxReward = cumulativeReward 
    
    rewardHistory.append(cumulativeReward)
    rollingAvg = np.mean(rewardHistory[-100:])
    print("\nEpisode: {}, Number of Steps : {}, Cumulative reward: {:0.1f} (Avg. {:0.2f} / Max {:0.1f} at {})".format(episodeId, steps, cumulativeReward, rollingAvg, maxReward, maxEpisode))

    if cumulativeReward >= args.maxreward or rollingAvg >= args.maxavgreward:
        print("*********************Solved********************")
        sys.exit()
