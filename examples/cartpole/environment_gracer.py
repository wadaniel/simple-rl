#!/usr/bin/env python3

import sys
import time
import numpy as np
sys.path.append('../../') # path to gracer

# Init argparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--maxgen', type=int, default=1000)
parser.add_argument('--maxreward', type=float, default=1e6)
parser.add_argument('--maxavgreward', type=float, default=1e6)
parser.add_argument('--maxexp', type=float, default=10e6)
args = parser.parse_args()

# Init cartpole
from cartpole import *
cart = CartPole()

# Dimension of state and action space
stateSpace = cart.stateSpace
actionSpace = cart.actionSpace

# Initialize Gracer
from gracer import *
gracer = Gracer(stateSpace, actionSpace, learningRate=0.0001, miniBatchSize=32, experienceReplaySize=8192, hiddenLayers=[32,32])

# Statistics init
maxEpisode = -1
maxReward = -np.inf
rewardHistory = []

# Number of experiences
numexp = 0

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
    
    numexp += len(episode)
    rewardHistory.append(cumulativeReward)
    rollingAvg = np.mean(rewardHistory[-100:])
    print("\nEpisode: {}, Number of Steps (total): {} ({}), Cumulative reward: {:0.3f} (Avg. {:0.3f})".format(episodeId, steps, numexp, cumulativeReward, rollingAvg))

    if cumulativeReward >= args.maxreward or rollingAvg >= args.maxavgreward:
        print("********************* Solved ********************")
        break

    if numexp >= args.maxexp:
        print("********************* Terminated ********************")
        break

t = time.time()
outfile = '_rewards_gracer_{}.npy'.format(int(t))
np.save(outfile, np.array(rewardHistory))
