#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('../../') # path to vracer

# Init argparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--maxgen', type=int, default=1000)
parser.add_argument('--maxreward', type=float, default=500)
parser.add_argument('--maxexp', type=float, default=10e6)
args = parser.parse_args()

# Dimension of state and action space
stateDim = 2
actionDim = 1
stepSize = 0.1
planeUpperBound = 1.

# Initialize Vracer
from vracer import *
agent = Vracer(stateDim, actionDim, maxEpisodes=args.maxgen, maxExperiences=args.maxexp, learningRate=0.001, miniBatchSize=32, experienceReplaySize=4096, hiddenLayers=[32,32])

# Training loop
while(agent.isTraining() == True):
  
    # Initialize episode
    steps = 0
    cumulativeReward = 0
    done = False
    
    # Receive initial state from Environment
    state = np.random.normal(loc=0., scale=1., size=2)

    agent.sendInitialState(state)
  
    while (steps < 100):
 
            # Evaluate policy on current state
            action = agent.getAction(state)
             
            # Calculate walking direction
            direction = np.array([np.cos(action), np.sin(action)]).flatten()

            # Move walker
            state += stepSize * direction

            # How far we walked in y direction
            reward = state[1]
            
            # Update agent
            agent.sendStateAndReward(state, reward)
            
            steps += 1
            cumulativeReward += reward
    
    # Traing agent
    agent.train()

    # Print training information
    agent.print()

    # Check termination
    if cumulativeReward >= args.maxreward:
        print("********************* Solved ********************")
        break
