#!/usr/bin/env python3

import sys
import time
import numpy as np
sys.path.append('../../') # path to vracer

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
stateDim = cart.stateSpace
actionDim = cart.actionSpace

# Initialize Vracer
from vracer import *
agent = Vracer(stateDim, actionDim, maxEpisodes=args.maxgen, learningRate=0.001, miniBatchSize=32, experienceReplaySize=4096, hiddenLayers=[32,32])

# Training loop
while(agent.isTraining() == True):
  
    # Reset env
    cart.reset()
    
    # Initialize episode
    steps = 0
    cumulativeReward = 0
    done = False
    
    # Receive initial state from Environment
    state = cart.getState()

    agent.sendInitialState(state)
  
    while (not done and steps < 500):
 
            # Evaluate policy on current state
            action = agent.getAction(state)
            
            # Execute action and observe reward & next state from Environment
            done = cart.advance(action)
            reward = cart.getReward()
            state = cart.getState()
            
            # Update agent
            agent.sendStateAndReward(state, reward)
            
            steps += 1
            cumulativeReward += reward
    
    # Traing agent
    agent.train()

    agent.print()

    if cumulativeReward >= args.maxreward:
        print("********************* Solved ********************")
        break

t = time.time()
outfile = '_rewards_vracer_{}.npy'.format(int(t))
np.save(outfile, np.array(rewardHistory))
