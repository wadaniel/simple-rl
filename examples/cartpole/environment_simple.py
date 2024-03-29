#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('../../') # path to linearAgent.py

from linearAgent import *

maxEpisodes = 100000

# Init cartpole
from cartpole import *
cart = CartPole()

# Dimension of state and action space
stateDim = cart.stateSpace
actionDim = cart.actionSpace

# Initialize Simple RL Agent
agent = LinearAgent(stateDim, actionDim, maxEpisodes=maxEpisodes, learningRate=0.0001, sigma=0.1)

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
 
            # Evaluate policy on last seen state
            action = agent.getAction()
            
            # Apply action and observe reward & next state from Environment
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

    if cumulativeReward == 500.:
        print("*********************Solved********************")
        sys.exit()
