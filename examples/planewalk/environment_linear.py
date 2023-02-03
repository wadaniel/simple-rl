#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('../../') # path to linearAgent

from linearAgent import *

maxEpisodes = 1000

stateDim = 2
actionDim = 1
stepSize = 0.1

# Initialize Linear RL Agent
agent = LinearAgent(stateDim, actionDim, maxEpisodes=maxEpisodes, learningRate=0.0001, sigma=0.1)

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
 
            # Evaluate policy on last seen state
            action = agent.getAction()
    
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
    if cumulativeReward > 500:
        print("*********************Solved********************")
        sys.exit()
