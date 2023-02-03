#!/usr/bin/env python3

import sys
import time
import numpy as np
sys.path.append('../../') # path to Vracer

# Init argparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--maxgen', type=int, default=1000)
parser.add_argument('--maxreward', type=float, default=1e6)
parser.add_argument('--maxavgreward', type=float, default=1e6)
parser.add_argument('--maxexp', type=float, default=10e6)
parser.add_argument('--env', type=str, default="Hopper-v4")
args = parser.parse_args()

# Choose Gym environment
import gym
env = gym.envs.make(args.env)

# Dimension of state and action space
stateDim = len(env.observation_space.high)
actionDim = len(env.action_space.high)

# Initialize Vracer
from vracer import *
agent = Vracer(stateDim, actionDim, maxEpisodes=args.maxgen, learningRate=0.001, miniBatchSize=32, experienceReplaySize=131072, hiddenLayers=[32,32])

# Training loop
while(agent.isTraining() == True):
    
    # Receive initial state from Environment
    state, _ = env.reset()
  
    # Initialize episode
    cumulativeReward = 0
    done = False
 
    agent.sendInitialState(state)

    while (not done):
 
            # Evaluate policy on current state
            action = agent.getAction(state)
            
            # Execute action and observe reward & next state from Environment
            state, reward, done, _, _ = env.step(action)
            
            # Update agent
            agent.sendStateAndReward(state, reward)
 
    # Traing agent
    agent.train()
 
    agent.print()

    if cumulativeReward >= args.maxreward:
        print("********************* Solved ********************")
        break
    
t = time.time()
outfile = '_rewards_vracer_{}_{}.npy'.format(args.env,int(t))
np.save(outfile, np.array(rewardHistory))
