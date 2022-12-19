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
parser.add_argument('--env', type=str, default="Pendulum-v1")
args = parser.parse_args()

# Choose Gym environment
import gym
env = gym.envs.make(args.env)

# Dimension of state and action space
stateSpace = len(env.observation_space.high)
actionSpace = len(env.action_space.high)

# Initialize Vracer
from vracer import *
vracer = Vracer(stateSpace, actionSpace, learningRate=0.001, miniBatchSize=128, experienceReplaySize=131072, hiddenLayers=[128,128])

maxReward = -np.inf
rewardHistory = []

# Training loop
for episodeId in range(args.maxgen):
  
    # Initialize episode
    steps = 0
    cumulativeReward = 0
    done = False
    
    # Receive initial state from Environment
    state, _ = env.reset()
  
    episode = []

    while (not done):
 
            # Evaluate policy on current state
            action = vracer.getAction(state)
            
            # Execute action and observe reward & next state from Environment
            nextState, reward, done, _, _ = env.step(action)
            
            # Collect state-action-reward tuple
            episode.append((state, action, reward))

            # Update variables
            steps += 1
            state = nextState
            cumulativeReward += reward

    
    # Traing agent
    vracer.train(episode)
 
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
outfile = '_rewards_vracer_{}_{}.npy'.format(args.env,int(t))
np.save(outfile, np.array(rewardHistory))
