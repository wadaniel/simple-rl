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
parser.add_argument('--env', type=str, default="Pendulum-v1")
args = parser.parse_args()

# Choose Gym environment
import gym
env = gym.envs.make(args.env)
env.seed(42)

# Dimension of state and action space
stateSpace = len(env.observation_space.high)
actionSpace = len(env.action_space.high)

# Initialize Gracer
from gracer import *
gracer = Gracer(stateSpace, actionSpace, learningRate=0.0001, miniBatchSize=32, experienceReplaySize=16384, hiddenLayers=[32,32])

maxReward = -np.inf
rewardHistory = []

# Training loop
for episodeId in range(args.maxgen):
  
    # Initialize episode
    steps = 0
    cumulativeReward = 0
    done = False
    
    # Receive initial state from Environment
    state = env.reset() 
  
    episode = []

    while (not done):
 
            # Evaluate policy on current state
            action = gracer.getAction(state)
            
            # Execute action and observe reward & next state from Environment
            nextState, reward, done, _ = env.step(action)
            
            # Collect state-action-reward tuple
            episode.append((state, action, reward))

            # Update variables
            steps += 1
            state = nextState
            cumulativeReward += reward

    
    # Traing agent
    gracer.train(episode)
 
    # Statistics
    if cumulativeReward > maxReward:
        maxEpisode = episodeId
        maxReward = cumulativeReward 
    
    rewardHistory.append(cumulativeReward)
    rollingAvg = np.mean(rewardHistory[-100:])
    print("\nEpisode: {}, Number of Steps : {}, Cumulative reward: {:0.3f} (Avg. {:0.3f})".format(episodeId, steps, cumulativeReward, rollingAvg))


    if cumulativeReward >= args.maxreward or rollingAvg >= args.maxavgreward:
        print("********************* Solved ********************")
        sys.exit()
