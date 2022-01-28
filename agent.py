import tensorflow as tf
import numpy as np
from replaymemory import *
from network import *

## Problem Configuration

seed = 42
stateSpace = 2
actionSpace = 1
num_episodes = 100

import gym
env = gym.envs.make("MountainCarContinuous-v0") 
#env = gym.envs.make("Pendulum-v1")  
env.seed(seed)


## Agent Configuration

experienceReplaySize = 2**18
miniBatchSize = 128
hiddenLayers = [128, 128]
learningRate = 0.0001
discountFactor = 0.99


## ReplayMemory, Neural Network

replayMemory = ReplayMemory(experienceReplaySize, miniBatchSize)
valuePolicyNetwork = value_policy_network(stateSpace, actionSpace, hiddenLayers)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


## History

episode_history = []


## Training loop

for episodeId in range(num_episodes):
  
    # Initialize episode
    steps = 0
    cumulativeReward = 0 
    done = False
    
    # Receive initial state from Environment
    state = env.reset() 
        
    episode = []   
    while (not done):
 
            # Evaluate policy on current state
            value_mean_sigma = valuePolicyNetwork(tf.convert_to_tensor([state]))
            
            mean = value_mean_sigma[0,: actionSpace]
            sdev = value_mean_sigma[0,1+actionSpace:]
            
            # Sample action according to current policy
            action = tf.random.normal(shape=(actionSpace,1), mean=mean, stddev=sdev)[0,:]
            # Clip action to bounds
            action = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])
            
            # Execute action and observe reward & next state from Environment
            next_state, reward, done, _ = env.step(action)

            episode.append((episodeId, state, action, reward, mean, sdev))

            #V_state = valueNetwork(state)
               
            state = next_state
            cumulativeReward += reward

    replayMemory.processAndStoreEpisode(episode, discountFactor)

    episode_history.append(cumulativeReward)
    print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}".format(
        episodeId, steps, cumulativeReward))
    print(episode_history[-10:])
    
    if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
        print("****************Solved***************")
        print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(
            np.mean(episode_history[-100:])))
