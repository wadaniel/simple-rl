import tensorflow as tf
import numpy as np
from replaymemory import *
from network import *

from scipy.stats import norm

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

learningRate = 0.0001
#experienceReplaySize = 32768
experienceReplaySize = 2048
miniBatchSize = 32
hiddenLayers = [32, 32]
learningRate = 0.0001
discountFactor = 0.99
offPolicyCutOff = 4.
offPolicyTarget = 0.1
offPolicyREFERBeta = 0.8


## ReplayMemory, Neural Network

replayMemory = ReplayMemory(experienceReplaySize, miniBatchSize, stateSpace, actionSpace)
valuePolicyNetwork = value_policy_network(stateSpace, actionSpace, hiddenLayers)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

## Variables

totalExperiences = 0
currentLearningRate = learningRate
policyUpdateCount = 0
offPolicyRatio = 0.
offPolicyCurrentCutOff = offPolicyCutOff
offPolicyAnnealingRate = 5e-7
episode_history = []

## Methods

def calculateLossGradient(stateValue, retraceValue):
    return np.mean(stateValue-retraceValue)

def calculateImportanceWeight(action, expMean, expSdev, curMean, curSdev):
    logpCurPolicy = 0.
    logpExpPolicy = 0.
    for d in range(actionSpace):
        logpCurPolicy += norm.logpdf(action[d], loc=curMean[d], scale=curSdev[d])
        logpExpPolicy += norm.logpdf(action[d], loc=expMean[d], scale=expSdev[d])
    
    logImportanceWeight = logpCurPolicy - logpExpPolicy;
    return np.exp(logImportanceWeight)

def calculateImportanceWeightGradient(action, expMean, expSdev, curMean, curSdev, importanceWeight):

    importanceWeightGradients = np.zeros((miniBatchSize,2*actionSpace))
    curActionDif = action - curMean
    curInvVar = 1./(curSdev * curSdev)
    importanceWeightGradients[:,:actionSpace] = curActionDif * curInvVar
    importanceWeightGradients[:,actionSpace:] = (curActionDif * curActionDif) * (curInvVar / curSdev) - 1. / curSdev
    importanceWeightGradients *= importanceWeight[:, np.newaxis]
    return importanceWeightGradients

def calculateKLGradient(expMean, expSdev, curMean, curSdev):
    
    klGradients = np.zeros((miniBatchSize,2*actionSpace))
    curInvSig = 1. / curSdev
    curInvVar = curInvSig * curInvSig
    curInvSig3 = curInvVar * curInvSig
    meanDiff = curMean - expMean

    klGradients[:,:actionSpace] =  meanDiff * curInvVar;

    gradTr = -curInvSig3 * expSdev * expSdev
    gradQuad = -(meanDiff * meanDiff) * curInvSig3;

    gradDet = curInvSig

    klGradients[:,actionSpace:] = gradTr + gradQuad + gradDet;
    return klGradients

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
            
            stateValue = value_mean_sigma[0,0]
            mean = value_mean_sigma[0,1:actionSpace+1]
            sdev = value_mean_sigma[0,1+actionSpace:]
            
            # Sample action according to current policy
            action = tf.random.normal(shape=(actionSpace,1), mean=mean, stddev=sdev)[0,:]
            # Clip action to bounds
            action = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])
            
            # Execute action and observe reward & next state from Environment
            nextState, reward, done, _ = env.step(action)
            episode.append((state, action, reward, stateValue, mean, sdev))

            steps += 1
            state = nextState
            cumulativeReward += reward

    totalExperiences += len(episode)
    replayMemory.processAndStoreEpisode(episode, episodeId, discountFactor)

    if replayMemory.size < 0.5*replayMemory.memorySize:
        print("Collecting experiences ({:.2f}%)".format(replayMemory.size/replayMemory.memorySize*100))
        continue
    
    miniBatchExpIds = replayMemory.sample()

    # Forward mini-batch 
    states = replayMemory.stateVector[miniBatchExpIds,:]
    valueMeanSigmas = valuePolicyNetwork(tf.convert_to_tensor(states))

    # Update meta-data
    for idx, expId in enumerate(set(miniBatchExpIds)):
        
        action = replayMemory.actionVector[expId,:]
        expMean = replayMemory.expMeanVector[expId,:]
        expSdev = replayMemory.expSdevVector[expId,:]
        isExpOnPolicy = replayMemory.isOnPolicyVector[expId]
        
        stateValue = valueMeanSigmas[idx,0]
        curMean = valueMeanSigmas[idx,1:actionSpace+1]
        curSdev = valueMeanSigmas[idx,1+actionSpace:]
        
        importanceWeight = calculateImportanceWeight(action, expMean, expSdev, curMean, curSdev)
        isCurOnPolicy = (importanceWeight <= offPolicyCurrentCutOff) and (importanceWeight >= 1./offPolicyCurrentCutOff)
        
        if isExpOnPolicy == True and isCurOnPolicy == False:
                replayMemory.offPolicyCount += 1
        elif isExpOnPolicy == False and isCurOnPolicy == True:
                replayMemory.offPolicyCount -= 1
     
        replayMemory.curSdevVector[expId] = curSdev
        replayMemory.stateValueVector[expId] = stateValue
        replayMemory.importanceWeightVector[expId] = importanceWeight
        replayMemory.isOnPolicyVector[expId] = isCurOnPolicy
    
    retraceMiniBatch = []
    
    # Find retrace mini-batch
    for i in range(1, len(miniBatchExpIds)):
        nextExpId = miniBatchExpIds[-i-1]
        if(replayMemory.episodeIdVector[expId] != replayMemory.episodeIdVector[nextExpId]):
            retraceMiniBatch.append(expId)

    for expId in retraceMiniBatch:
        
        retraceValue = 0
        if (replayMemory.isTerminalVector[expId] == 1):
            retraceValue = replayMemory.stateValueVector[expId]
        else:
            retraceValue = replayMemory.retraceValueVector[(expId+1)%replayMemory.memorySize]

    episodeId = replayMemory.episodeIdVector[expId]
    curId = expId
    while(curId >= 0 and replayMemory.episodeIdVector[curId] == episodeId):
        reward = replayMemory.importanceWeightVector[curId] * replayMemory.rewardScalingFactor
        stateValue = replayMemory.stateValueVector[curId]
        impportanceWeight = replayMemory.importanceWeightVector[curId]
        truncatedImportanceWeight = min(1., importanceWeight)
        retraceValue = stateValue + truncatedImportanceWeight * (reward + discountFactor * retraceValue - stateValue);
        replayMemory.retraceValueVector[expId] = retraceValue
        curId -= 1

    # Update Value-Policy Network
    
    lossGradient = calculateLossGradient(replayMemory.stateValueVector[miniBatchExpIds],replayMemory.retraceValueVector[miniBatchExpIds])
    print(lossGradient)

    importanceWeightGradients = calculateImportanceWeightGradient(replayMemory.actionVector[miniBatchExpIds,:], replayMemory.expMeanVector[miniBatchExpIds,:], replayMemory.expSdevVector[miniBatchExpIds,:], replayMemory.curMeanVector[miniBatchExpIds,:], replayMemory.curSdevVector[miniBatchExpIds,:], replayMemory.importanceWeightVector[miniBatchExpIds])
    
    klGradients = calculateKLGradient(replayMemory.expMeanVector[miniBatchExpIds,:], replayMemory.expSdevVector[miniBatchExpIds,:], replayMemory.curMeanVector[miniBatchExpIds,:], replayMemory.curSdevVector[miniBatchExpIds,:])


    klGradMultiplier = -(1. - offPolicyREFERBeta)
    policyGradient = np.mean(offPolicyREFERBeta * importanceWeightGradients * replayMemory.isOnPolicyVector[miniBatchExpIds][:, np.newaxis] + klGradMultiplier * klGradients, axis=0)
    print(policyGradient)
    
    # Update variables
    policyUpdateCount += 1
    offPolicyCurrentCutOff = offPolicyCutOff / (1. + offPolicyAnnealingRate * policyUpdateCount)
    offPolicyRatio = replayMemory.offPolicyCount / replayMemory.size
    episode_history.append(cumulativeReward)
    currentLearningRate = learningRate / (1. + offPolicyAnnealingRate * policyUpdateCount)

    if offPolicyRatio > offPolicyTarget:
        offPolicyREFERBeta = (1. - currentLearningRate) * offPolicyREFERBeta
    else:
        offPolicyREFERBeta = (1. - currentLearningRate) * offPolicyREFERBeta + currentLearningRate


    # Print output

    print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.3f}".format(episodeId, steps, cumulativeReward))
    print("Total Experiences: {}\nCurrent Learning Rate {}\nOff Policy Ratio {:0.3f}\nOff-Policy Ref-ER Beta {}\n Reward Scaling Factor {:0.3f}".format(replayMemory.size, currentLearningRate, offPolicyRatio, offPolicyREFERBeta, replayMemory.rewardScalingFactor))
    #print(episode_history[-10:])
    
    if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
        print("****************Solved***************")
        print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(np.mean(episode_history[-100:])))
