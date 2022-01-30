import tensorflow as tf
import numpy as np
from replaymemory import *
from network import *

from scipy.stats import norm

## Problem Configuration

seed = 42
stateSpace = 3
actionSpace = 1
numEpisodes = 1000

import gym
#env = gym.envs.make("MountainCarContinuous-v0") 
env = gym.envs.make("Pendulum-v1")  
env.seed(seed)


## Agent Configuration

learningRate = 0.0001
experienceReplaySize = 32768
#experienceReplaySize = 2048
miniBatchSize = 128
hiddenLayers = [32, 32]
learningRate = 0.0001
discountFactor = 0.99
offPolicyCutOff = 4.
offPolicyTarget = 0.1
offPolicyREFERBeta = 0.3
policyUpdatesPerExperience = 1.0


## ReplayMemory, Neural Network

replayMemory = ReplayMemory(experienceReplaySize, miniBatchSize, stateSpace, actionSpace)
valuePolicyNetwork = initValuePolicyNetwork(stateSpace, actionSpace, hiddenLayers)
optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)

## Variables

totalExperiences = 0
currentLearningRate = learningRate
policyUpdateCount = 0
experienceReplayStartSize = 0.5*experienceReplaySize
offPolicyRatio = 0.
offPolicyCurrentCutOff = offPolicyCutOff
offPolicyAnnealingRate = 5e-7
episodeHistory = []

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

for episodeId in range(numEpisodes):
  
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
            #action = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])
            
            # Execute action and observe reward & next state from Environment
            nextState, reward, done, _ = env.step(action)
            episode.append((state, action, reward, stateValue, mean, sdev))

            steps += 1
            state = nextState
            cumulativeReward += reward

    totalExperiences += len(episode)
    episodeHistory.append(cumulativeReward)
    replayMemory.processAndStoreEpisode(episode, episodeId, discountFactor)

    if replayMemory.size < experienceReplayStartSize:
        print("Collecting experiences ({:.2f}%/{:.2f}%)".format(replayMemory.size/replayMemory.memorySize*100, experienceReplayStartSize/replayMemory.memorySize*100))
        continue
    
    # Netowrk update with Mini-Batch
    for update in range(int(steps*policyUpdatesPerExperience)):
        miniBatchExpIds = replayMemory.sample()

        # Forward mini-batch 
        states = replayMemory.stateVector[miniBatchExpIds,:]
        with tf.GradientTape() as tape:
            valueMeanSigmas = valuePolicyNetwork(tf.convert_to_tensor(states))
        gradOutputWeights = tape.gradient(valueMeanSigmas, valuePolicyNetwork.trainable_variables)
        
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

        eId = replayMemory.episodeIdVector[expId]
        curId = expId
        while(curId >= 0 and replayMemory.episodeIdVector[curId] == eId):
            reward = replayMemory.importanceWeightVector[curId] * replayMemory.rewardScalingFactor
            stateValue = replayMemory.stateValueVector[curId]
            impportanceWeight = replayMemory.importanceWeightVector[curId]
            truncatedImportanceWeight = min(1., importanceWeight)
            retraceValue = stateValue + truncatedImportanceWeight * (reward + discountFactor * retraceValue - stateValue);
            replayMemory.retraceValueVector[expId] = retraceValue
            curId -= 1

        # Calculate Gradients
        vracerGradient = np.zeros(1+2*actionSpace)
       
        # V - Vtbc
        vracerGradient[0] = calculateLossGradient(replayMemory.stateValueVector[miniBatchExpIds],replayMemory.retraceValueVector[miniBatchExpIds])

        # Qret - V
        offPgLoss = replayMemory.rewardVector[miniBatchExpIds] + discountFactor * (replayMemory.isTerminalVector[miniBatchExpIds] == False) * replayMemory.retraceValueVector[(miniBatchExpIds+1)%replayMemory.size] - replayMemory.stateValueVector[miniBatchExpIds]

        importanceWeightGradients = calculateImportanceWeightGradient(replayMemory.actionVector[miniBatchExpIds,:], replayMemory.expMeanVector[miniBatchExpIds,:], replayMemory.expSdevVector[miniBatchExpIds,:], replayMemory.curMeanVector[miniBatchExpIds,:], replayMemory.curSdevVector[miniBatchExpIds,:], replayMemory.importanceWeightVector[miniBatchExpIds])
        
        vracerGradient[1:] = -offPolicyREFERBeta * np.mean(offPgLoss[:, newaxis] * importanceWeightGradients * replayMemory.isOnPolicyVector[miniBatchExpIds][:, np.newaxis])
        
        klGradients = calculateKLGradient(replayMemory.expMeanVector[miniBatchExpIds,:], replayMemory.expSdevVector[miniBatchExpIds,:], replayMemory.curMeanVector[miniBatchExpIds,:], replayMemory.curSdevVector[miniBatchExpIds,:])

        vracerGradient[1:] += (1. - offPolicyREFERBeta) * np.mean(klGradients, axis=0)

        # Update Value-Policy Network
        for grad in gradOutputWeights[-2*(actionSpace+1)]:
            grad = grad * sum(vracerGradient)

        for outIdx, grad in enumerate(vracerGradient):
            gradOutputWeights[-2*(actionSpace+1)+2*outIdx] = grad*gradOutputWeights[-2*(actionSpace+1)+2*outIdx] 
            gradOutputWeights[-2*(actionSpace+1)+2*outIdx+1] = grad*gradOutputWeights[-2*(actionSpace+1)+2*outIdx+1] 
        
        policyUpdateCount += 1
        optimizer.apply_gradients(zip(gradOutputWeights, valuePolicyNetwork.trainable_variables))
        
    # Update Variables
    offPolicyCurrentCutOff = offPolicyCutOff / (1. + offPolicyAnnealingRate * policyUpdateCount)
    offPolicyRatio = replayMemory.offPolicyCount / replayMemory.size
    currentLearningRate = learningRate / (1. + offPolicyAnnealingRate * policyUpdateCount)

    if offPolicyRatio > offPolicyTarget:
        offPolicyREFERBeta = (1. - currentLearningRate) * offPolicyREFERBeta
    else:
        offPolicyREFERBeta = (1. - currentLearningRate) * offPolicyREFERBeta + currentLearningRate


    # Print Summary
    avgPast100 = np.mean(episodeHistory[-100:])
    print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.3f} (Avg. {:0.3f}".format(episodeId, steps, cumulativeReward, avgPast100))
    print("Total Experiences: {}\nCurrent Learning Rate {}\nOff Policy Ratio {:0.3f}\nOff-Policy Ref-ER Beta {}\n Reward Scaling Factor {:0.3f}".format(replayMemory.size, currentLearningRate, offPolicyRatio, offPolicyREFERBeta, replayMemory.rewardScalingFactor))
