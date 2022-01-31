import tensorflow as tf
import numpy as np
from replaymemory import *
from network import *

## Problem Configuration

seed = 42
numEpisodes = 1000

import gym
#env = gym.envs.make("MountainCarContinuous-v0") 
env = gym.envs.make("Pendulum-v1")  
env.seed(seed)

stateSpace = len(env.observation_space.high)
actionSpace = len(env.action_space.high)

## Agent Configuration

experienceReplaySize = 32768
miniBatchSize = 128
hiddenLayers = [128, 128]
learningRate = 0.001
discountFactor = 0.99
offPolicyCutOff = 4.
offPolicyTarget = 0.1
offPolicyREFERBeta = 0.3
offPolicyAnnealingRate = 5e-7
policyUpdatesPerExperience = 1.0

## ReplayMemory, Neural Network

replayMemory = ReplayMemory(experienceReplaySize, miniBatchSize, stateSpace, actionSpace, discountFactor)
valuePolicyNetwork = initValuePolicyNetwork(stateSpace, actionSpace, hiddenLayers)
optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)

## Variables

totalExperiences = 0
currentLearningRate = learningRate
policyUpdateCount = 0
offPolicyRatio = 0.
offPolicyCurrentCutOff = offPolicyCutOff
experienceReplayStartSize = 0.5*experienceReplaySize
episodeHistory = []

## Methods

def calculateLoss(valueMeanSigmas, Vtbc, importanceWeights, offPgDiff, isOnPolicy, expMeans, expSdevs):
    stateValue = valueMeanSigmas[:,0]
    curMeans = valueMeanSigmas[:,1]
    curSdevs = valueMeanSigmas[:,2]
    valueLoss = 0.5*tf.losses.mean_squared_error(stateValue, Vtbc)
    negAdvantage = -tf.math.reduce_mean(tf.boolean_mask(importanceWeights*offPgDiff,isOnPolicy))
    expKLdiv = 0.5*tf.math.reduce_mean(2*tf.math.log(curSdevs/expSdevs) + (expSdevs/curSdevs)**2 + ((curMeans - expMeans) / curSdevs)**2)
    return valueLoss + offPolicyREFERBeta * negAdvantage + (1.- offPolicyREFERBeta) * expKLdiv

def calculateLossGradient(stateValue, retraceValue):
    return np.mean(stateValue-retraceValue)

def calculateImportanceWeight(action, expMean, expSdev, curMean, curSdev):
    logpExpPolicy = -0.5*((action-expMean)/expSdev)**2 - tf.math.log(expSdev)
    logpCurPolicy = -0.5*((action-curMean)/curSdev)**2 - tf.math.log(curSdev)
    logImportanceWeight = tf.reduce_sum(logpCurPolicy - logpExpPolicy, 1)
    return tf.math.exp(logImportanceWeight)

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
    replayMemory.processAndStoreEpisode(episode)

    if replayMemory.size < experienceReplayStartSize:
        print("Collecting experiences ({:.2f}%/{:.2f}%)".format(replayMemory.size/replayMemory.memorySize*100, experienceReplayStartSize/replayMemory.memorySize*100))
        continue
 
    # Reset retrace values with scaled reward   
    if policyUpdateCount == 0:
        replayMemory.updateAllRetraceValues()
    
    # Network update with Mini-Batch
    for update in range(int(steps*policyUpdatesPerExperience)):
        
        # Sorted mini-batch
        miniBatchExpIds = replayMemory.sample()

        # Forward mini-batch 
        states = replayMemory.stateVector[miniBatchExpIds,:]
        valueMeanSigmas = valuePolicyNetwork(tf.convert_to_tensor(states))
        
        actions = replayMemory.actionVector[miniBatchExpIds,:]
        expMeans = replayMemory.expMeanVector[miniBatchExpIds,:]
        expSdevs = replayMemory.expSdevVector[miniBatchExpIds,:]
  
        stateValues = valueMeanSigmas[:,0]
        curMeans = valueMeanSigmas[:,1:actionSpace+1]
        curSdevs = valueMeanSigmas[:,1+actionSpace:]
 
        # Calculate importance weigts and check on policy
        isExpOnPolicy = replayMemory.isOnPolicyVector[miniBatchExpIds]
        importanceWeights = calculateImportanceWeight(actions, expMeans, expSdevs, curMeans, curSdevs)
        isCurOnPolicy = tf.logical_and(tf.less(importanceWeights, offPolicyCurrentCutOff), tf.greater(importanceWeights, 1./offPolicyCurrentCutOff))
        
        #print(importanceWeights)
        #print(isOnPolicy)
        
        # Calcuate off policy count and update is on policy
        for idx, expId in enumerate(miniBatchExpIds):
            if replayMemory.isOnPolicyVector[expId] == True and isCurOnPolicy[idx] == False:
                    replayMemory.offPolicyCount += 1
            elif replayMemory.isOnPolicyVector[expId] == False and isCurOnPolicy[idx] == True:
                    replayMemory.offPolicyCount -= 1
            replayMemory.isOnPolicyVector[expId] = isCurOnPolicy[idx]
  
        # Update policy parameter, importance weight
        replayMemory.curMeanVector[miniBatchExpIds] = curMeans
        replayMemory.curSdevVector[miniBatchExpIds] = curSdevs
        replayMemory.stateValueVector[miniBatchExpIds] = stateValues
        replayMemory.importanceWeightVector[miniBatchExpIds] = importanceWeights
         
        retraceMiniBatch = [miniBatchExpIds[-1]]
        
        # Find retrace mini-batch
        for i in range(1, len(miniBatchExpIds)):
            prevExpId = miniBatchExpIds[-i-1]
            if(replayMemory.episodeIdVector[expId] != replayMemory.episodeIdVector[prevExpId]):
                retraceMiniBatch.append(prevExpId)

        # Update retrace values
        for expId in retraceMiniBatch:
            retraceValue = 0
            if (replayMemory.isTerminalVector[expId] == 1):
                retraceValue = replayMemory.stateValueVector[expId]
            else:
                retraceValue = replayMemory.retraceValueVector[(expId+1)%replayMemory.size]

            curId = expId
            expEpisodeId = replayMemory.episodeIdVector[expId]

            # Backward update episode
            while(replayMemory.episodeIdVector[curId] == expEpisodeId):
                reward = replayMemory.getScaledReward(curId)
                stateValue = replayMemory.stateValueVector[curId]
                importanceWeight = replayMemory.importanceWeightVector[curId]
                truncatedImportanceWeight = min(1., importanceWeight)
                retraceValue = stateValue + truncatedImportanceWeight * (reward + discountFactor * retraceValue - stateValue);
                replayMemory.retraceValueVector[curId] = retraceValue
                curId = (curId-1)%replayMemory.size
        
        # Vtbcs
        Vtbcs = replayMemory.retraceValueVector[miniBatchExpIds]

        # Qret - V
        advantage = replayMemory.getScaledReward(miniBatchExpIds) + discountFactor * (replayMemory.isTerminalVector[miniBatchExpIds] == False) * replayMemory.retraceValueVector[(miniBatchExpIds+1)%replayMemory.size] - replayMemory.stateValueVector[miniBatchExpIds]

        # Calculate Loss and the gradient
        with tf.GradientTape() as tape:
            valueMeanSigmas = valuePolicyNetwork(tf.convert_to_tensor(states))
 
            curMeans = valueMeanSigmas[:,1:actionSpace+1]
            curSdevs = valueMeanSigmas[:,1+actionSpace:]
 
            importanceWeights = calculateImportanceWeight(actions, expMeans, expSdevs, curMeans, curSdevs)
            loss = calculateLoss(valueMeanSigmas, Vtbcs, importanceWeights, advantage, isCurOnPolicy, expMeans, expSdevs)
 
        gradLoss = tape.gradient(loss, valuePolicyNetwork.trainable_variables)
        
        policyUpdateCount += 1
        norm = tf.math.sqrt(sum([tf.math.reduce_sum(tf.math.square(g)) for g in gradLoss]))
        print("Update: {}\tCurrent loss {:0.2f},\tGradient norm {:0.2f}".format(policyUpdateCount, loss, norm))
        optimizer.apply_gradients(zip(gradLoss, valuePolicyNetwork.trainable_variables))
        
        # Update off policy ratio and beta
        offPolicyRatio = replayMemory.offPolicyCount / replayMemory.size
        if offPolicyRatio > offPolicyTarget:
            offPolicyREFERBeta = (1. - currentLearningRate) * offPolicyREFERBeta
        else:
            offPolicyREFERBeta = (1. - currentLearningRate) * offPolicyREFERBeta + currentLearningRate

    # Update Variables
    currentLearningRate = learningRate / (1. + offPolicyAnnealingRate * policyUpdateCount)
    offPolicyCurrentCutOff = offPolicyCutOff / (1. + offPolicyAnnealingRate * policyUpdateCount)
    optimizer.learning_rate = currentLearningRate

    # Print Summary
    rollingAvg = np.mean(episodeHistory[-100:])
    print("\nEpisode: {}, Number of Steps : {}, Cumulative reward: {:0.3f} (Avg. {:0.3f})".format(episodeId, steps, cumulativeReward, rollingAvg))
    print("Total Experiences: {}\nCurrent Learning Rate {}\nOff Policy Ratio {:0.3f}\nOff-Policy Ref-ER Beta {}\nReward Scaling Factor {:0.3f}".format(replayMemory.size, currentLearningRate, offPolicyRatio, offPolicyREFERBeta, replayMemory.rewardScalingFactor))
