import sys
import numpy as np

import time

class LinearAgent:

    def __init__(self, stateDim, actionDim, **kwargs):

        # Environment configuration
        self.stateDim = stateDim
        self.actionDim = actionDim

        # Agent Configuration
        self.maxEpisodes                = kwargs.pop('maxEpisodes', 100000)
        self.learningRate               = kwargs.pop('learningRate', 0.001)
        self.sigma                      = kwargs.pop('sigma', 1.)
        self.discountFactor             = kwargs.pop('discountFactor', 0.99)

        # Check for unused args
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

        # Variables
        self.episodeCount               = 0
        self.totalExperiences           = 0
        self.currentEpisodeValues       = []
        self.currentEpisodeMeans        = []
        self.currentEpisodeStates       = []
        self.currentEpisodeActions      = []
        self.currentEpisodeRewards      = []
  
        # Linear Policy and Value function
        self.policyMatrix = np.random.normal(0., 0.001, size=(stateDim, actionDim))
        self.valueMatrix = np.random.normal(0., 0.001, size=(stateDim))
        
        # Stats
        self.returnHistory = []
        self.lastEpisodeReturn = -np.infty
        self.maxEpisodeReturn = -np.infty
        self.bestEpisode = 0.
   
    def getValue(self, state):
        return np.matmul(state,self.valueMatrix)
 
    def getPolicy(self, state):
        return np.matmul(state,self.policyMatrix)

    def getAction(self):
            
        # Take last seen state
        state = self.currentEpisodeStates[-1]

        # Evaluate policy on current state
        value = self.getValue(state)
        mean = self.getPolicy(state)
 
        # Sample action according to current policy
        action = np.random.normal(loc=mean, scale=1)

        # Store values for training
        self.currentEpisodeValues.append(value)
        self.currentEpisodeMeans.append(mean)
        self.currentEpisodeActions.append(action)
        
        return action

    def sendInitialState(self, state):

        if self.lastEpisodeReturn > self.maxEpisodeReturn:
            self.maxEpisodeReturn = self.lastEpisodeReturn
            self.bestEpisode = self.episodeCount

        self.episodeCount += 1
        self.lastEpisodeReturn = np.sum(self.currentEpisodeRewards)
        self.returnHistory.append(self.lastEpisodeReturn)

        # Empty episode for next episode
        self.currentEpisodeMeans = []
        self.currentEpisodeActions = []
        self.currentEpisodeValues = []
        self.currentEpisodeStates = []
        self.currentEpisodeRewards = []

        self.currentEpisodeStates.append(state)

    def sendStateAndReward(self, state, reward):
        self.currentEpisodeStates.append(state)
        self.currentEpisodeRewards.append(reward)
        self.totalExperiences += 1
 
    def train(self):

        # Transofrm lists
        rewards = np.array(self.currentEpisodeRewards)
        values  = np.array(self.currentEpisodeValues)
        means   = np.stack(self.currentEpisodeMeans)
        states  = np.stack(self.currentEpisodeStates[:-1])
        actions = np.stack(self.currentEpisodeActions)

        episodeLength = len(rewards)
        vtbc = np.zeros(episodeLength)
        vtbc[-1] = rewards[-1]
        
        # Value calculation
        for i in range(1, episodeLength):
            vtbc[-i-1] = rewards[-i-1] + self.discountFactor*vtbc[-i]

        # Avantage estimate
        advantage = vtbc - values

        # Calculate gradient of expected advantage
        gradPolicy = -advantage@(actions-means)/episodeLength

        # Calculate gradient of loss
        gradValue = ((values - vtbc)@states)/episodeLength

        # Maximize expected advantage
        self.policyMatrix += self.learningRate * gradPolicy/(self.sigma*self.sigma)

        # Minimize loss
        self.valueMatrix -= self.learningRate * gradValue
        
        # Empty episode for next iteration
        self.currentEpisodeMeans = []
        self.currentEpisodeActions = []
        self.currentEpisodeValues = []
        self.currentEpisodeStates = []

    def isTraining(self):
        return self.episodeCount < self.maxEpisodes

    def print(self):
        avg = np.mean(self.returnHistory[-100:])
        print(f"\n[LinearAgent] Episode: {self.episodeCount}, Number of Steps: {self.totalExperiences}, Last Episode Return: {self.lastEpisodeReturn:.1f} (Avg. {avg:.1f} / Max {self.maxEpisodeReturn:.1f})")

