import sys
import numpy as np

import time

class SAgent:

    def __init__(self, stateSpace, actionSpace, **kwargs):

        # Environment configuration
        self.stateSpace = stateSpace
        self.actionSpace = actionSpace

        # Agent Configuration
        self.learningRate               = kwargs.pop('learningRate', 0.001)
        self.sigma                      = kwargs.pop('sigma', 1.)
        self.discountFactor             = kwargs.pop('discountFactor', 0.99)

        # Check for unused args
        if kwargs:
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

        # Variables
        self.totalExperiences           = 0
        self.currentEpisodeValues       = []
        self.currentEpisodeMeans        = []
        self.currentEpisodeStates       = []
        self.currentEpisodeActions      = []
  
        # Linear Policy and Value function
        self.policyMatrix = np.random.normal(0., 0.001, size=(stateSpace, actionSpace))
        self.valueMatrix = np.random.normal(0., 0.001, size=(stateSpace))
   
    def getValue(self, state):
        return np.matmul(state,self.valueMatrix)
 
    def getPolicy(self, state):
        return np.matmul(state,self.policyMatrix)

    def getAction(self, state):
            
        # Evaluate policy on current state
        value = self.getValue(state)
        mean = self.getPolicy(state)
 
        # Sample action according to current policy
        action = np.random.normal(loc=mean, scale=1)

        # Store values for training
        self.currentEpisodeValues.append(value)
        self.currentEpisodeMeans.append(mean)
        self.currentEpisodeStates.append(state)
        self.currentEpisodeActions.append(action)
        self.totalExperiences += 1
        
        return action
 
    def train(self, episodeRewards):

        episodeLength = len(episodeRewards)

        # Safety check
        if episodeLength != len(self.currentEpisodeValues):
            print("[SAGENT] Error: Number of generated actions {} does not coincide with episode length ({})! Exit..".format(len(self.currentEpisodeValues), len(episodeRewards)))
            sys.exit()

        # Transofrm lists
        rewards = np.array(episodeRewards)
        values  = np.array(self.currentEpisodeValues)
        means   = np.stack(self.currentEpisodeMeans)
        states  = np.stack(self.currentEpisodeStates)
        actions = np.stack(self.currentEpisodeActions)

        vtbc = np.zeros(episodeLength)
        vtbc[-1] = rewards[-1]
        
        # Value calculation
        for i in range(1, episodeLength):
            vtbc[-i-1] = rewards[-i-1] + self.discountFactor*vtbc[-i]

        # Qret - V
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

        print("[SAGENT] Total Experiences: {}".format(self.totalExperiences))
