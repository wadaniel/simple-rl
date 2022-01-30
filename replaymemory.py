import numpy as np

class ReplayMemory:


    def __init__(self, memorySize, miniBatchSize, stateDim, actionDim):

        self.memorySize = memorySize
        self.miniBatchSize = miniBatchSize
        
        self.stateDim = stateDim
        self.actionDim = actionDim
        
        self.stateVector = np.zeros((self.memorySize, self.stateDim))
        self.actionVector = np.zeros((self.memorySize, self.actionDim))
        self.rewardVector = np.zeros(self.memorySize)
        self.isTerminalVector = np.zeros(self.memorySize)
        self.episodeIdVector = np.zeros(self.memorySize)
        self.isOnPolicyVector = np.zeros(self.memorySize)
        self.importanceWeightVector = np.zeros(self.memorySize)
        self.retraceValueVector = np.zeros(self.memorySize)
        self.expMeanVector = np.zeros((self.memorySize, self.actionDim))
        self.expSdevVector = np.zeros((self.memorySize, self.actionDim))
        self.curMeanVector = np.zeros((self.memorySize, self.actionDim))
        self.curSdevVector = np.zeros((self.memorySize, self.actionDim))
        self.stateValueVector = np.zeros(self.memorySize)
        
        self.currentIndex = 0
        self.size = 0
        self.sumSquaredReward = 0
        self.rewardScalingFactor = 1.
        self.offPolicyCount = 0
        self.stateRescaling = None

    def store(self, episodeId, state, action, reward, isTerminal, stateValue, mean, sdev, retraceValue, isOnPolicy, importanceWeight):

        if(self.size == self.memorySize):
            self.offPolicyCount -= (self.isOnPolicyVector[self.currentIndex] == False)
            self.sumSquaredReward -= self.rewardVector[self.currentIndex]**2

        self.stateVector[self.currentIndex,:] = state
        self.actionVector[self.currentIndex,:] = action
        self.rewardVector[self.currentIndex] = reward
        self.isTerminalVector[self.currentIndex] = isTerminal
        self.episodeIdVector[self.currentIndex] = episodeId
        self.stateValueVector[self.currentIndex] = stateValue
        self.retraceValueVector[self.currentIndex] = retraceValue
        self.isOnPolicyVector[self.currentIndex] = isOnPolicy
        self.importanceWeightVector[self.currentIndex] = importanceWeight
        self.curMeanVector[self.currentIndex,:] = mean
        self.curSdevVector[self.currentIndex,:] = sdev
        self.expMeanVector[self.currentIndex,:] = mean
        self.expSdevVector[self.currentIndex,:] = sdev

        self.offPolicyCount += (isOnPolicy == False)
        self.sumSquaredReward += reward*reward

        self.size = min(self.size+1, self.memorySize)
        self.currentIndex = (self.currentIndex + 1) % self.memorySize

    def sample(self):
        if self.size <  self.miniBatchSize:
            return []

        sampleIndexes = np.floor(np.random.random((self.miniBatchSize,))*self.size).astype(int)
        return np.sort(sampleIndexes)

    def processAndStoreEpisode(self, episode, episodeId, discountFactor):
 
        retV = 0.
        retraceValues = np.zeros(len(episode))
        for idx, experience in enumerate(reversed(episode)):
            
            reward = experience[2]
            retV = discountFactor * retV + reward
            retraceValues[-idx] = retV
    
        for idx, experience in enumerate(episode):
            state, action, reward, stateValue, mean, sdev = experience
            isTerminal = (idx == len(episode) - 1)
            self.store(episodeId, state, action, reward, isTerminal, stateValue, mean, sdev, retraceValues[idx], True, 1.0)

        self.rewardScalingFactor = np.sqrt(self.size/self.sumSquaredReward+1e-12)
