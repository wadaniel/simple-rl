import numpy as np

class ReplayMemory:


    def __init__(self, memorySize, miniBatchSize, stateDim, actionDim, discountFactor):

        self.memorySize = memorySize
        self.miniBatchSize = miniBatchSize
        
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.discountFactor = discountFactor
        
        self.stateVector = np.zeros((self.memorySize, self.stateDim), dtype=np.float32)
        self.actionVector = np.zeros((self.memorySize, self.actionDim), dtype=np.float32)
        self.rewardVector = np.zeros(self.memorySize, dtype=np.float32)
        self.isTerminalVector = np.zeros(self.memorySize, dtype=np.float32)
        self.episodeIdVector = np.zeros(self.memorySize, dtype=np.int32)
        self.episodePosVector = np.zeros(self.memorySize, dtype=np.int32)
        self.isOnPolicyVector = np.zeros(self.memorySize, dtype=np.int32)
        self.importanceWeightVector = np.zeros(self.memorySize, dtype=np.float32)
        self.truncatedImportanceWeightVector = np.zeros(self.memorySize, dtype=np.float32)
        self.retraceValueVector = np.zeros(self.memorySize, dtype=np.float32)
        self.expMeanVector = np.zeros((self.memorySize, self.actionDim), dtype=np.float32)
        self.expSdevVector = np.zeros((self.memorySize, self.actionDim), dtype=np.float32)
        self.curMeanVector = np.zeros((self.memorySize, self.actionDim), dtype=np.float32)
        self.curSdevVector = np.zeros((self.memorySize, self.actionDim), dtype=np.float32)
        self.stateValueVector = np.zeros(self.memorySize, dtype=np.float32)
        
        self.size = 0
        self.episodeId = 0
        self.currentIndex = 0
        self.offPolicyCount = 0
        self.sumSquaredReward = 0
        self.rewardScalingFactor = 1.
        self.totalExperiences = 0

    def sample(self):
        if self.size <  self.miniBatchSize:
            return []

        sampleIndexes = np.floor(np.random.random((self.miniBatchSize,))*self.size).astype(int)
        return np.sort(sampleIndexes)

    def processAndStoreEpisode(self, episode):
        retV = 0.
        retraceValues = np.zeros(len(episode))
        for idx, experience in enumerate(reversed(episode)):
            reward = experience[2]
            retV = self.discountFactor * retV + self.rewardScalingFactor*reward
            retraceValues[-idx] = retV
    
        for idx, experience in enumerate(episode):
            state, action, reward, stateValue, mean, sdev = experience
            isTerminal = (idx == len(episode) - 1)
            self.__store(idx, state, action, reward, isTerminal, stateValue, mean, sdev, retraceValues[idx])

        self.episodeId += 1
        self.rewardScalingFactor = np.sqrt(self.size/(self.sumSquaredReward+1e-12))

    def updateAllRetraceValues(self):
        if self.size == 0:
            return # nothing to do

        # backward update retrace values
        for idx in range(1, self.size+1):
            expId = (self.currentIndex-idx)%self.size
            if self.isTerminalVector[expId] == True:
                retV = self.stateValueVector[expId]
            else:
                reward = self.getScaledReward(expId)
                retV = self.stateValueVector[expId] + self.truncatedImportanceWeightVector[expId] * (reward + self.discountFactor * retV - self.stateValueVector[expId])
            
            self.retraceValueVector[expId] = retV
    
    def getScaledReward(self, expIds):
        return self.rewardScalingFactor*self.rewardVector[expIds]
    
    def __store(self, pos, state, action, reward, isTerminal, stateValue, mean, sdev, retraceValue):

        if(self.size == self.memorySize):
            self.offPolicyCount -= (self.isOnPolicyVector[self.currentIndex] == 0)
            self.sumSquaredReward -= self.rewardVector[self.currentIndex]**2

        self.stateVector[self.currentIndex,:] = state
        self.actionVector[self.currentIndex,:] = action
        self.rewardVector[self.currentIndex] = reward
        self.isTerminalVector[self.currentIndex] = isTerminal
        self.episodeIdVector[self.currentIndex] = self.episodeId
        self.episodePosVector[self.currentIndex] = pos
        self.stateValueVector[self.currentIndex] = stateValue
        self.retraceValueVector[self.currentIndex] = retraceValue
        self.isOnPolicyVector[self.currentIndex] = True
        self.importanceWeightVector[self.currentIndex] = 1.
        self.truncatedImportanceWeightVector[self.currentIndex] = 1.
        self.curMeanVector[self.currentIndex,:] = mean
        self.curSdevVector[self.currentIndex,:] = sdev
        self.expMeanVector[self.currentIndex,:] = mean
        self.expSdevVector[self.currentIndex,:] = sdev

        self.sumSquaredReward += reward*reward

        self.totalExperiences += 1
        self.size = min(self.totalExperiences, self.memorySize)
        self.currentIndex = (self.currentIndex + 1) % self.memorySize
