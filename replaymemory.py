import numpy as np

class ReplayMemory:


    def __init__(self, memory_size, minibatch_size):

        self.memory_size = memory_size

        self.minibatch_size = minibatch_size

        self.experience = [None]*self.memory_size  
        self.current_index = 0
        self.size = 0


    def store(self, episodeId, state, action, reward, isTerminal, mean, sdev, retraceValue, isOnPolicy, importanceWeight, truncatedImportanceWeight):

        self.experience[self.current_index] = (episodeId, state, action, reward, isTerminal, mean, sdev, retraceValue, isOnPolicy, importanceWeight, truncatedImportanceWeight)

        self.current_index += 1

        self.size = min(self.size+1, self.memory_size)
               
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self):
        if self.size <  self.minibatch_size:
            return []

        samples_index  = np.floor(np.random.random((self.minibatch_size,))*self.size)

        samples = [self.experience[int(i)] for i in samples_index]

        return samples

    def processAndStoreEpisode(self, episode, discountFactor):
 
        retV = 0.
        retraceValues = np.zeros(len(episode))
        for idx, experience in enumerate(reversed(episode)):
            
            episodeId, state, action, reward, mean, sdev = experience

            retV = discountFactor * retV + reward
            retraceValues[-idx] = retV
    
        for idx, experience in enumerate(episode):
            episodeId, state, action, reward, mean, sdev = experience
            isTerminal = (idx == len(episode) - 1)
            self.store(episodeId, state, action, reward, isTerminal, mean, sdev, retraceValues[idx], True, 1.0, 1.0)
