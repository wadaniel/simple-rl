import sys
import numpy as np
import tensorflow as tf

from replaymemory import *
import time

class Vracer:

    def __init__(self, stateDim, actionDim, **kwargs):

        # Environment configuration
        self.stateDim = stateDim
        self.actionDim = actionDim

        # Agent Configuration
        self.maxEpisodes                = kwargs.pop('maxEpisodes', 100000)
        self.maxExperiences             = kwargs.pop('maxExperiences', 100000)
        self.experienceReplaySize       = kwargs.pop('experienceReplaySize', 32768)
        self.miniBatchSize              = kwargs.pop('miniBatchSize', 128)
        self.hiddenLayers               = kwargs.pop('hiddenLayers', [128, 128])
        self.activationFunction         = kwargs.pop('activationFunction', 'tanh')
        self.learningRate               = kwargs.pop('learningRate', 0.001)
        self.discountFactor             = kwargs.pop('discountFactor', 0.99)
        self.offPolicyCutOff            = kwargs.pop('offPolicyCutOff', 4.)
        self.offPolicyTarget            = kwargs.pop('offPolicyTarget', .1)
        self.offPolicyREFERBeta         = kwargs.pop('offPolicyREFERBeta', .3)
        self.offPolicyAnnealingRate     = kwargs.pop('offPolicyAnnealingRate', 5e-7)
        self.policyUpdatesPerExperience = kwargs.pop('policyUpdatesPerExperience', 1.)
        self.verbose                    = kwargs.pop('verbose', 0)

        # Check for unused args
        if kwargs:
            raise TypeError('[VRACER] Unepxected kwargs provided: %s' % list(kwargs.keys()))

        # ReplayMemory
        self.doRescale = True
        self.replayMemory = ReplayMemory(self.experienceReplaySize, self.stateDim, self.actionDim, self.discountFactor)
        
        # Variables
        self.episodeCount               = 0
        self.totalExperiences           = 0
        self.currentLearningRate        = self.learningRate
        self.policyUpdateCount          = 0
        self.offPolicyRatio             = 0.
        self.offPolicyCurrentCutOff     = self.offPolicyCutOff
        self.experienceReplayStartSize  = 0.5*self.experienceReplaySize
        self.episodeHistory             = []
        self.currentEpisodeStates       = []
        self.currentEpisodeActions      = []
        self.currentEpisodeMeansAndSdevs = []
        self.currentEpisodeRewards      = []
  
        # Neural Network and Optimizer
        self.__initValuePolicyNetwork(self.stateDim, self.actionDim, self.hiddenLayers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.currentLearningRate)

        # Stats
        self.returnHistory = []
        self.lastEpisodeReturn = -np.infty
        self.maxEpisodeReturn = -np.infty
        self.bestEpisode = 0.
 
   
    def getValueAndPolicy(self, state):
        rescaledState = np.multiply((state - self.replayMemory.stateMean), self.replayMemory.invStateSdev)
        valueMeanSigma = self.valuePolicyNetwork(tf.convert_to_tensor([rescaledState]))
        return valueMeanSigma
    
    def getBatchValueAndPolicy(self, state):
        rescaledState = np.multiply((state - self.replayMemory.stateMean), self.replayMemory.invStateSdev)
        valueMeanSigma = self.valuePolicyNetwork(tf.convert_to_tensor(rescaledState))
        return valueMeanSigma

    def getAction(self, state):
            
        # Evaluate policy on current state
        valueMeanSdev = self.getValueAndPolicy(state)
        value = valueMeanSdev[0,0]
        mean = valueMeanSdev[0,1:self.actionDim+1]
        sdev = valueMeanSdev[0,self.actionDim+1:]
        
        # Sample action according to current policy
        action = tf.random.normal(shape=(self.actionDim,1), mean=mean, stddev=sdev)[0,:]

        # Collect mean and sigmas for later use
        self.currentEpisodeActions.append(action)
        self.currentEpisodeValues.append(value)
        self.currentEpisodeMeansAndSdevs.append((mean,sdev))
 
        return action
 
    def sendInitialState(self, state):

        if self.lastEpisodeReturn > self.maxEpisodeReturn:
            self.maxEpisodeReturn = self.lastEpisodeReturn
            self.bestEpisode = self.episodeCount

        self.episodeCount += 1
        self.lastEpisodeReturn = np.sum(self.currentEpisodeRewards)
        self.returnHistory.append(self.lastEpisodeReturn)

        # Empty episode for next episode
        self.currentEpisodeMeansAndSdevs = []
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
        
        # Mix episode with means and sigmas
        episode = [ (state, action, reward, value, mean, sdev) for (state, action, reward, value, (mean, sdev)) in zip(self.currentEpisodeStates, self.currentEpisodeActions, self.currentEpisodeRewards, self.currentEpisodeValues, self.currentEpisodeMeansAndSdevs) ] 

        # Store eisode
        self.replayMemory.processAndStoreEpisode(episode)

        # Empty container, prepare for next episode
        self.currentEpisodeMeansAndSdevs = []

        # Exit during exploration phase  
        if self.replayMemory.size < self.experienceReplayStartSize:
            print("[VRACER] Filling replay memory with experiences before training.. ({:.2f}%/{:.2f}%)".format(self.replayMemory.size/self.replayMemory.memorySize*100, self.experienceReplayStartSize/self.replayMemory.memorySize*100))
            return
 
        # Rescale states once
        if self.doRescale:
            self.replayMemory.setStateRescaling()
            self.doRescale = False

        # Measure update time
        tforward = 0.
        tretrace = 0.
        tgradient = 0.

        start = time.time()

        # Reset retrace values with scaled reward   
        if self.policyUpdateCount == 0:
            self.replayMemory.updateAllRetraceValues()
        
        numExperiences = len(episode)
        numUpdates = int(numExperiences*self.policyUpdatesPerExperience)

        # Network update with Mini-Batch
        for _ in range(numUpdates):
            
            # Sorted mini-batch
            miniBatchExpIds = self.replayMemory.sample(self.miniBatchSize)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.valuePolicyNetwork.trainable_variables)
                
                start0 = time.time()
                
                # Forward mini-batch 
                states = self.replayMemory.stateVector[miniBatchExpIds, :]
                valueMeanSdev = self.getBatchValueAndPolicy(states)
                
                stateValues = valueMeanSdev[:,0]
                curMeans = valueMeanSdev[:,1:self.actionDim+1]
                curSdevs = valueMeanSdev[:,self.actionDim+1:]

                actions = self.replayMemory.actionVector[miniBatchExpIds,:]
                expMeans = self.replayMemory.expMeanVector[miniBatchExpIds,:]
                expSdevs = self.replayMemory.expSdevVector[miniBatchExpIds,:]
           
                # Calculate importance weigts and check on-policyness
                isExpOnPolicy = self.replayMemory.isOnPolicyVector[miniBatchExpIds]
                importanceWeights = self.__calculateImportanceWeight(actions, expMeans, expSdevs, curMeans, curSdevs)
                isCurOnPolicy = tf.logical_and(tf.less(importanceWeights, self.offPolicyCurrentCutOff), tf.greater(importanceWeights, 1./self.offPolicyCurrentCutOff))
                
                # Calcuate off policy count and update on-policyness
                for idx, expId in enumerate(miniBatchExpIds):
                    if self.replayMemory.isOnPolicyVector[expId] == True and isCurOnPolicy[idx] == False:
                            self.replayMemory.offPolicyCount += 1
                    elif self.replayMemory.isOnPolicyVector[expId] == False and isCurOnPolicy[idx] == True:
                            self.replayMemory.offPolicyCount -= 1
                    self.replayMemory.isOnPolicyVector[expId] = isCurOnPolicy[idx]
          
                # Update policy parameter and importance weight
                self.replayMemory.curMeanVector[miniBatchExpIds] = curMeans
                self.replayMemory.curSdevVector[miniBatchExpIds] = curSdevs
                self.replayMemory.stateValueVector[miniBatchExpIds] = stateValues
                self.replayMemory.importanceWeightVector[miniBatchExpIds] = importanceWeights
                self.replayMemory.truncatedImportanceWeightVector[miniBatchExpIds] = np.minimum(np.ones(self.miniBatchSize), importanceWeights)
                 
                end0 = time.time()
                tforward += (end0-start0)
                start1 = time.time()

                episodeIds = self.replayMemory.episodeIdVector[miniBatchExpIds]
                idMisMatch = episodeIds[:-1] != episodeIds[1:]
                idMisMatch = np.append(idMisMatch, 1)

                # Find retrace mini-batch
                retraceMiniBatch = miniBatchExpIds[idMisMatch == 1]

                # Update retrace values in episodes 
                [ self.__updateRetraceValues(expId) for expId in retraceMiniBatch ] #TODO: this updates are expensive
                
                end1 = time.time()
                tretrace += (end1-start1)
                start2 = time.time()
                
                # Vtbcs
                Vtbcs = self.replayMemory.retraceValueVector[miniBatchExpIds]

                # Qret - V
                advantage = self.replayMemory.getScaledReward(miniBatchExpIds) + self.discountFactor * (self.replayMemory.isTerminalVector[miniBatchExpIds] == False) * self.replayMemory.retraceValueVector[(miniBatchExpIds+1)%self.replayMemory.size] - self.replayMemory.stateValueVector[miniBatchExpIds]

                # Calculate Loss
                loss = self.__calculateLoss(stateValues, curMeans, curSdevs, Vtbcs, importanceWeights, advantage, isCurOnPolicy, expMeans, expSdevs)
        
                # Calculate gradient of loss
                gradLoss = tape.gradient(loss, self.valuePolicyNetwork.trainable_variables)

                end2 = time.time()
                tgradient += (end2-start2)
            
            self.policyUpdateCount += 1
            norm = tf.math.sqrt(sum([tf.math.reduce_sum(tf.math.square(g)) for g in gradLoss]))
            if self.verbose > 0:
                print("[VRACER] Update: {}\t\tCurrent loss {:0.2f},\tGradient norm {:0.2f}\t".format(self.policyUpdateCount, loss, norm))
            self.optimizer.learning_rate = self.currentLearningRate
            self.optimizer.apply_gradients(zip(gradLoss, self.valuePolicyNetwork.trainable_variables))
            
            # Update off-policy ratio and beta
            self.offPolicyRatio = self.replayMemory.offPolicyCount / self.replayMemory.size
            if self.offPolicyRatio > self.offPolicyTarget:
                self.offPolicyREFERBeta = (1. - self.currentLearningRate) * self.offPolicyREFERBeta
            else:
                self.offPolicyREFERBeta = (1. - self.currentLearningRate) * self.offPolicyREFERBeta + self.currentLearningRate

        # Update Variables
        self.currentLearningRate = self.learningRate / (1. + self.offPolicyAnnealingRate * self.policyUpdateCount)
        self.offPolicyCurrentCutOff = self.offPolicyCutOff / (1. + self.offPolicyAnnealingRate * self.policyUpdateCount)
        
        # Measure update time
        end = time.time()

        ttotal = end-start
        pctForward = tforward/ttotal*100.
        pctRetrace = tretrace/ttotal*100.
        pctGradient = tgradient/ttotal*100.
        
        print("[VRACER] Total Experiences: {}\n[VRACER] Current Learning Rate {:0.4f}\n[VRACER] Off Policy Ratio {:0.3f}\n[VRACER] Off-Policy Ref-ER Beta {:0.4f}\n[VRACER] Reward Scaling Factor {:0.3f}\n[VRACER] Updates Per Sec: {:0.3f}\n[VRACER] Pct Forward {:0.1f}\n[VRACER] Pct Retrace {:0.1f}\n[VRACER] Pct Gradient {:0.1f}".format(self.replayMemory.totalExperiences, self.currentLearningRate, self.offPolicyRatio, self.offPolicyREFERBeta, self.replayMemory.rewardScalingFactor, numUpdates/(ttotal), pctForward, pctRetrace, pctGradient))
    
    def __initValuePolicyNetwork(self, stateDim, actionDim, hiddenLayers):
    
        inputs = tf.keras.Input(shape=(stateDim,), dtype='float32')
        for i, size in enumerate(hiddenLayers):
            if i == 0:
                x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation=self.activationFunction, dtype='float32')(inputs)
            else:
                x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation=self.activationFunction, dtype='float32')(x)


        scaledGlorot = lambda shape, dtype : 0.001*tf.keras.initializers.GlorotNormal()(shape)

        value = tf.keras.layers.Dense(1, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
        mean  = tf.keras.layers.Dense(actionDim, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
        sigma = tf.keras.layers.Dense(actionDim, kernel_initializer=scaledGlorot, activation = "softplus", dtype='float32')(x)

        outputs = tf.keras.layers.Concatenate()([value, mean, sigma])
        self.valuePolicyNetwork = tf.keras.Model(inputs=inputs, outputs=outputs, name='valuePolicyNetwork')
 
    def __calculateLoss(self, stateValues, curMeans, curSdevs, Vtbc, importanceWeights, offPgDiff, isOnPolicy, expMeans, expSdevs):
        valueLoss = 0.5*tf.losses.mean_squared_error(stateValues, Vtbc)
        negAdvantage = -tf.math.reduce_mean(tf.boolean_mask(importanceWeights*offPgDiff,isOnPolicy))
        expKLdiv = 0.5*tf.math.reduce_mean(2*tf.math.log(curSdevs/expSdevs) + (expSdevs/curSdevs)**2 + ((curMeans - expMeans) / curSdevs)**2)
        return valueLoss + self.offPolicyREFERBeta * negAdvantage + (1.- self.offPolicyREFERBeta) * expKLdiv

    def __calculateImportanceWeight(self, action, expMean, expSdev, curMean, curSdev):
        logpExpPolicy = -0.5*((action-expMean)/expSdev)**2 - tf.math.log(expSdev)
        logpCurPolicy = -0.5*((action-curMean)/curSdev)**2 - tf.math.log(curSdev)
        logImportanceWeight = tf.reduce_sum(logpCurPolicy - logpExpPolicy, 1)
        return tf.math.exp(logImportanceWeight)

    def __updateRetraceValues(self, expId):
        
        episodeId = self.replayMemory.episodeIdVector[expId]
        episodePos = self.replayMemory.episodePosVector[expId]
        episodeStart = (expId - episodePos)%self.replayMemory.size
        
        # Start id is not part of same episode
        if (self.replayMemory.episodeIdVector[episodeStart] != episodeId):
            episodeStart = min(np.argwhere(self.replayMemory.episodeIdVector == episodeId))
            # Episode is split in RM
            if (self.replayMemory.episodeIdVector[0] == episodeId and self.replayMemory.episodeIdVector[-1] == episodeId):
                episodeStart = max(np.argwhere(self.replayMemory.episodeIdVector == episodeId))
            else:
                episodeStart = min(np.argwhere(self.replayMemory.episodeIdVector == episodeId))
            episodePos = int(expId - episodeStart)%self.replayMemory.size

        episodeIdxs = np.arange(expId - episodePos, expId+1, dtype=int)%self.replayMemory.size
        
        # Extract episode values
        episodeRewards      = self.replayMemory.getScaledReward(episodeIdxs)
        episodeTrIWs        = self.replayMemory.truncatedImportanceWeightVector[episodeIdxs]
        episodeStateValues  = self.replayMemory.stateValueVector[episodeIdxs]
        episodeRetraceValues = episodeStateValues + episodeTrIWs*(episodeRewards-episodeStateValues)

        # Init retrace value for the latest sampled experience in an episode
        if (self.replayMemory.isTerminalVector[expId] != 1):
            episodeRetraceValues[-1] += episodeTrIWs[-1]*self.discountFactor*self.replayMemory.retraceValueVector[(expId+1)%self.replayMemory.size]

        # Backward update retrace value through episode (TODO: replace this op by filter)
        for idx in range(episodePos):
            episodeRetraceValues[-idx-2] += episodeTrIWs[-idx-2]*self.discountFactor*episodeRetraceValues[-idx-1]
        
        self.replayMemory.retraceValueVector[episodeIdxs] = episodeRetraceValues

    def isTraining(self):
        return self.episodeCount < self.maxEpisodes and self.totalExperiences < self.maxExperiences

    def print(self):
        avg = np.mean(self.returnHistory[-100:])
        print(f"\n[VRACER] Episode: {self.episodeCount}, Number of Steps: {self.totalExperiences}, Last Episode Return: {self.lastEpisodeReturn:.1f} (Avg. {avg:.1f} / Max {self.maxEpisodeReturn:.1f})")

