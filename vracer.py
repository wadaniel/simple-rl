import sys
import numpy as np
import tensorflow as tf

from replaymemory import *
import time

class Vracer:

    def __init__(self, stateSpace, actionSpace, **kwargs):

        # Environment configuration
        self.stateSpace = stateSpace
        self.actionSpace = actionSpace

        # Agent Configuration
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
            raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

        # ReplayMemory
        self.replayMemory = ReplayMemory(self.experienceReplaySize, self.miniBatchSize, self.stateSpace, self.actionSpace, self.discountFactor)
        
        # Variables
        self.totalExperiences           = 0
        self.currentLearningRate        = self.learningRate
        self.policyUpdateCount          = 0
        self.offPolicyRatio             = 0.
        self.offPolicyCurrentCutOff     = self. offPolicyCutOff
        self.experienceReplayStartSize  = 0.5*self.experienceReplaySize
        self.episodeHistory             = []
        self.currentEpisodeMeansAndSdevs = []
  
        # Neural Network and Optimizer
        self.__initValuePolicyNetwork(self.stateSpace, self.actionSpace, self.hiddenLayers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.currentLearningRate)
   
    def getValue(self, state):
        valueMeanSigma = self.valuePolicyNetwork(tf.convert_to_tensor([state]))
        return valueMeanSigma[0,0]
 
    def getPolicy(self, state):
        valueMeanSigma = self.valuePolicyNetwork(tf.convert_to_tensor([state]))
        return valueMeanSigma[0,1:self.actionSpace+1], valueMeanSigma[0, 1:self.actionSpace:]

    def getAction(self, state):
            
        # Evaluate policy on current state
        valueMeanSigma = self.valuePolicyNetwork(tf.convert_to_tensor([state]))
            
        value = valueMeanSigma[0,0]
        mean = valueMeanSigma[0,1:self.actionSpace+1]
        sdev = valueMeanSigma[0,1+self.actionSpace:]
        
        # Collect mean and sigmas for later use
        self.currentEpisodeMeansAndSdevs.append((value,mean,sdev))

        # Sample action according to current policy
        action = tf.random.normal(shape=(self.actionSpace,1), mean=mean, stddev=sdev)[0,:]
        return action
 
    def train(self, episode):

        # Safety check
        if len(episode) != len(self.currentEpisodeMeansAndSdevs):
            print("[VRACER] Error: Number of generated actions {} does not coincide with episode length ({})! Exit..".format(len(self.currentEpisodeMeansAndSdevs), len(episode)))
            sys.exit()

        # Mix episode with means and sigmas
        episode = [ (state, action, reward, value, mean, sdev) for (state, action, reward), (value, mean, sdev) in zip(episode, self.currentEpisodeMeansAndSdevs) ] 

        # Store eisode
        self.replayMemory.processAndStoreEpisode(episode)

        # Empty container, prepare for next episode
        self.currentEpisodeMeansAndSdevs = []

        # Exit during exploration phase  
        if self.replayMemory.size < self.experienceReplayStartSize:
            print("[VRACER] Filling replay memory with experiences before training.. ({:.2f}%/{:.2f}%)".format(self.replayMemory.size/self.replayMemory.memorySize*100, self.experienceReplayStartSize/self.replayMemory.memorySize*100))
            return
 
        # Measure update time
        start = time.time()

        # Reset retrace values with scaled reward   
        if self.policyUpdateCount == 0:
            self.replayMemory.updateAllRetraceValues()
        
        numExperiences = len(episode)
        numUpdates = int(numExperiences*self.policyUpdatesPerExperience)

        # Network update with Mini-Batch
        for _ in range(numUpdates):
            
            # Sorted mini-batch
            miniBatchExpIds = self.replayMemory.sample()

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.valuePolicyNetwork.trainable_variables)
                
                # Forward mini-batch 
                states = self.replayMemory.stateVector[miniBatchExpIds,:]
                valueMeanSigmas = self.valuePolicyNetwork(tf.convert_to_tensor(states))
                
                actions = self.replayMemory.actionVector[miniBatchExpIds,:]
                expMeans = self.replayMemory.expMeanVector[miniBatchExpIds,:]
                expSdevs = self.replayMemory.expSdevVector[miniBatchExpIds,:]
          
                stateValues = valueMeanSigmas[:,0]
                curMeans = valueMeanSigmas[:,1:self.actionSpace+1]
                curSdevs = valueMeanSigmas[:,1+self.actionSpace:]
         
                # Calculate importance weigts and check on policy
                isExpOnPolicy = self.replayMemory.isOnPolicyVector[miniBatchExpIds]
                importanceWeights = self.__calculateImportanceWeight(actions, expMeans, expSdevs, curMeans, curSdevs)
                isCurOnPolicy = tf.logical_and(tf.less(importanceWeights, self.offPolicyCurrentCutOff), tf.greater(importanceWeights, 1./self.offPolicyCurrentCutOff))
                
                # Calcuate off policy count and update is on policy
                for idx, expId in enumerate(miniBatchExpIds):
                    if self.replayMemory.isOnPolicyVector[expId] == True and isCurOnPolicy[idx] == False:
                            self.replayMemory.offPolicyCount += 1
                    elif self.replayMemory.isOnPolicyVector[expId] == False and isCurOnPolicy[idx] == True:
                            self.replayMemory.offPolicyCount -= 1
                    self.replayMemory.isOnPolicyVector[expId] = isCurOnPolicy[idx]
          
                # Update policy parameter, importance weight
                self.replayMemory.curMeanVector[miniBatchExpIds] = curMeans
                self.replayMemory.curSdevVector[miniBatchExpIds] = curSdevs
                self.replayMemory.stateValueVector[miniBatchExpIds] = stateValues
                self.replayMemory.importanceWeightVector[miniBatchExpIds] = importanceWeights
                self.replayMemory.truncatedImportanceWeightVector[miniBatchExpIds] = np.minimum(np.ones(self.miniBatchSize), importanceWeights)
                 
                retraceMiniBatch = [miniBatchExpIds[-1]]
                
                # Find retrace mini-batch
                for i in range(1, len(miniBatchExpIds)):
                    prevExpId = miniBatchExpIds[-i-1]
                    if(self.replayMemory.episodeIdVector[expId] != self.replayMemory.episodeIdVector[prevExpId]):
                        retraceMiniBatch.append(prevExpId)

                # Update retrace values
                for expId in retraceMiniBatch:
                    retraceValue = 0
                    if (self.replayMemory.isTerminalVector[expId] == 1):
                        retraceValue = self.replayMemory.stateValueVector[expId]
                    else:
                        retraceValue = self.replayMemory.retraceValueVector[(expId+1)%self.replayMemory.size]

                    curId = expId
                    expEpisodeId = self.replayMemory.episodeIdVector[expId]

                    # Backward update episode
                    while(self.replayMemory.episodeIdVector[curId] == expEpisodeId):
                        reward = self.replayMemory.getScaledReward(curId)
                        stateValue = self.replayMemory.stateValueVector[curId]
                        retraceValue = stateValue + self.replayMemory.truncatedImportanceWeightVector[curId] * (reward + self.discountFactor * retraceValue - stateValue);
                        self.replayMemory.retraceValueVector[curId] = retraceValue
                        curId = (curId-1)%self.replayMemory.size
                
                # Vtbcs
                Vtbcs = self.replayMemory.retraceValueVector[miniBatchExpIds]

                # Qret - V
                advantage = self.replayMemory.getScaledReward(miniBatchExpIds) + self.discountFactor * (self.replayMemory.isTerminalVector[miniBatchExpIds] == False) * self.replayMemory.retraceValueVector[(miniBatchExpIds+1)%self.replayMemory.size] - self.replayMemory.stateValueVector[miniBatchExpIds]

                # Calculate Loss
                loss = self.__calculateLoss(valueMeanSigmas, Vtbcs, importanceWeights, advantage, isCurOnPolicy, expMeans, expSdevs)
        
                # Calculate gradient of loss
                gradLoss = tape.gradient(loss, self.valuePolicyNetwork.trainable_variables)
            
            self.policyUpdateCount += 1
            norm = tf.math.sqrt(sum([tf.math.reduce_sum(tf.math.square(g)) for g in gradLoss]))
            if self.verbose > 0:
                print("[VRACER] Update: {}\t\tCurrent loss {:0.2f},\tGradient norm {:0.2f}\t".format(self.policyUpdateCount, loss, norm))
            self.optimizer.learning_rate = self.currentLearningRate
            self.optimizer.apply_gradients(zip(gradLoss, self.valuePolicyNetwork.trainable_variables))
            
            # Update off policy ratio and beta
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

        print("[VRACER] Total Experiences: {}\n[VRACER] Current Learning Rate {}\n[VRACER] Off Policy Ratio {:0.3f}\n[VRACER] Off-Policy Ref-ER Beta {}\n[VRACER] Reward Scaling Factor {:0.3f}\n[VRACER] Updates Per Sec: {:0.3f}".format(self.replayMemory.totalExperiences, self.currentLearningRate, self.offPolicyRatio, self.offPolicyREFERBeta, self.replayMemory.rewardScalingFactor, numUpdates/(end-start)))
    
    def __initValuePolicyNetwork(self, stateSpace, actionSpace, hiddenLayers):
    
        inputs = tf.keras.Input(shape=(stateSpace,), dtype='float32')
        for i, size in enumerate(hiddenLayers):
            if i == 0:
                x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation=self.activationFunction, dtype='float32')(inputs)
            else:
                x = tf.keras.layers.Dense(size, kernel_initializer='glorot_uniform', activation=self.activationFunction, dtype='float32')(x)


        scaledGlorot = lambda shape, dtype : 0.001*tf.keras.initializers.GlorotNormal()(shape)

        value = tf.keras.layers.Dense(1, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
        mean  = tf.keras.layers.Dense(actionSpace, kernel_initializer=scaledGlorot, activation = "linear", dtype='float32')(x)
        sigma = tf.keras.layers.Dense(actionSpace, kernel_initializer=scaledGlorot, activation = "softplus", dtype='float32')(x)

        outputs = tf.keras.layers.Concatenate()([value, mean, sigma])
        self.valuePolicyNetwork = tf.keras.Model(inputs=inputs, outputs=outputs, name='valuePolicyNetwork')
 
    def __calculateLoss(self, valueMeanSigmas, Vtbc, importanceWeights, offPgDiff, isOnPolicy, expMeans, expSdevs):
        stateValue = valueMeanSigmas[:,0]
        curMeans = valueMeanSigmas[:,1]
        curSdevs = valueMeanSigmas[:,2]
        valueLoss = 0.5*tf.losses.mean_squared_error(stateValue, Vtbc)
        negAdvantage = -tf.math.reduce_mean(tf.boolean_mask(importanceWeights*offPgDiff,isOnPolicy))
        expKLdiv = 0.5*tf.math.reduce_mean(2*tf.math.log(curSdevs/expSdevs) + (expSdevs/curSdevs)**2 + ((curMeans - expMeans) / curSdevs)**2)
        return valueLoss + self.offPolicyREFERBeta * negAdvantage + (1.- self.offPolicyREFERBeta) * expKLdiv

    def __calculateImportanceWeight(self, action, expMean, expSdev, curMean, curSdev):
        logpExpPolicy = -0.5*((action-expMean)/expSdev)**2 - tf.math.log(expSdev)
        logpCurPolicy = -0.5*((action-curMean)/curSdev)**2 - tf.math.log(curSdev)
        logImportanceWeight = tf.reduce_sum(logpCurPolicy - logpExpPolicy, 1)
        return tf.math.exp(logImportanceWeight)
