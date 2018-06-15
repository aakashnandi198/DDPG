import tensorflow as tf
import random
import numpy as np

import globalFunctions as gf

class Critic:
    # initialisation
    def __init__(self, sess, env, modelName, inputStateSrc, inputActionSrc, trainable):
        self.env = env
        self.sess = sess
        self.modelName = modelName
        self.inputStateSrc = inputStateSrc
        self.inputActionSrc = inputActionSrc
        self.output = self.createCriticModel()
        self.weights = tf.trainable_variables(modelName)

        if trainable:
            # variables related to critic training
            with tf.name_scope(modelName+'Trainer'):

                # placeholder to hold the TDEstimates
                self.trainOutput = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='trainOutput')

                # metrics to minimize
                self.error = tf.subtract(self.output, self.trainOutput, name='error')
                self.mse = tf.reduce_mean(tf.square(self.error), name='mse')

                # optimizers
                self.optimizer = tf.train.AdamOptimizer(learning_rate=gf.criticLearningRate, name='criticOptimizer')
                self.trainingOp = self.optimizer.minimize(self.mse, name='trainingOp')

                # used in training actor
                self.actionGradient = tf.gradients(self.output, self.inputActionSrc, name='actionGradient')

    # create critic neural network
    def createCriticModel(self):
        hidden = []

        #input layer
        hidden.append([tf.layers.dense(tf.concat([self.inputStateSrc, self.inputActionSrc], 1), gf.criticConfig[0], name=self.modelName + "/hidden0",activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.he_normal())])

        #hidden layers
        for i in range(1, len(gf.criticConfig)):
            hidden.append([tf.layers.dense(hidden[i - 1][0], gf.criticConfig[i], name=self.modelName + "/hidden" + str(i), activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.he_normal())])

        #output layer
        output = tf.layers.dense(hidden[len(gf.criticConfig) - 1][0], 1, name=self.modelName + "/output", kernel_initializer=tf.keras.initializers.he_normal())

        return output

    # predict a Q value
    def predict(self, givenStates, givenActions):
        # reshape to input
        givenStates = givenStates.reshape((-1, self.env.observation_space.shape[0]))
        givenActions = givenActions.reshape((-1, self.env.action_space.shape[0]))

        # predict the Q values
        predictedQValue = self.sess.run(self.output, feed_dict={self.inputStateSrc: givenStates, self.inputActionSrc: givenActions})

        return predictedQValue

    # train the critic
    def train(self, targetActor, targetCritic, experience):

        if len(experience) > gf.miniBatchSize:
            # take a minibatch from experience
            miniExperience = random.sample(experience, gf.miniBatchSize)

            # extract curState, action and one step look ahead value
            initialStates = np.array([np.array(t[0]) for t in miniExperience])
            actions = np.array([np.array(t[1]) for t in miniExperience])
            TDEstimates = gf.computeTDEstimate(targetActor, targetCritic, miniExperience)

            # reshape for the placeholders
            initialStates = initialStates.reshape((gf.miniBatchSize, self.env.observation_space.shape[0]))
            actions = actions.reshape((gf.miniBatchSize, self.env.action_space.shape[0]))
            TDEstimates = TDEstimates.reshape((gf.miniBatchSize, 1))

            # finally perform the training
            self.sess.run(self.trainingOp, feed_dict={self.inputStateSrc: initialStates, self.inputActionSrc: actions, self.trainOutput: TDEstimates})
