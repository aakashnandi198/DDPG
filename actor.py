import tensorflow as tf
import random
import numpy as np

import globalFunctions as gf

class Actor:
    # initialisation
    def __init__(self, sess, env, modelName, inputStateSrc, inputActionSrc, assCritic, trainable):
        self.env = env
        self.sess = sess
        self.modelName = modelName
        self.inputStateSrc = inputStateSrc
        self.inputActionSrc = inputActionSrc
        self.assCritic = assCritic
        self.output = self.createActorModel()
        self.weights = tf.trainable_variables(modelName)

        # placeholder to capture action gradients from critic
        self.criticActionGradient = tf.placeholder(dtype=tf.float32, shape=(None,env.action_space.shape[0]),name='criticActionGradient')

        if trainable:
            with tf.name_scope(modelName+'Trainer'):

                # how does the actor change around a given state input s'
                self.actorGrads = tf.gradients(self.output, self.weights, -self.criticActionGradient, name='actorGrads')

                # actor training operation
                self.optimizer = tf.train.AdamOptimizer(learning_rate=gf.actorLearningRate,name='ActorOptimizer')
                self.trainingOp = self.optimizer.apply_gradients(zip(self.actorGrads, self.weights))

    # create the actor neural network
    def createActorModel(self):
        hidden = []

        #input layer
        hidden.append([tf.layers.dense(self.inputStateSrc, gf.actorConfig[0], name=self.modelName + "/hidden0", activation=tf.nn.elu,kernel_initializer=tf.keras.initializers.he_normal())])

        #hidden layers
        for i in range(1, len(gf.actorConfig)):
            hidden.append([tf.layers.dense(hidden[i - 1][0], gf.actorConfig[i], name=self.modelName + "/hidden" + str(i),activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.he_normal())])

        #output layer
        output = tf.layers.dense(hidden[len(gf.actorConfig) - 1][0], self.env.action_space.shape[0], name=self.modelName + "/output",activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.he_normal())

        #scale the output
        scaledOutput = tf.multiply(self.env.action_space.high, output)

        return scaledOutput

    # compute the action
    def act(self, givenState):

        # convert the given state to input format
        givenState = givenState.reshape((-1, self.env.observation_space.shape[0]))

        # predict an action based on state input
        predictedAction = self.sess.run(self.output, feed_dict={self.inputStateSrc: givenState})

        return predictedAction

    # train the actor
    def train(self, experience):

        if len(experience) > gf.miniBatchSize:
            #sample a miniexperience from experience
            miniExperience = random.sample(experience, gf.miniBatchSize)

            # extract curState
            curStates = np.array([np.array(t[0]) for t in miniExperience])

            # reshape for the placeholders
            curStates = curStates.reshape((gf.miniBatchSize, self.env.observation_space.shape[0]))
            actions = self.act(curStates)

            # compute the criticActionGradient here
            actionGrads = self.sess.run(self.assCritic.actionGradient, feed_dict={self.assCritic.inputStateSrc: curStates, self.assCritic.inputActionSrc: actions})

            #perform training
            self.sess.run(self.trainingOp, feed_dict={self.inputStateSrc: curStates, self.criticActionGradient: actionGrads[0]})
