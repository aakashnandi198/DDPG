import tensorflow as tf
import random
import collections
from time import sleep

import actor as ac
import critic as cr
import globalFunctions as gf


class ActorCritic:

    # initialisation function
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.experience = collections.deque(maxlen=gf.episodeLength*gf.numEpisodes)

        # seed random number
        random.seed()

        # create placeholders for input action and state
        with tf.name_scope('InputPlaceHolders'):
            self.inputState = tf.placeholder(dtype=tf.float32, shape=(None, self.env.observation_space.shape[0]), name='inputState')
            self.inputAction = tf.placeholder(dtype=tf.float32, shape=(None, self.env.action_space.shape[0]), name='inputAction')

        # create critic and target critic
        self.critic = cr.Critic(sess, env, 'critic', self.inputState, self.inputAction, True)
        self.targetCritic = cr.Critic(sess, env, 'targetCritic', self.inputState, self.inputAction, False)

        # create actor and target actor
        self.actor = ac.Actor(sess, env, 'actor', self.inputState, self.inputAction, self.critic, True)
        self.targetActor = ac.Actor(sess, env, 'targetActor', self.inputState, self.inputAction, self.targetCritic, False)

        # create weight update operations
        with tf.name_scope('weightedCopy'):
            self.tau = tf.placeholder(dtype=tf.float32, shape=(), name='tau')
            self.update_target_actor = [tf.assign(self.targetActor.weights[i], (tf.multiply(self.actor.weights[i], self.tau) +tf.multiply(self.targetActor.weights[i], 1. - self.tau)))for i in range(len(self.targetActor.weights))]
            self.update_target_critic = [tf.assign(self.targetCritic.weights[i], (tf.multiply(self.critic.weights[i], self.tau) + tf.multiply(self.targetCritic.weights[i], 1. - self.tau))) for i in range(len(self.targetCritic.weights))]

        # create data logs holder variables
        self.rewardPerEpisode = tf.placeholder(dtype=tf.float32, shape=(), name='rewardPerEpisodePlaceHolder')
        self.rpe_summary = tf.summary.scalar('rewardPerEpisode', self.rewardPerEpisode)

        # create a fileWriter (only after construction phase is over)
        self.fileWriter = tf.summary.FileWriter(gf.logdir, tf.get_default_graph())

    # predict an action with GLIE in-effect
    def act(self, givenState, train):

        # predict an action
        predictedAction = self.actor.act(givenState)

        # convert predicted action to env format
        predictedAction = predictedAction.reshape(self.env.action_space.shape)

        if train:
            # adding GLIE gaussian noise with mean 0 and decaying variance
            for i in range(len(predictedAction)):
                noise = random.gauss(0, gf.variance)
                predictedAction[i] += noise

            # reduce the variance as per GLIE
            gf.variance *= gf.varianceDecay

        return predictedAction

    #  update weights of the network from actual to target
    def updateActorWeights(self, tauValue):
        # copy weights of actor to target actor
        self.sess.run(self.update_target_actor, feed_dict={self.tau: tauValue})

    def updateCriticWeights(self, tauValue):
        # copy weights of critic to target critic
        self.sess.run(self.update_target_critic, feed_dict={self.tau: tauValue})

    # perform training and weight transfer
    def train(self, steps):
        self.critic.train(self.targetActor, self.targetCritic, self.experience)
        self.updateCriticWeights(gf.tau)
        if (steps % (gf.trainsPerEpisodeFraction*gf.episodeLength)) == 0:
            self.actor.train(self.experience)
            self.updateActorWeights(gf.tau)

    # logs data to logs directory
    def logEpisodicData(self, step, value):
        rpe_summary_str = self.sess.run(self.rpe_summary, feed_dict={self.rewardPerEpisode: value})
        self.fileWriter.add_summary(rpe_summary_str, step)

    # adds single interaction into experience list
    def remember(self, curState, action, reward, newState, doneStatus):
        self.experience.append([curState, action, reward, newState, doneStatus])

    # evaluate the agent over an episode
    def evaluate(self, step):
        rewardAcc = 0
        done = False
        curState = self.env.reset()
        for k in range(gf.episodeLength):
            if done:
                break

            action = self.act(curState, False)
            newState, reward, done, _ = self.env.step(action)
            rewardAcc += reward
            curState = newState

            if step % gf.renderEvaluationAfter == 0:
                self.env.render()
                sleep(0.1)

        self.logEpisodicData(step, rewardAcc)