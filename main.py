import gym
import tensorflow as tf
from tqdm import *
from time import sleep

import actorCritic as agent
import globalFunctions as gf


def main():
    # generate environment
    env = gym.make('Pendulum-v0')

    # spawn a tensorflow session and a saver
    sess = tf.Session()

    # generate actor-critic model
    ac = agent.ActorCritic(env, sess)

    # initializer node
    init = tf.global_variables_initializer()

    # saver created after construction phase
    saver = tf.train.Saver()

    # restore model from archive
    if gf.restore:
        saver.restore(sess, gf.archive)
    else:
        sess.run(init)

        # make sure that actor and critic have the same weights
        ac.updateActorWeights(1)
        ac.updateCriticWeights(1)

        # perform warmup by sampling random actions
        curState=env.reset()
        done = False

        for i in range(gf.warmUpSteps):
            if done or (i % gf.episodeLength == 0):
                curState = env.reset()

            action = env.action_space.sample()
            newState, reward, done, _ = env.step(action)
            ac.remember(curState, action, reward, newState, done)
            curState = newState

    # step counter
    steps = 0

    # begin learning
    for i in tqdm(range(gf.numEpisodes), desc='Episodes'):

        # init all the flags
        done = 'False'
        curState = env.reset()

        for j in tqdm(range(gf.episodeLength), desc='Action Steps'):

            if done == 'True':
                break

            # predict an action to be taken
            action = ac.act(curState,True)

            # take the predicted action
            newState, reward, done, _ = env.step(action)

            if gf.train:
                # memorize as experience
                ac.remember(curState, action, reward, newState, done)

                # perform training
                ac.train(steps)

            # render the environment
            if gf.displayEnv:
                env.render()
                sleep(0.1)

            # update curState to new State
            curState = newState
            steps += 1

        # evaluate the reward per episode
        if gf.train:
            ac.evaluate(i)
            # save the model
            if i % gf.saveAfter == 0:
                saver.save(sess, gf.archive)

    ac.fileWriter.close()
    sess.close()


if __name__ == "__main__":
    main()
