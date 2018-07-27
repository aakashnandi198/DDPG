import numpy as np
from datetime import datetime

#---------------------------------------------#
#         Global Parameters                   #
#---------------------------------------------#

# general global variables
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# general global variables
# -1 restoration off
#  0 restore latest model
#num restore snapshot-num from snapshot database
restore = -1

train = True
archive = "./tmp/latest_model.ckpt"
snapshotDB = "./history/snapshot-"
saveAfter = 50
snapshotAfter = 200
displayEnv = False
miniBatchSize = 64
warmUpSteps = 1000

# model learning parameters
actorLearningRate = 0.0001
criticLearningRate = 0.001
variance = 0.4
varianceDecay = 1
gamma = 0.99
tau = 0.001
numEpisodes = 100000
episodeLength = 600
experienceLength = 300000
trainsPerEpisodeFraction = 0.1
renderEvaluationAfter = 100

# neural net configuration
actorConfig = [400, 300]
criticConfig = [400, 300]

#---------------------------------------------#
#                Misc functions               #
#---------------------------------------------#

# function to compute one step TD estimate
def computeTDEstimate(actor, critic, miniExperience):

    # pull out rewards and final states
    rewards = np.array([np.array(t[2]) for t in miniExperience])
    destStates = np.array([np.array(t[3]) for t in miniExperience])
    doneStatus = np.array([np.array(t[4]) for t in miniExperience])

    # reshape the rewards and finalStates
    rewards = rewards.reshape((-1, 1))
    destStates = destStates.reshape((-1, actor.env.observation_space.shape[0]))
    doneStatus = doneStatus.reshape((-1, 1))

    # compute actions to be taken from finalStates
    destActions = actor.act(destStates)

    # compute Q(destState,destAction) values
    QEstimates = critic.predict(destStates, destActions)

    # logical not of doneStatus
    notDone = np.logical_not(doneStatus)

    # compute td values
    TDestimates = rewards + gamma*np.multiply(QEstimates, notDone.astype(float))

    return TDestimates
