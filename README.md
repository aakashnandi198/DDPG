# Deep Deterministic Policy Gradient

## Associated Paper
Inspired from ["Continuous Control with deep reinforcement learning"](https://arxiv.org/abs/1509.02971) published by the following scholars:
1. Timothy P. Lillicrap and Jonathan J. Hunt
2. Alexander Pritzel
3. Nicolas Heess
4. Tom Erez
5. Yuval Tassa
6. David Silver
7. Daan Wiestra

## Acknowledgements
A great place to learn tensorflow before you begin understanding this code would be ["Hands-on Machine Learning with Scikit-Learn and Tensorflow"](https://github.com/ageron/handson-ml) by Aurélien Géron.

There were several places where I was stuck and had to refer to the following people's repository:
1. [slowbull](https://github.com/slowbull/DDPG) - tensorflow code to perform weight update from actual to target. 


## Code 
This piece of code runs on the pendulum-v0 environment provided in the 'openAI' gym and can be modified easily to run on any other environment.

1. actorCritic.py
.. ActorCritic class which performs the following functions:
...1. Initialise
...2. Act
...3. Remember (add to experience queue)
...4. Train (train its internal actor and internal critic)
...5. Evaluate (evaluate the internal actor)

2. actor.py
.. Actor class used in the actorCritic class, with the following functions:
...1. Initialise
...2. Act
...3. Train (using samples from experience queue)

3. critic.py
.. Critic class used in actorCritic class, with the following functions:
...1.Initialise
...2.Predict
...3.Train (using samples from experience queue)

4. globalFunctions.py
.. Consist of all the global variables that govern the algorithm.
.. Also defines a function that computes the one step look ahead estimate using target critic and target actor

5. main.py
.. Spawns an actorCritic object and performs the necessary interactions with the environment over several episodes


