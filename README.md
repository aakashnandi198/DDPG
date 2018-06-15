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

1. ["Hands-on Machine Learning with Scikit-Learn and Tensorflow"](https://github.com/ageron/handson-ml) by Aurélien Géron - Tensorflow Reference .
2. [slowbull](https://github.com/slowbull/DDPG) - Tensorflow code to perform weight update from actual to target. 

## Code 
This piece of code runs on the pendulum-v0 environment provided in the 'openAI' gym and can be modified easily to run on any other environment.

- ActorCritic class which performs the following functions:
	- Initialise
	- Act
	- Remember (add to experience queue)
	- Train (train its internal actor and internal critic)
	- Evaluate (evaluate the internal actor)

- Actor class used in the actorCritic class, with the following functions:
	- Initialise
	- Act
	- Train (using samples from experience queue)

- Critic class used in actorCritic class, with the following functions:
	- Initialise
	- Predict
	- Train (using samples from experience queue)

- Global Functions
	- Consist of all the global variables that govern the algorithm.
	- Also defines a function that computes the one step look ahead estimate using target critic and target actor

- main.py
        - Spawns an actorCritic object and performs the necessary interactions with the environment over several episodes


