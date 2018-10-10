# Simple implementation of Reinforcement Learning (A3C) using Pytorch

This is a toy example of using multiprocessing in Python to asynchronously train a
neural network to play discrete action [CartPole](https://gym.openai.com/envs/CartPole-v0/) and
continuous action [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) games.
The asynchronous algorithm called [Asynchronous Advantage Actor-Critic](https://arxiv.org/pdf/1602.01783.pdf) or A3C.


## What are the main focuses in this implementation?

* Pytorch + multiprocessing (NOT threading) for parallel training
* Both discrete and continuous action environments
* To be simple and easy to dig into the code (less than 200 lines)

## Codes & Results

* [shared_adam.py](/shared_adam.py): optimizer that shares its parameters in parallel
* [utils.py](/utils.py): useful function that can be used more than once
* [discrete_A3C.py](/discrete_A3C.py): CartPole, neural net and training for discrete action space
* [continuous_A3C.py](/continuous_A3C.py): Pendulum, neural net and training for continuous action space

CartPole result
![cartpole](/results/cartpole.png)

Pendulum result
![pendulum](/results/pendulum.png)

## Dependencies

* pytorch >= 0.4.0
* numpy
* gym


## Credits

Forked from https://github.com/MorvanZhou/pytorch-A3C and adopted for the workshop.