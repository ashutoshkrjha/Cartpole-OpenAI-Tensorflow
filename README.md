# Cartpole-OpenAI-Tensorflow
A Tensorflow implementation of an RL agent to balance a Cartpole from OpenAI Gym

## Implementation Details:

I have used a Policy gradient (as explained in Chapter 13 of book by Sutton and Barto) based agent to solve the MDP for the cartpole. The position of the cart is fed as an input to a neural network which then produces a probability of the action to choose(only two in this case: right/left).

The neural net is a simple 3 layer feed forward network for which the hidden layer activation is ReLU and output function is sigmoid.
The environment resets once the average reward reaches 200. It also starts to render the environment only after average reward is 100 as anything before that is not a good agent and rendering only makes training slower.

## Usage

Run:
```
$ python cartpole.py
```
## Fun-Side Note:

I came across this while reading about actor-critic methods as a part of review on Generative Adversarial Networks. Turns out Policy Gradient Methods are much better in the continuous space than Q-learning or other value function based methods.
