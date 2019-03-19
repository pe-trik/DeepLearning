#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import gym
import numpy as np
import tensorflow as tf

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", default="gym_cartpole_model.h5", nargs="?", type=str, help="Name of model.")
parser.add_argument("--episodes", default=100, type=int, help="Number of episodes.")
parser.add_argument("--render", default=False, action="store_true", help="Render the environment.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Use given number of threads
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Load the model
model = tf.keras.models.load_model(args.model, compile=False)

# Choose output_mode
output_mode = None
if len(model.output_shape) == 2 and model.output_shape[-1] == 1:
    output_mode = "sigmoid"
if len(model.output_shape) == 2 and model.output_shape[-1] == 2:
    output_mode = "softmax"
if output_mode is None:
    raise ValueError("Unknown model output_shape, only 1 or 2 outputs are supported")

# Create the environment
env = gym.make("CartPole-v1")
env.seed(42)

# Evaluate the episodes
total_score = 0
for episode in range(args.episodes):
    observation = env.reset()
    score = 0
    for i in range(env.spec.timestep_limit):
        if args.render:
            env.render()
        if output_mode == "sigmoid":
            action = model.predict(observation[np.newaxis, ...])[0, 0] >= 0.5
        else:
            action = np.argmax(model.predict(observation[np.newaxis, ...])[0])

        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break

    total_score += score
    print("The episode {} finished with score {}.".format(episode + 1, score))

print("The average reward per episode was {:.2f}.".format(total_score / args.episodes))
