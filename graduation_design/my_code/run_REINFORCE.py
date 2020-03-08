#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import argparse, math, os
import numpy as np
import gym
from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
from .REINFORCE import REINFORCE
from .REINFORCE import NormalizedActions


args =   # 参数
agent = REINFORCE(args.hidden_size, env.observation_space.shape[0], env.action_space)

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(args.num_steps):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break

    agent.update_parameters(rewards, log_probs, entropies, args.gamma)



    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

