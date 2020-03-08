#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from pylab import *

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -----------------    REINFORCE with nonlinear approximators    -----------------#



class approximator(nn.Module):
    def __init__(self, paras):
        super(approximator, self).__init__()
        # 定义层
        self.input_dim = paras['state_dim']
        self.output_dim = paras['action_dim']
        self.alpha_policy = paras['alpha_policy']
        self.structure_policy = paras['structure_policy']

        self.l1 = nn.Linear(self.input_dim, 36)
        self.l2 = nn.Linear(36, 18)
        self.l3 = nn.Linear(18, self.output_dim)
        self.l4 = nn.Linear(18, self.output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha_policy)

    def forward(self, x):
        x = T.tensor(x, dtype=T.float)
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        mu = self.l3(x)
        sigma = T.exp(self.l4(x))
        return mu, sigma


# 思维决策
class REINFORCE_NN(object):
    # 初始化
    def __init__(self,
                 paras={}):
        super(REINFORCE_NN, self).__init__()
        self.state_dim = paras['state_dim']
        self.action_dim = paras['action_dim']
        self.lower_bound_action = paras['lower_bound_action']
        self.upper_bound_action = paras['upper_bound_action']

        self.gamma = 1
        self.approximator = approximator(paras)
        self.losses = []

    # 选择行为
    def get_actions(self, state):
        mu, sigma = self.approximator.forward(state)
        mu = mu.detach().numpy()  # 把数值拿出来
        sigma = sigma.detach().numpy()
        a = np.random.normal(mu, sigma)
        a = np.clip(a, self.lower_bound_action, self.upper_bound_action)  # 小于lower的变成lower，大于upper的变成upper
        return a

    # 学习更新参数
    def update_paras(self, states, actions, rewards):
        G = np.array([sum([rewards[k] * (self.gamma ** (k - t - 1)) for k in range(t + 1, len(states))]) for t in
                      range(len(states))])

        s = np.array([list(states[idx][0]) for idx in range(len(states))])


        a = np.array([list(actions[idx][0]) for idx in range(len(actions))])


        gammas = np.array([self.gamma ** t for t in range(len(states))])


        G = G * gammas
        G = (G - G.min() + 1e-10) / (G.max() - G.min() + 1e-10)

        mu, sigma = self.approximator.forward(s)
        action_probs = T.distributions.Normal(mu, sigma)
        a = T.tensor(a, dtype=T.float)
        G = T.tensor(G, dtype=T.float).view(len(a), 1)
        log_probs = action_probs.log_prob(a)
        self.approximator_loss = -log_probs * G
        # print(self.approximator_loss)
        self.approximator.optimizer.zero_grad()
        self.approximator_loss.sum().backward()
        self.approximator.optimizer.step()
        #print(self.approximator_loss[0][0])









