#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from flask import Flask, jsonify, render_template, request
import json
from flask_cors import CORS

from other_tolls.No_tolls.env_no_tolls import *
from other_tolls.No_tolls.utils import Graph, Edge
# from version_1.Agents import *
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from collections import deque

# 路网
road_network = \
    [[0, Edge(0, 0, 1), Edge(1, 0, 2), Edge(2, 0, 3)],
     [Edge(3, 1, 0), 0, 0, Edge(4, 1, 3)],
     [Edge(5, 2, 0), 0, 0, Edge(6, 2, 3)],
     [Edge(7, 3, 0), Edge(8, 3, 1), Edge(9, 3, 2), 0]]


# 处理state，将每个路段的车辆数累加
def process_state(state_matrix):
    new_s = []
    for i in range(len(state_matrix)):
        new_s.append(sum(state_matrix[i]))
    return new_s


# 进行训练
def launch_train():
    gra = Graph(road_network)
    env = TrafficEnvironment_NO_TOOLS(gra)
    env.reset()
    # lower_bound_action, upper_bound_action = env.low_bound_action, env.upper_bound_action

    agent_config = {}
    # agent_config['num_state'] = env.edges_num * env.zones_num  # 40个输入
    # agent_config['num_action'] = len(env.action_vector)
    # agent_config['lower_bound_action'] = lower_bound_action
    # agent_config['upper_bound_action'] = upper_bound_action
    # agent_config['alpha_policy'] = 1e-10

    # agent = REINFORCE_NN(agent_config)

    # 所有状态矩阵信息
    all_state_matrix = {}
    # 所有reward
    rewards = []
    # 加载模型
    # agent.approximator.load_state_dict(T.load('../REINFORCE_NN_parameters_4_7_10roads.pkl'))

    state = []
    s = np.array([[14,13,12,11],
         [12,14,16,13],
         [10,12,15,18],
         [12,19,12,16],
         [11,12,15,18],
         [10,11,11,12],
         [13,16,18,19],
         [12,14,15,13],
         [12,12,13,15],
         [14,12,11,16]])
    state.append(process_state(s))
    done = False
    total_rewards = 0
    print("****************************************************")

    while True:
        # a = agent.get_actions(s.reshape(1,-1))
        next_s, r, done, info = env.step()
        print(r)
        state.append(process_state(next_s))
        total_rewards += r

        if done:
            rewards.append(total_rewards)
            print("Score", total_rewards)
            break
        s = next_s
    for i in range(len(state)):
        print(state[i])
    # print('\n')
    # print(rewards)
    # for i in range(len(state)):
    #     print(state[i])
    # print('\n')


# print(all_state_matrix[episode.__str__()])
    print(sum(rewards))

launch_train()
