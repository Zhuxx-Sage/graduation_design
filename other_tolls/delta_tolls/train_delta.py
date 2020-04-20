#!/usr/bin/env python 
# -*- coding:utf-8 -*-



import numpy as np
from flask import Flask, jsonify, render_template, request
import json
from flask_cors import CORS

from other_tolls.delta_tolls.env_delta import *
from other_tolls.delta_tolls.utils import Graph, Edge
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from collections import deque

# #实例化app对象
# app = Flask(__name__)
# CORS(app, resources=r'/*')

writer = SummaryWriter('log_delta')

# road_network = \
#     [[0, Edge(0,0,1), Edge(1,0,2), 0, Edge(2,0,4)],
#     [Edge(3,1,0), 0, Edge(4,1,2), Edge(5,1,3), 0],
#     [Edge(6,2,0), Edge(7,2,1), 0, Edge(8,2,3), 0],
#     [0, Edge(9,3,1), Edge(10,3,2), 0, Edge(11,3,4)],
#     [Edge(12,4,0), 0, 0, Edge(13,4,3), 0]]


road_network = \
    [[0, Edge(0, 0, 1), Edge(1, 0, 2), Edge(2, 0, 3)],
     [Edge(3, 1, 0), 0, 0, Edge(4, 1, 3)],
     [Edge(5, 2, 0), 0, 0, Edge(6, 2, 3)],
     [Edge(7, 3, 0), Edge(8, 3, 1), Edge(9, 3, 2), 0]]


# ----------    Different Agents Training    ---------- #


def train_REINFORCE_NN():
    # 环境
    gra = Graph(road_network)
    env = TrafficEnvironment_DELTA(gra)
    env.reset()

    # 加载模型
    # agent.approximator.load_state_dict(T.load('REINFORCE_NN_parameters1.pkl'))
    Iter = 10

    print("----------  start   ----------\n")
    for epoch in range(Iter):
        # print("----------  epoch: " +  epoch.__str__() + " start  ----------")
        done = False
        s = env.reset()
        G = 0

        while not done:
            a = env.get_now_action()
            print(a)
            next_s, r, done, info = env.step(a)
            s = next_s
            G = G + r

    print(G)

    print("----------  end   ----------\n")



train_REINFORCE_NN()
