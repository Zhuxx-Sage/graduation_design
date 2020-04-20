#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from other_tolls.No_tolls.env_no_tolls import TrafficEnvironment_NO_TOOLS
from other_tolls.No_tolls.utils import *
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from collections import deque

# #实例化app对象
# app = Flask(__name__)
# CORS(app, resources=r'/*')

writer = SummaryWriter('log_no_tolls')


# road_network = \
#     [[0, Edge(0,0,1), Edge(1,0,2), 0, Edge(2,0,4)],
#     [Edge(3,1,0), 0, Edge(4,1,2), Edge(5,1,3), 0],
#     [Edge(6,2,0), Edge(7,2,1), 0, Edge(8,2,3), 0],
#     [0, Edge(9,3,1), Edge(10,3,2), 0, Edge(11,3,4)],
#     [Edge(12,4,0), 0, 0, Edge(13,4,3), 0]]


road_network = \
    [[0, Edge(0,0,1), Edge(1,0,2), Edge(2,0,3)],
    [Edge(3,1,0), 0, 0, Edge(4,1,3)],
    [Edge(5,2,0), 0, 0, Edge(6,2,3)],
    [Edge(7,3,0), Edge(8,3,1), Edge(9,3,2), 0]]



# ----------    Different Agents Training    ---------- #



def train_NO_TOLLS():
    # 环境
    gra = Graph(road_network)
    env = TrafficEnvironment_NO_TOOLS(gra)
    env.reset()
    # lower_bound_action, upper_bound_action = env.low_bound_action, env.upper_bound_action

    agent_config = {}
    agent_config['num_state'] = env.edges_num * env.zones_num  # 40个输入
    # agent_config['num_action'] = len(env.action_vector)
    # agent_config['lower_bound_action'] = lower_bound_action
    # agent_config['upper_bound_action'] = upper_bound_action
    # agent_config['alpha_policy'] = 1e-10

    # agent = REINFORCE_NN(agent_config)

    # 加载模型
    # agent.approximator.load_state_dict(T.load('REINFORCE_NN_parameters1.pkl'))
    Iter = 1000


    print("----------  start training!  ----------\n")
    for epoch in range(Iter):
        # print("----------  epoch: " +  epoch.__str__() + " start  ----------")
        done = False
        s = env.reset()
        G = 0
        # memory = deque()

        while not done:
            # a = agent.get_actions(s.reshape(1, -1))  # 转换成一行
            next_s, r, done, info = env.step()
            if done:
                mask = 0
            else:
                mask = 1
            # memory.append([s.reshape(1, -1), a, r, mask])
            s = next_s
            G = G + r
            # print(env.state_matrix)

        # agent.train_model(memory)
        if (epoch + 1) % 10 == 0:
            print("----------  epoch: " + (epoch + 1).__str__() + "  ----------")
            # writer.add_scalar('Train/loss', agent.loss, epoch)
            print(G)
            writer.add_scalar('Train/G', G, epoch)
            # writer.add_scalar('Train/mean', np.mean(G_log[:]), epoch)
            # G_log.append(G)
            # G_mean.append(np.mean(G_log[:]))
            # print(G_log)
            # print("G_log.mean(): " + int(np.mean(G_log[:])).__str__())
            # print('Epoch {} / {}, loss:{:.4f}'.format(epoch+1, Iter, agent.loss.item()))
            # plt.plot(G_mean)
            # plt.plot(pd.DataFrame(G_log).rolling(200).mean())
            # plt.show()

    print("----------  end training!  ----------\n")

    # 保存模型
    # T.save(agent.approximator.state_dict(), 'REINFORCE_NN_parameters_4_8_10roads.pkl')



train_NO_TOLLS()




























