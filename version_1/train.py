#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from version_1.env_pg import *
from version_1.utils import Graph, Edge
from version_1.Agents import *
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from collections import deque

writer = SummaryWriter('log')


road_network = \
    [[0, Edge(0,0,1), Edge(1,0,2), Edge(2,0,3)],
    [Edge(3,1,0), 0, 0, Edge(4,1,3)],
    [Edge(5,2,0), 0, 0, Edge(6,2,3)],
    [Edge(7,3,0), Edge(8,3,1), Edge(9,3,2), 0]]



# ----------    Different Agents Training    ---------- #



def train_REINFORCE_NN():
    # 环境
    gra = Graph(road_network)
    env = TrafficEnvironment(gra)
    env.reset()
    lower_bound_action, upper_bound_action = env.low_bound_action, env.upper_bound_action
    action_dim = env.action_vector

    agent_config = {}
    agent_config['num_state'] = env.edges_num * env.zones_num  # 40个输入
    agent_config['num_action'] = len(env.action_vector)
    agent_config['lower_bound_action'] = lower_bound_action
    agent_config['upper_bound_action'] = upper_bound_action
    agent_config['alpha_policy'] = 1e-10

    agent = REINFORCE_NN(agent_config)

    # 加载模型
    agent.approximator.load_state_dict(T.load('REINFORCE_NN_parameters1.pkl'))
    Iter = 30000
    G_log = []
    G_mean = []


    print("----------  start training!  ----------\n")
    for epoch in range(Iter):
        # print("----------  epoch: " +  epoch.__str__() + " start  ----------")
        done = False
        s = env.reset()
        G = 0
        memory = deque()

        while not done:
            a = agent.get_actions(s.reshape(1, -1))  # 转换成一行
            next_s, r, done, info = env.step(a)
            if done:
                mask = 0
            else:
                mask = 1
            memory.append([s.reshape(1, -1), a, r, mask])
            s = next_s
            G = G + r

        agent.train_model(memory)
        if (epoch + 1) % 100 == 0:
            print("----------  epoch: " + (epoch + 1).__str__() + "  ----------")
            writer.add_scalar('Train/loss', agent.loss, epoch)
            writer.add_scalar('Train/G', G, epoch)
            # writer.add_scalar('Train/mean', np.mean(G_log[:]), epoch)
            # G_log.append(G)
            # G_mean.append(np.mean(G_log[:]))
            # print(G_log)
            # print("G_log.mean(): " + int(np.mean(G_log[:])).__str__())
            print('Epoch {} / {}, loss:{:.4f}'.format(epoch+1, Iter, agent.loss.item()))
            # plt.plot(G_mean)
            # plt.plot(pd.DataFrame(G_log).rolling(200).mean())
            # plt.show()

    print("----------  end training!  ----------\n")
    # print(T.load('REINFORCE_NN_parameters.pth'))

    # 保存模型
    T.save(agent.approximator.state_dict(), 'REINFORCE_NN_parameters1.pkl')

    # 绘图
    # X = np.arange(0, Iter, 50)
    # plt.subplot(121)
    # plt.plot(X, G_log, color="blue", linewidth=1.0, linestyle='-', label="hh")
    # Y = np.arange(0, Iter, 50)
    # plt.subplot(122)
    # plt.plot(Y, G_mean, color="red", linewidth=1.0, linestyle='-', label="hh")
    # plt.show()

train_REINFORCE_NN()




























