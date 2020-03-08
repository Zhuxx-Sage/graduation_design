#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from graduation_design.version_1.env_pg import *
from graduation_design.others.env_no_tolls import *
from graduation_design.version_1.utils import Graph, Edge
from graduation_design.version_1.Agents import *
from matplotlib import pyplot as plt
import visdom

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
    agent_config['state_dim'] = len(env.state_matrix) * len(env.state_matrix[0])
    agent_config['action_dim'] = len(env.action_vector)
    agent_config['lower_bound_action'] = lower_bound_action
    agent_config['upper_bound_action'] = upper_bound_action
    agent_config['alpha_policy'] = 0.5e-10
    agent_config['structure_policy'] = [36, 18]

    agent = REINFORCE_NN(agent_config)
    agent.approximator.load_state_dict(T.load('REINFORCE_NN_parameters.pkl'))
    Iter = 1000
    G_log = []
    G_mean = []


    print("----------  start training!  ----------\n")
    for epoch in range(Iter):
        # print("----------  epoch: " +  epoch.__str__() + " start  ----------")
        done = False
        s = env.reset()
        G = 0
        states = []
        actions = []
        rewards = []

        while not done:
            a = agent.get_actions(s.reshape(1, -1))  # 转换成一行
            # print("a is " + a.__str__())
            next_s, r, done, info = env.step(a)
            # print("r is " + r.__str__())
            states.append(s.reshape(1, -1))
            actions.append(a)
            rewards.append(r)
            s = next_s
            G = G + r
        # print("----------  epoch: " + epoch.__str__() + " end  ----------")
        #print(rewards)
        print(actions)
        # print("G is " + G.__str__())
        # G_log.append(G)
        agent.update_paras(states, actions, rewards)
        if (epoch + 1) % 20 == 0:
            print("----------  epoch: " + (epoch + 1).__str__() + "  ----------")
            G_log.append(G)
            # agent.losses.append(agent.approximator_loss[0][0])
            G_mean.append(np.mean(G_log[:]))
            # print(G_log)
            print("G_log.mean(): " + int(np.mean(G_log[:])).__str__())
            # plt.plot(G_mean)
            # plt.plot(pd.DataFrame(G_log).rolling(200).mean())
            # plt.show()

    print("----------  end training!  ----------\n")
    # print(T.load('REINFORCE_NN_parameters.pth'))

    # 保存模型
    T.save(agent.approximator.state_dict(), 'REINFORCE_NN_parameters.pkl')

    # 绘图
    X = np.arange(0, Iter, 20)
    plt.subplot(121)
    plt.plot(X, G_log, color="blue", linewidth=1.0, linestyle='-', label="hh")
    Y = np.arange(0, Iter, 20)
    plt.subplot(122)
    plt.plot(Y, G_mean, color="red", linewidth=1.0, linestyle='-', label="hh")
    plt.show()






























