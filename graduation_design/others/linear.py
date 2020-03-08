#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# -----------------    REINFORCE with nonlinear approximators    -----------------#

























# -----------------    REINFORCE with linear approximators    -----------------#
class REINFORCE(object):
    def __init__(self, paras={}):
        self.state_dim = paras['state_dim']
        self.action_dim = paras['action_dim']
        self.lower_bound_action = paras['lower_bound_action']
        self.upper_bound_action = paras['upper_bound_action']

        self.alpha_mu = paras['alpha_mu']
        self.paras_mu = np.zeros((self.state_dim, self.action_dim))
        self.alpha_sigma = paras['alpha_sigma']
        self.paras_sigma = np.zeros((self.state_dim, self.action_dim))
        self.gamma = 0.999

    def pi(self, s, a):
        mu = self.get_mu(s)
        sigma = self.get_sigma(s)
        pi = np.exp(0.5 * ((mu - a) / (sigma)) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return pi

    def get_mu(self, s):
        return s.dot(self.paras_mu)

    def get_sigma(self, s):
        return np.exp(s.dot(self.paras_sigma))

    def get_actions(self, s):
        mu = self.get_mu(s)
        sigma = self.get_sigma(s)
        a = np.random.normal(mu, sigma)
        a = np.clip(a, self.lower_bound_action, self.upper_bound_action)
        return a

    def update_paras(self, states, actions, rewards):
        T = len(states)
        rewards = np.array(rewards)
        rewards = (rewards - rewards.min() + 1e-10) / (rewards.max() - rewards.min() + 1e-10)
        for t in range(T):
            G = sum([rewards[k] * (self.gamma ** (k - t - 1)) for k in range(t + 1, T)])
            a = actions[t]
            s = states[t]
            prob = self.pi(s, a)
            mu = self.get_mu(s)
            sigma = self.get_sigma(s)

            d_mu = s.reshape(-1, 1).dot(((1 / sigma ** 2) * (a - mu)).reshape(1, -1))
            d_sigma = s.reshape(-1, 1).dot(((((a - mu) / sigma) ** 2) - 1).reshape(1, -1))

            self.paras_mu += self.alpha_mu * (self.gamma ** t) * G * d_mu
            self.paras_sigma += self.alpha_sigma * (self.gamma ** t) * G * d_sigma


# -----------------    REINFORCE with linear approximators    -----------------#


def train_REINFORCE():
    # 环境初始化
    gra = Graph(road_network)
    env = Traffic_Environment_No_tolls(gra)
    env.reset()
    lower_bound_action, upper_bound_action = env.low_bound_action, env.upper_bound_action
    action_dim = env.action_vector

    agent_config = {}
    agent_config['state_dim'] = len(env.state_matrix) * len(env.state_matrix[0])
    agent_config['action_dim'] = len(env.action_vector)
    agent_config['lower_bound_action'] = lower_bound_action
    agent_config['upper_bound_action'] = upper_bound_action
    agent_config['alpha_policy'] = 1e-10
    agent_config['structure_policy'] = [36, 18]
    agent_config['alpha_mu'] = 1e-5
    agent_config['alpha_sigma'] = 1e-5

    agent = REINFORCE(agent_config)
    Iter = 2000
    G_log = []
    print("----------  start training!  ----------\n")
    for epoch in range(Iter):
        done = False
        s = env.reset()
        G = 0
        states = []
        actions = []
        rewards = []
        while not done:
            a = agent.get_actions(s)
            next_s, r, done, info = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = next_s
            G += r
        G_log.append(G)
        agent.update_paras(states, actions, rewards)
        if (epoch + 1) % 50 == 0:
            print("----------  epoch: " + (epoch + 1).__str__() + "  ----------")
            print(epoch + 1, np.mean(G_log[-50:]))

    print("----------  end training!  ----------\n")
    X = np.arange(0, Iter, 50)
    plt.plot(X, G_mean, color="blue", linewidth=1.0, linestyle='-', label="hh")
    plt.show()






# ----------    Different Agents Training    ---------- #


# ----------    Different tolls schemes Training    ---------- #

# no tolls
def train_no_tolls():
    # 环境初始化
    gra = Graph(road_network)
    env = Traffic_Environment_No_tolls(gra)
    env.reset()
    lower_bound_action, upper_bound_action = env.low_bound_action, env.upper_bound_action
    action_dim = env.action_vector

    agent_config = {}
    agent_config['state_dim'] = len(env.state_matrix) * len(env.state_matrix[0])
    agent_config['action_dim'] = len(env.action_vector)
    agent_config['lower_bound_action'] = lower_bound_action
    agent_config['upper_bound_action'] = upper_bound_action
    agent_config['alpha_policy'] = 1e-10
    agent_config['structure_policy'] = [36, 18]
    agent_config['alpha_mu'] = 1e-5
    agent_config['alpha_sigma'] = 1e-5
    agent_config['alpha_baseline'] = 1e-3
    agent_config['structure_baseline'] = [36, 18]

    agent = REINFORCE_NN(agent_config)
    Iter = 2000
    G_log = []
    G_mean = []

    # 保存模型
    # T.save(agent.approximator.state_dict(), "REINFORCE_NN_parameters.pth")

    print("----------  start training!  ----------\n")
    for epoch in range(Iter):
        # print("----------  epoch: " +  epoch.__str__() + "  ----------")
        done = False
        s = env.reset()
        G = 0
        states = []
        actions = []
        rewards = []

        while not done:
            a = agent.get_actions(s.reshape(1, -1))  # 转换成一行
            next_s, r, done, info = env.step(a)
            # print(r)
            states.append(s.reshape(1, -1))
            actions.append(a)
            rewards.append(r)
            s = next_s
            G = G + r
        # print(G)
        # G_log.append(G)
        agent.update_paras(states, actions, rewards)
        if (epoch + 1) % 20 == 0:
            print("----------  epoch: " + (epoch + 1).__str__() + "  ----------")
            G_log.append(G)
            # agent.losses.append(agent.approximator_loss[0][0])
            G_mean.append(np.mean(G_log[:]))
            print("G_log.mean(): " + int(np.mean(G_log[-20:])).__str__())
            # plt.plot(G_mean)
            # plt.plot(pd.DataFrame(G_log).rolling(200).mean())
            # plt.show()

    print("----------  end training!  ----------\n")
    # print(T.load('REINFORCE_NN_parameters.pth'))

    # 绘图
    X = np.arange(0, Iter, 20)
    plt.plot(X, G_mean, color="blue", linewidth=1.0, linestyle='-', label="hh")
    plt.show()


# ----------    Different tolls schemes Training    ---------- #

train_REINFORCE_NN()