#!/usr/bin/env python 
# -*- coding:utf-8 -*-


# -----------------    REINFORCE with nolinear approximators & baseline    -----------------#
class approximator_baseline(nn.Module):
    def __init__(self, paras):
        super(approximator_baseline, self).__init__()
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
        # print(x.size())
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # for idx, layer in enumerate(self.structure[:-1]):
        #     x = F.relu(layer(x))
        mu = self.l3(x)
        # mu = self.structure[-1][0](x)
        sigma = T.exp(self.l4(x))
        # sigma = T.exp(self.structure[-1][1](x))
        return mu, sigma

class REINFORCE_baseline_NN(object):
    def __init__(self, paras={}):
        self.state_dim = paras['state_dim']
        self.action_dim = paras['action_dim']
        self.lower_bound_action = paras['lower_bound_action']
        self.upper_bound_action = paras['upper_bound_action']

        self.gamma = 1
        self.approximator = approximator_baseline(paras)
        self.baseline = self.get_baseline(paras)

    def get_baseline(self, paras):
        state = input(shape=[self.state_dim, ])
        for idx, layer in enumerate(paras['structure_baseline']):
            if idx == 0:
                dense_layers = [Dense(layer, activation='relu')(state)]
            else:
                dense_layers.append(Dense(layer, activation='relu')(dense_layers[-1]))

        value = Dense(1, activation='linear')(dense_layers[-1])
        baseline = Model(input=[state], output=[value])
        baseline.compile(optimizer=Adam(lr=paras['alpha_baseline']), loss='mse')
        return baseline

    def get_actions(self, s):
        mu, sigma = self.approximator.forward()
        mu = mu.detach().numpy()
        sigma = sigma.detach().numpy()
        a = np.random.normal(mu, sigma)
        a = np.clip(a, self.lower_bound_action, self.upper_bound_action)
        return a

    def update_paras(self, states, actions, rewards):
        G = np.array([sum([rewards[k] * (self.gamma ** (k - t - 1)) for k in range(t + 1, len(states))]) for t in
                      range(len(states))])
        s = np.array([list(states[idx][0]) for idx in range(len(states))])
        a = np.array([list(actions[idx][0]) for idx in range(len(actions))])
        gammas = np.array([self.gamma ** t for t in range(len(states))])
        v_s = self.baseline.predict(s)
        G = (G - G.min() + 1e-10) / (G.max() - G.min() + 1e-10)
        delta = G.reshape(-1, 1) - v_s
        self.baseline.fit(s, delta, verbose=False)

        delta = delta.T
        delta *= gammas
        mu, sigma = self.approximator.forward(s)
        action_prob = T.distribution.Normal(mu, sigma)
        a = T.tensor(a, dtype=T.float)
        delta = T.tensor(delta, dtype=T.float).view(len(a), 1)
        log_probs = action_prob.log_prob(a)
        self.approximator_loss = -log_probs * delta
        self.approximator.optimizer.zero_grad()
        self.approximator_loss.sum().backward()
        self.approximator.optimizer.step()

    # -----------------    REINFORCE with nolinear approximators & baseline   -----------------#




def train_REINFORCE_baseline_NN():
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

    agent = REINFORCE_baseline_NN(agent_config)
    Iter = 50000
    G_log = []
    G_mean = []

    print("----------  start training!  ----------\n")
    for epoch in range(Iter):
        done = False
        s = env.reset()
        G = 0
        states = []
        actions = []
        rewards = []
        while not done:
            a = agent.get_actions(s.reshape(1, -1))
            sp, r, done, info = env.step(a)
            states.append(s.reshape(1, -1))
            actions.append(a)
            rewards.append(r)
            s = sp
            G += r
        G_log.append(G)
        agent.update_paras(states, actions, rewards)
        if (epoch + 1) % 50 == 0:
            print("----------  epoch: " + (epoch + 1).__str__() + "  ----------")
            G_mean.append(np.mean(G_log[:]))
            print(epoch + 1, np.mean(G_log[-50:]))


    print("----------  end training!  ----------\n")
    # print(T.load('REINFORCE_NN_parameters.pth'))

    # 绘图
    X = np.arange(0, Iter, 100)
    plt.plot(X, G_mean, color="blue", linewidth=1.0, linestyle='-', label="hh")
    plt.show()