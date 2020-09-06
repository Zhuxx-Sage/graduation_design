import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from itertools import count
import os
from work.pg_beta.utils import *
from work.pg_beta.env_pg_beta import TrafficEnvironment
from scipy import integrate
from scipy.special import gamma
import math
from work.pg_beta.utils import OrnsteinUhlenbeckActionNoise

road_network = \
    [[0, Edge(0, 0, 1), Edge(1, 0, 2), Edge(2, 0, 3)],
     [Edge(3, 1, 0), 0, 0, Edge(4, 1, 3)],
     [Edge(5, 2, 0), 0, 0, Edge(6, 2, 3)],
     [Edge(7, 3, 0), Edge(8, 3, 1), Edge(9, 3, 2), 0]]

gra = Graph(road_network)
env = TrafficEnvironment(gra)
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size = 1440
action_size = len(env.action_vector)
lr = 1e-10
max_action = 6
EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v, v)
    return x.type(torch.FloatTensor)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_lim):  # 输出动作
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_size, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4 = nn.Linear(64, action_size)
        self.fc4.weight.data.uniform_(-EPS, EPS)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, state):
        state = state.view(1, 1440)
        state = state.type(torch.FloatTensor)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.relu(self.fc4(x))
        action_x = np.clip(action.detach().numpy(), 0, 1)

        # parameters = actor.state_dict()
        # e_para = self.fc3(parameters['fc3.weight'])
        # l_para = self.fc3(parameters['fc3.weight'])
        # e_para = self.fc4(e_para).view(10, 64)
        # e_para = self.fc5(e_para).view(1, 10)
        # l_para = self.fc4(l_para).view(10, 64)
        # l_para = self.fc5(l_para).view(1, 10)
        # lambda_e = l_para * action_x
        # epislon_e = e_para * action_x
        #
        # # 计算积分
        # nparray_x = action_x.detach().numpy()
        # nparray_lamda = lambda_e.detach().numpy()
        # nparray_epislon = epislon_e.detach().numpy()
        # policy = torch.zeros(1, 10)
        # for index in range(len(nparray_x[0])):
        #     if nparray_x[0][index] != 0:
        #         x_e = action_x[0][index]
        #         if x_e != 0:
        #             integration, err = integrate.quad(compute_integration, 0, 1,
        #                                               args=(nparray_lamda[0][index], nparray_epislon[0][index]),
        #                                               points=[0])
        #             if integration != 0:
        #                 policy[0][index] = np.power(x_e.detach().numpy(),
        #                                             nparray_lamda[0][index] - 1) * np.power(
        #                     1 - x_e.detach().numpy(), nparray_epislon[0][index] - 1) / integration
        # self.actionXX = action
        # self.lambda_e = lambda_e
        # self.epislon_e = epislon_e
        # action = policy * self.action_lim
        # # action = action * self.action_lim
        # action = np.clip(action.detach().numpy(), 0, max_action)

        return action, action_x


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fcs1 = nn.Linear(self.state_size, 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(self.action_size, 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        state = state.view(1, 1440)
        state = state.type(torch.FloatTensor)

        action = torch.from_numpy(action)
        action = action.type(torch.FloatTensor)
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))

        a1 = F.relu(self.fca1(action))

        x = torch.cat((s2, a1), dim=1)
        x = self.fc3(x)
        return x


def double_gamma_integration(x, t):
    return (np.exp(-x) / x) - (np.exp(-t * x) / (1 - np.exp(-x)))


def compute_integration(x, lamda_e, epislon_e):
    return np.power(x, lamda_e - 1) * np.power(1 - x, epislon_e - 1)



def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def choose_action(actor, action, action_x):
    actor.lambda_e = (actor.fc5(actor.fc4.weight.data.view(10, 64)).view(1, 10) * action).detach().numpy()
    actor.epsilon_e = (actor.fc5(actor.fc4.weight.data.view(10, 64)).view(1, 10) * action).detach().numpy()
    policy = torch.zeros(1, 10)
    for index in range(len(actor.lambda_e)):
        x_e = action_x[0][index]
        if x_e != 0:
            integration, err = integrate.quad(compute_integration, 0, 1, args=(actor.lambda_e[0][index], actor.epsilon_e[0][index]), points=[0])
            if integration != 0:
                policy[0][index] = np.power(x_e, actor.lambda_e[0][index] - 1) * np.power(
                                1 - x_e, actor.epsilon_e[0][index] - 1) / integration
    return policy

def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        # entropy = 0

        for i in count():
            state = torch.FloatTensor(state).to(device)
            action, action_x = actor.forward(state)
            new_action = choose_action(actor, action, action_x)
            new_action = new_action * max_action
            new_action = np.clip(new_action, 0, max_action).numpy()
            value = critic(state, new_action)
            next_state, reward, done, _ = env.step(new_action)
            log_prob = torch.from_numpy(np.log(new_action))
            # entropy += new_action.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration:{}, Score:{}'.format(iter, i))
                break

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state, new_action)
        returns = compute_returns(next_value, rewards, masks)

        # log_probs = torch.from_numpy(np.array(log_probs))

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        # 要修改！
        # diagamma_lambda = torch.Tensor([math.lgamma(actor.lambda_e[0][index]) for index in range(len(actor.lambda_e[0]))])
        # diagamma_epislon = torch.Tensor([math.lgamma(actor.lambda_e[0][index] + actor.epsilon_e[0][index]) for index in range(len(actor.lambda_e[0]))])
        # diagamma_lambda = torch.Tensor([ integrate.quad(double_gamma_integration, 0, float('inf'), args=np.float(actor.lambda_e[0][index].detach().numpy())) for index in range(len(actor.lambda_e[0]))])
        # diagamma_epislon = torch.Tensor([integrate.quad(double_gamma_integration, 0, float('inf'), args=np.float((actor.lambda_e[0][index].detach().numpy())) + np.float(actor.epislon_e[0][index].detach().numpy())) for index in range(len(actor.lambda_e[0]))])
        # actor_loss = advantage.detach().numpy() * (torch.log(actor.actionXX) - diagamma_lambda + diagamma_epislon) * 2
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        actor_loss.requires_grad = True

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

    torch.save(actor, 'actor.pkl')
    torch.save(critic, 'critic.pkl')


if __name__ == '__main__':
    if os.path.exists('work/pg_beta/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size, max_action).to(device)
    if os.path.exists('work/pg_beta/critic.pkl'):
        critic = torch.load('critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=1000)
