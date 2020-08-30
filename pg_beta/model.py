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
from sympy import *
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


class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_lim):  # 输出动作
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.fc5 = nn.Linear(64, 1)
        # self.linear1 = nn.Linear(self.state_size, 1200)
        # self.linear2 = nn.Linear(1200, 360)
        # self.linear3 = nn.Linear(360, self.action_size)

    def forward(self, state):
        state = state.view(1, 1440)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))
        action_x = np.clip(action.detach().numpy(), 0, 1)
        action_x = torch.from_numpy(action_x)
        # copy_action_x = action_x

        parameters = actor.state_dict()
        e_para = self.fc3(parameters['fc3.weight'])
        l_para = self.fc3(parameters['fc3.weight'])
        e_para = self.fc4(e_para).view(10, 64)
        e_para = self.fc5(e_para).view(1, 10)
        l_para = self.fc4(l_para).view(10, 64)
        l_para = self.fc5(l_para).view(1, 10)
        lambda_e = l_para * action_x
        epislon_e = e_para * action_x

        # 计算积分
        nparray_x = action_x.detach().numpy()
        nparray_lamda = lambda_e.detach().numpy()
        nparray_epislon = epislon_e.detach().numpy()
        policy = torch.zeros(1, 10)
        for index in range(len(nparray_x[0])):
            if nparray_x[0][index] != 0:
                x_e = action_x[0][index]
                if x_e != 0:
                    integration, err = integrate.quad(compute_integration, 0, 1,
                                                      args=(nparray_lamda[0][index], nparray_epislon[0][index]),
                                                      points=[0])
                    if integration != 0:
                        policy[0][index] = np.power(x_e.detach().numpy(),
                                                    nparray_lamda[0][index] - 1) * np.power(
                            1 - x_e.detach().numpy(), nparray_epislon[0][index] - 1) / integration
        self.actionXX = action
        self.lambda_e = lambda_e
        self.epislon_e = epislon_e
        action = policy * self.action_lim
        # action = action * self.action_lim
        action = np.clip(action.detach().numpy(), 0, max_action)

        return action


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fcs1 = nn.Linear(self.state_size, 256)
        self.fcs2 = nn.Linear(256, 128)

        self.fca1 = nn.Linear(self.action_size, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        state = state.view(1, 1440)
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        x = self.fc3(s2)

        return x


def double_gamma_integration(x, t):
    return (torch.exp(-x) / x) - (torch.exp(-t * x) / (1 - torch.exp(-x)))


def compute_integration(x, lamda_e, epislon_e):
    return np.power(x, lamda_e - 1) * np.power(1 - x, epislon_e - 1)


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)
    for iter in range(n_iters):
        state = env.reset()
        # log_probs = []
        values = []
        rewards = []
        masks = []
        # entropy = 0

        for i in count():
            # state = Variable(torch.from_numpy(state))
            # state = torch.IntTensor(state).to(device)
            state = torch.FloatTensor(state).to(device)
            dist = actor.forward(state)
            noise = OrnsteinUhlenbeckActionNoise(10)
            new_action = dist + (noise.sample() * max_action)
            new_action = np.clip(new_action, 0, max_action)
            value = critic(state)
            next_state, reward, done, _ = env.step(new_action)
            # log_prob = dist.log_prob(new_action).unsqueeze(0)
            # entropy += dist.entropy().mean()

            # log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            state = next_state

            if done:
                print('Iteration:{}, Score:{}'.format(iter, i))
                break

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        # log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values
        # 要修改！
        diagamma_lambda = integrate.quad(double_gamma_integration, 0, float('inf'), args=actor.lambda_e)
        diagamma_epislon = integrate.quad(double_gamma_integration, 0, float('inf'), args=(actor.lambda_e+actor.epislon_e))
        actor_loss = advantage.detach().numpy() * (torch.log(actor.actionXX) - diagamma_lambda + diagamma_epislon) * 2
        # actor_losss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

    torch.save(actor, 'actor.pkl')
    torch.save(critic, 'critic.pkl')
    env.close()


if __name__ == '__main__':
    if os.path.exists('actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size, max_action).to(device)
    if os.path.exists('critic.pkl'):
        critic = torch.load('critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=20)
