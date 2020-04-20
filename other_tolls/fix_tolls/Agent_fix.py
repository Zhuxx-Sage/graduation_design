#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from pylab import *
import math

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter



class AGENT_FIX_TOLLS(object):
    # 初始化
    def __init__(self,
                 paras={}):
        self.state_dim = paras['num_state']
        self.action_dim = paras['num_action']
        self.upper_bound_action = paras['upper_bound_action']



    # 选择行为
    def get_actions(self, state):
        pass

    # 获得当前平均 traffic demand
