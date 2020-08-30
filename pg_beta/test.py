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

x = 2
np.power(x, 3)
print(x)