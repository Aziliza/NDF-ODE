from scipy.integrate import odeint
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def REIS(t,u0,a,noise=0):
    def f(t,u):
        s,e,i,r = u
        b,g,si,m,n=a
        dsdt = m*(1-s)-b*s*i-n*s
        dedt=b*s*i-(m+si)*e
        didt=si*e-(g+m)*i
        drdt=g*i-m*r+n*s
        return [dsdt,dedt,didt,drdt]
    data=odeint(f,u0,t,tfirst=True)
    data+=np.random.normal(0,noise,data.shape)*data
    return data


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class block_x(nn.Module):
    def __init__(self, x_length,a_length):
        super().__init__()
        self.l1 = nn.Linear(x_length+a_length,64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,32)
        # self.dropout=nn.Dropout(0.5)
        self.l4 = nn.Linear(32,x_length)
        self.x_length=x_length
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))
        # out = self.dropout(out)
        return out+x[:,:self.x_length]


class block_a(nn.Module):
    def __init__(self, x_length,a_length):
        super().__init__()
        self.l1 = nn.Linear(x_length,32)
        self.l2 = nn.Linear(32,32)
        # self.dropout=nn.Dropout(0.5)
        self.l3 = nn.Linear(32,a_length)
    def forward(self, x):
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        # out = self.dropout(out)
        return a


