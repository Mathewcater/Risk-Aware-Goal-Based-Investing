"""
Models

Policy: fully-connected feedforward ANN

"""
# imports

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import silu
from torch.nn import Softmax
from envs import *
import pdb

# Build a fully-connected neural net for the policy

class PolicyANN(nn.Module):
    
    # constructor
    def __init__(self, env, algo_params: dict, step_size=50, gamma=0.95):
        super(PolicyANN, self).__init__()
        
        self.input_size = env.params["num_assets"] + 1 # number of inputs
        self.hidden_size = algo_params["hidden_size"] # number of hidden nodes
        self.output_size = env.params["num_assets"] # number of outputs
        self.n_layers = algo_params["num_layers"] # number of layers
        self.env = env # environment (for normalisation purposes)

        # build all layers
        self.layer_in = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.n_layers-1)])
        self.layer_out = nn.Linear(self.hidden_size, self.output_size)

        # initializers for weights and biases
        nn.init.normal_(self.layer_in.weight, mean=0, std=1/np.sqrt(self.input_size)/2)
        nn.init.constant_(self.layer_in.bias, 0)
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(self.input_size)/2)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.layer_out.weight, mean=0, std=1/np.sqrt(self.input_size)/2)
        nn.init.constant_(self.layer_out.bias, 0)
        
        # batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for _ in range(self.n_layers-1)])
        self.instance_norm = nn.InstanceNorm1d(self.hidden_size) # for batch size of 1

        # optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=algo_params["learn_rate"]) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    # forward propagation
    def forward(self, x):
        
        y = x.clone()

        # normalize features with environment parameters
        y[0] = y[0] / self.env.params["Ndt"] # time
        y[...,1:(1+len(self.env.params['S0']))] = (y[...,1:(1+len(self.env.params['S0']))] / T.tensor(self.env.params['S0'])) - 1.0
            
        # output of input layer 
        action = silu(self.layer_in(x))
    
        for layer in self.hidden_layers:
            action = silu(layer(action))
        
        action = self.layer_out(action)
        
        return Softmax(dim=0)(action)