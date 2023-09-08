"""
Market models 
"""
# imports
import numpy as np
import torch as T
from hyperparams import *
from torch.distributions.uniform import Uniform
import pdb


class BS_Environment:
    # constructor
    def __init__(self, params: dict):
        # parameters and spaces
        self.params = params
        # self.spaces = {'s1_space' : np.linspace(self.S0[0] * T.exp( (self.mu[0] - 0.5 * self.sigma[0]**2)*self.T + self.sigma[0]*np.sqrt(self.T)*(-4) ),
        #                                         self.S0[0] * T.exp( (self.mu[0] - 0.5 * self.sigma[0]**2)*self.T + self.sigma[0]*np.sqrt(self.T)*(4) ), 21),
        #               's2_space' : np.linspace(self.S0[1] * T.exp( (self.mu[1] - 0.5 * self.sigma[1]**2)*self.T + self.sigma[1]*np.sqrt(self.T)*(-3) ),
        #                                         self.S0[1] * T.exp( (self.mu[1] - 0.5 * self.sigma[1]**2)*self.T + self.sigma[1]*np.sqrt(self.T)*(3) ), 21),
        #               's3_space' : np.linspace(self.S0[2] * T.exp( (self.mu[2] - 0.5 * self.sigma[2]**2)*self.T + self.sigma[2]*np.sqrt(self.T)*(-3) ),
        #                                         self.S0[2] * T.exp( (self.mu[2] - 0.5 * self.sigma[2]**2)*self.T + self.sigma[2]*np.sqrt(self.T)*(3) ), 21),
        #               'action_space' : np.linspace(0.0, 1.0, 21)}
        self.cholesky = T.linalg.cholesky(self.params["cov_matrix"])
        self.dt = self.params["T"]/self.params["Ndt"]
        
    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        
        t0 = T.zeros(Nsims)
        s0 = T.ones(self.params["num_assets"]) # all initial asset prices are 1.0.
        x0 = T.tensor([self.params["init_wealth"]])
        # qn1 = (self.params["init_wealth"]/self.params["num_assets"])*T.ones(self.params["num_assets"])
        
        return T.cat((t0, s0, x0))

    # initialization of the environment with its random initial state
    def random_reset(self, Nsims=1):
        
        t0 = T.zeros(Nsims)
        s0 = T.clamp(T.normal(1, 0.8, size=(self.params["num_assets"],)), min=0.0, max=2.0)
        x0 = T.tensor([self.params["init_wealth"]])

        return T.cat((t0, s0, x0))
    
    # simulation engine
    def step(self, curr_state, action):
        
        # decompose state into time, asset prices and wealth.
        
        # time
        time_t = curr_state[0].unsqueeze(dim=-1)
        
        # asset prices, risk-free and risky resp.
        risk_free_price_t = curr_state[1].unsqueeze(dim=-1)
        risky_prices_t = curr_state[2:-1]
        
        # current wealth
        x_t = curr_state[-1].unsqueeze(dim=-1)
                
        # risky assets' price modification via cholesky decomposition
        corr_samps = T.matmul(self.cholesky, T.normal(0, self.dt**(1/2), (self.params["num_assets"] - 1, )))
        risky_prices_tp1 = risky_prices_t*T.exp( (T.tensor(self.params["drifts"]) - \
                             (T.tensor(self.params["vols"])**2)/2)*self.dt + T.tensor(self.params["vols"])*corr_samps)  
        
        # risk-free asset price modification via interest rate appreciation
        risk_free_price_tp1 = risk_free_price_t*T.exp(T.tensor(self.params["interest_rate"]*self.dt))
        
        S_tp1 = T.cat((risk_free_price_tp1, risky_prices_tp1))

        # wealth modification 
        x_tp1 = x_t*action[0]*T.exp(T.tensor(self.params["interest_rate"]*self.dt)) + x_t*T.sum(action[1:]*(risky_prices_tp1 / risky_prices_t))
        
        # time modification 
        tp1 = time_t + 1

        new_state = T.cat((tp1, S_tp1, x_tp1))
        reward = x_tp1 - x_t
                       
        return new_state, reward