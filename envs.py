"""
Market models 
"""
# imports
import numpy as np
import torch as T
from hyperparams import *
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
import pdb

# Black-Scholes Environment

class BS_Environment:
    # constructor
    def __init__(self, params: dict):
        # parameters and spaces
        self.params = params
        self.cholesky = T.linalg.cholesky(self.params["corr_matrix"])
        self.dt = self.params["T"]/self.params["Ndt"]
        
    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        
        t0 = T.zeros(Nsims,1)
        s0 = T.ones(Nsims,self.params["num_assets"]) # all initial asset prices are 1.0.
        x0 = self.params["init_wealth"] * T.ones(Nsims,1)
        
        return T.cat((t0, s0, x0), axis=1)

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
        time_t = curr_state[:,0].unsqueeze(dim=-1)
        
        # asset prices, risk-free and risky resp.
        risk_free_price_t = curr_state[:,1].unsqueeze(dim=-1)
        risky_prices_t = curr_state[:,2:-1]
        
        # current wealth
        x_t = curr_state[:,-1].unsqueeze(dim=-1)
                
        # risky assets' price modification via cholesky decomposition
        corr_samps = T.matmul(T.normal(0,
                                       self.dt**(1/2), 
                                       (curr_state.shape[0], self.params["num_assets"] - 1,)),
                              self.cholesky)
        
        risky_prices_tp1 = risky_prices_t * T.exp( (self.params["drifts"].reshape(1,-1) - \
                                                    0.5*self.params["vols"].reshape(1,-1)**2)*self.dt \
                                                  + self.params["vols"].reshape(1,-1)*corr_samps)  

        # risk-free asset price modification via interest rate appreciation
        risk_free_price_tp1 = risk_free_price_t*np.exp(self.params["interest_rate"]*self.dt)
        
        S_tp1 = T.cat((risk_free_price_tp1, risky_prices_tp1), axis=1)

        # wealth modification 
        x_tp1 = x_t*action[:,0].reshape(-1,1)*np.exp(self.params["interest_rate"]*self.dt) \
                + x_t* T.sum(action[:,1:]*(risky_prices_tp1 / risky_prices_t), axis=1).reshape(-1,1)
        
        # time modification 
        tp1 = time_t + 1

        new_state = T.cat((tp1, S_tp1, x_tp1), axis=1)
        reward = x_tp1 - x_t
                       
        return new_state, reward

##########################################################################

# Factor model environment

class FactorModel_Environment:
    
    # constructor
    def __init__(self, params: dict):
        # parameters and spaces
        self.params = params
        self.mean = params['idio_means']
        self.cov_matrix = (0.02 ** 2) * T.ones((params['num_assets'] - 1, params['num_assets'] - 1)) + T.diag(params['idio_vars'])
        self.return_dist = MultivariateNormal(loc=self.mean, covariance_matrix=self.cov_matrix)
        self.dt = self.params["T"]/self.params["Ndt"]
        
    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        
        t0 = T.zeros(Nsims,1)
        s0 = T.ones(Nsims,self.params["num_assets"]) # all initial asset prices are 1.0.
        x0 = self.params["init_wealth"] * T.ones(Nsims,1)
        
        return T.cat((t0, s0, x0), axis=1)

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
        time_t = curr_state[:,0].unsqueeze(dim=-1)
        
        # asset prices, risk-free and risky resp.
        risk_free_price_t = curr_state[:,1].unsqueeze(dim=-1)
        risky_prices_t = curr_state[:,2:-1]
        
        # current wealth
        x_t = curr_state[:,-1].unsqueeze(dim=-1)
                
        # risky assets' price modification via systemic and idiosyncratic factors
        batch_size = risky_prices_t.shape[0]
        risky_prices_tp1 = risky_prices_t + risky_prices_t * (self.return_dist.sample((batch_size,)))
        
        # risk-free asset price modification via interest rate appreciation
        risk_free_price_tp1 = risk_free_price_t*np.exp(self.params["interest_rate"]*self.dt)
        
        # all asset prices
        S_tp1 = T.cat((risk_free_price_tp1, risky_prices_tp1), axis=1)

        # wealth modification 
        x_tp1 = x_t*action[:,0].reshape(-1,1)*np.exp(self.params["interest_rate"]*self.dt) \
                + x_t* T.sum(action[:,1:]*(risky_prices_tp1 / risky_prices_t), axis=1).reshape(-1,1)
        
        # time modification 
        tp1 = time_t + 1

        new_state = T.cat((tp1, S_tp1, x_tp1), axis=1)
        reward = x_tp1 - x_t
                       
        return new_state, reward