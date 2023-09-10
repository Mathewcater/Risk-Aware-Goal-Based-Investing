"""
Hyperparameters
Initialization of all hyperparameters
"""
import torch as T
# initialize parameters for the environment and algorithm
def init_params(market_model: str):
    # parameters for the model                                         
    if market_model == 'Black_Scholes':
        env_params = {'num_assets' : 3, # number of total assets in market (risk-free and risky)
                'vols' : T.tensor([0.02, 0.03]), # volatilities of risky assets
                'drifts' : T.tensor([0.08, 0.05]), # drifts of risky assets
                'cov_matrix' : T.tensor([[1.0, 0.3], [0.3, 1.0]]), # covariance structure of risky asset prices
                'returns_req' : 0.05, # returns requirement for goal
                'goal_prob' : 0.95, # confidence level of meeting returns requirement 
                'interest_rate': 0.025, # interest rate of risk-free asset
                'alpha': 0.1,
                'beta': 0.9, 
                'q': 0.75, 
                'phi' : 0.0, # transaction costs
                'T' : 1, # trading horizon
                'Ndt' : 1, # number of periods
                'init_wealth': 100.0, # initial wealth
                'S0': T.tensor([1.0, 1.0, 1.0]) # initial risky asset prices
                }
    
    if market_model == 'SABR':
        pass
    
    # parameters for the algorithm
    
    algo_params = {'num_epochs' : 1_000, # number of iterations of entire training loop
                   'batch_size' : 250, # mini-batch size for gradient estimates
                   'num_layers': 10, # number of layer in policy network
                   'hidden_size': 16, # width of hidden layers of policy network
                   'learn_rate': 0.1, # learning rate of policy network
                   'init_lamb' : 1.0, # initial Lagrange multiplier    
                   'init_mu' : 10.0, # initial penalty strength
                   'pen_strength_lr' : 1.5, # penalty strength learning rate
                   'pen_update_freq' : 5, # Lagrange multiplier and penalty strength update frequency
                   } 

    return env_params, algo_params

