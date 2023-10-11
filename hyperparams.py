"""
Hyperparameters
Initialization of all hyperparameters
"""
import torch as T
# initialize parameters for the environment and algorithm
def init_params(market_model: str, num_assets=3):
    # parameters for the model                                         
    if market_model == 'BS':
        env_params = {'num_assets': num_assets, # number of total assets in market (risk-free and risky)
                'drifts': T.tensor([0.06, 0.09]), # drifts of risky assets
                'vols': T.tensor([0.12, 0.18]), # volatilities of risky assets
                'corr_matrix': T.tensor([[1.0, 0.2],
                                         [0.2, 1.0]]), # correlation structure of risky asset prices
                'interest_rate': 0.025, # interest rate of risk-free asset                
                'goal_prob': 0.0, # confidence level of meeting returns requirement 
                'returns_req': 0.0, # returns requirement for goal
                'alpha': 0.85,
                'beta': 0.9, 
                'q': 1.0, 
                'phi': 0.0, # transaction costs
                'T': 1.0, # trading horizon
                'Ndt': 1, # number of periods
                'init_wealth': 1.0, # initial wealth
                'S0': T.tensor([1.0, 1.0, 1.0]) # initial risky asset prices
                }
    
    if market_model == 'Factor':
        
        env_params = {'num_assets' : num_assets, # number of total assets in market (risk-free and risky)
                'systemic_var': 0.02**2, # variance of systemic factor
                'idio_means': T.tensor([0.03*i for i in range(1, num_assets)]), # means of idiosyncratic factors
                'idio_vars': T.tensor([(0.025)**2 * (i**2) for i in range(1, num_assets)]), # variances of idiosyncratic factors
                'returns_req' : 0.05, # returns requirement for goal
                'goal_prob' : 0.525, # confidence level of meeting returns requirement 
                'interest_rate': 0.05, # interest rate of risk-free asset
                'alpha': 0.99,
                'beta': 0.9, 
                'q': 1.0, 
                'phi' : 0.0, # transaction costs
                'T' : 1.0, # trading horizon
                'Ndt' : 1, # number of periods
                'init_wealth': 1.0, # initial wealth
                'S0': T.tensor([1.0, 1.0, 1.0]) # initial risky asset prices
                }
        
    # parameters for the algorithm
    
    algo_params = {'num_epochs' : 1_000, # number of iterations of entire training loop
                   'batch_size' : 750, # mini-batch size for gradient estimates
                   'num_layers': 10, # number of layer in policy network
                   'hidden_size': 16, # width of hidden layers of policy network
                   'learn_rate': 0.0001, # learning rate of policy network
                   'init_lamb' : 1.0, # initial Lagrange multiplier    
                   'init_mu' : 10.0, # initial penalty strength
                   'pen_strength_lr' : 1.5, # penalty strength learning rate
                   'pen_update_freq' : 5, # Lagrange multiplier and penalty strength update frequency
                   } 

    return env_params, algo_params

# print parameters for the environment and algorithm
def print_params(envParams, algoParams, market_model):
    
    if market_model == 'BS':
        print('*  Drifts: ', envParams["drifts"],
                '\n   Volatilites: ', envParams["vols"],
                '\n   Risky Asset Correlation: ', envParams["corr_matrix"][0][1],
                '\n   Initial Prices: ', envParams["S0"],
                '\n   Initial Interest rate: ', envParams["interest_rate"],
                '\n   Initial Wealth: ', envParams["init_wealth"],
                '\n   T: ', envParams["T"],
                '\n   Number of periods: ', envParams["Ndt"])
        print('*  Batch size: ', algoParams["batch_size"],
                '\n   Number of epochs: ', algoParams["num_epochs"],
                '\n   Learning Rate: ', algoParams["learn_rate"], 
                '\n   Width of Hidden Layers: ', algoParams["hidden_size"],
                '\n   Number of Hidden Layers: ', algoParams["num_layers"])
        
    if market_model == 'Factor':
        print('*  Means of Idiosyncratic Factors: ', envParams["idio_means"],
                '\n   Variances of Idiosyncratic Factors: ', envParams["idio_vars"],
                '\n   Initial Prices: ', envParams["S0"])
        print('*  Initial Interest rate: ', envParams["interest_rate"],
                '\n   Initial Wealth: ', envParams["init_wealth"],
                '\n   T: ', envParams["T"],
                '\n   Number of periods: ', envParams["Ndt"])
        print('*  Batch size: ', algoParams["batch_size"],
                '\n   Number of epochs: ', algoParams["num_epochs"],
                '\n   Learning Rate: ', algoParams["learn_rate"], 
                '\n   Width of Hidden Layers: ', algoParams["hidden_size"],
                '\n   Number of Hidden Layers: ', algoParams["num_layers"])