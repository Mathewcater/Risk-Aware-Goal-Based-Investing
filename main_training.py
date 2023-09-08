# # imports 
# import numpy as np
# import torch as T
# import pdb
# from models import *
# from envs import *
# from hyperparams import *
# from utils import *
# from agents import *
# from torch.distributions.multivariate_normal import MultivariateNormal
# from torch.distributions.normal import Normal
# from torch.nn import Softmax
# from tqdm import tqdm


# # For reproducibility 
# T.manual_seed(54321)

# # Anamoly Detection 
# T.autograd.set_detect_anomaly(True)


# def Train(algo_params: dict, env):
#     """Generate, and train, DPG model on the environment specifications
#     provided in 'env' and executed with the hyperparams specified in 'algoparams'.

#     Args:
#         algo_params (dict): Dictionary containing training hyperparams
#         env (Environment): Environment object containing environment/problem params
#                             (price dynamics, max and min inventory levels etc.)
#     Returns:
#         pi (PolicyANN): Trained network; learned policy.
#         cum_rews (T.tensor): 
#     """
    
#     # initialize agent 
#     agent = Agent()
    
#     # main training loop

#     for m in tqdm(range(algo_params["num_epochs"])):
        
#         loss, RDEU = agent.update_policy()
        
    
#         if m % algo_params["pen_update_freq"] == 0:
#             agent.update_multipliers()     
                      
#     return pi

