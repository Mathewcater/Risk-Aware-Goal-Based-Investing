# imports 
import os
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from hyperparams import *
from envs import *
from utils import *
from agents import *
from scipy import stats
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})

env_params, algo_params = init_params('Black_Scholes')
env = BS_Environment(env_params)
agent = Agent(env, algo_params)
agent.Train()


(fig1, ax1), (fig2, ax2) = plt.subplots(1, 1, sharey=True), \
                           plt.subplots(1, 1, sharey=True)   

# plot RDEU through training
ax1.plot(np.arange(algo_params["num_epochs"]) + 1, T.stack(agent.RDEU_history))
ax1.set(xlabel='Epochs', ylabel='RDEU', title='RDEU per Epoch; Learned ANN Policy')    

# visualise policy

    
# term_wealth of learned strategy
term_wealth = agent.term_wealth_dist.detach()
mean_term_wealth, std_term_wealth = T.mean(term_wealth), T.std(term_wealth)
num_bins = 50
domain = np.linspace(mean_term_wealth - 5*std_term_wealth, mean_term_wealth + 5*std_term_wealth, 1500)
term_wealth_kde = stats.gaussian_kde(term_wealth)
ax2.hist(term_wealth, density=True, bins=num_bins, color = colors[0], alpha=0.5)
ax2.set(title=r'Distribution of Terminal Wealth', xlabel=r'Terminal Wealth: $X^{\theta}$', ylabel='Density')
ax2.plot(domain, term_wealth_kde(domain))

# check if constraint satisfied
c = env_params["returns_req"]
print(f'Probability of returns exceeding {100*c}%: {agent.return_prob}')

plt.show() 