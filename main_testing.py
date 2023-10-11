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

T.random.manual_seed(54321)

# env_params, algo_params = init_params('BS')
# env = BS_Environment(env_params)
# policy = PolicyANN(env, algo_params)
# agent = Agent(env, algo_params, policy)
# agent.Train()

(fig1, ax1), (fig2, ax2) = plt.subplots(1, 1, sharey=True), \
                           plt.subplots(1, 1, sharey=True)   

# plot RDEU through training
# RDEU_hist = T.stack(agent.RDEU_history)
# ax1.plot(np.arange(algo_params["num_epochs"]) + 1, RDEU_hist)
# ax1.set(xlabel='Epochs', ylabel='RDEU', title='RDEU per Epoch; Learned ANN Policy')    
    
# # term_wealth of learned strategy
# term_wealth = agent.term_wealth_dist.detach().squeeze()
# mean_term_wealth, std_term_wealth = T.mean(term_wealth), T.std(term_wealth)
# num_bins = 50
# domain = np.linspace(mean_term_wealth - 5*std_term_wealth, mean_term_wealth + 5*std_term_wealth, 1500)

# if std_term_wealth >= 1.0e-5:
#     term_wealth_kde = stats.gaussian_kde(term_wealth)
#     ax2.plot(domain, term_wealth_kde(domain))
    
# ax2.hist(term_wealth, density=True, bins=num_bins, color=colors[0], alpha=0.5)
# ax2.set(title=r'Distribution of Terminal Wealth', xlabel=r'Terminal Wealth: $X^{\theta}$', ylabel='Density')

# # check if constraint satisfied
# c = env_params["returns_req"]
# print(f'Probability of returns exceeding {100*c}%: {agent.return_prob}')

# position_hist = agent.position_history.detach().squeeze()
# return_prob = T.tensor(agent.return_prob)
# risk = RDEU_hist[-1]
# print(f'position_hist: {position_hist}')
# print(f'term_wealth.unique(): {term_wealth.unique()}')
# print(f'return_prob: {return_prob}')
# print(f'risk: {risk}')
# T.save(position_hist, 'position_hist_p_0.50.pt')
# T.save(term_wealth, 'term_wealth_p_0.50.pt')
# T.save(return_prob, 'return_prob_p_0.95.pt')

# T.save(risk, 'risk_p_0.50.pt')


###################### VARYING THE GOAL ########################

# confidence levels considered:
conf_levels = np.array([0.95, 0.6, 0.55, 0.525, 0.50])
column_titles_p = [r'$\mathbold{p = 0.95}$',
                    r'$\mathbold{p = 0.60}$',
                    r'$\mathbold{p = 0.55}$',
                    r'$\mathbold{p = 0.525}$',
                    r'$\mathbold{p = 0.50}$']
column_titles_p.reverse()
col_tiles_p = [r'$\mathbold{p = 0.95}$',
                    r'$\mathbold{p = 0.60}$',
                    r'$\mathbold{p = 0.55}$',
                    r'$\mathbold{p = 0.525}$',
                    r'$\mathbold{p = 0.50}$']
row_titles = [r'$\mathbold{\pi_{\theta}^{(0)}}$',
              r'$\mathbold{\pi_{\theta}^{(1)}}$',
              r'$\mathbold{\pi_{\theta}^{(2)}}$'] 

# ===================== P&L PLOTS ========================

PnLs_p = [T.load('term_wealth_p_0.95.pt') - 1.0,
          T.load('term_wealth_p_0.60.pt') - 1.0,
          T.load('term_wealth_p_0.55.pt') - 1.0,
          T.load('term_wealth_p_0.525.pt') - 1.0,
          T.load('term_wealth_p_0.50.pt') - 1.0]

for idx, PnL in enumerate(PnLs_p):
    mean_PnL, std_PnL = T.mean(PnL), T.std(PnL)
    num_bins = 50
    domain = np.linspace(mean_PnL - 5*std_PnL, mean_PnL + 5*std_PnL, 1500)

    if std_PnL >= 1.0e-5:
        PnL_kde = stats.gaussian_kde(PnL)
        ax2.plot(domain, PnL_kde(domain), color=colors[idx])
        
    ax2.hist(PnL, density=True, bins=num_bins, color=colors[idx], alpha=0.5)
    ax2.set(title=r'Distribution of P\&L', xlabel=r'P\&L', ylabel='Density')
    
ax2.legend(col_tiles_p, loc="upper left")
ax2.set_ylim(top=14)
ax2.set_xlim(left=-0.50, right=0.7)

# ============================ POLICY PLOTS ============================

position_hists_p = [T.load('position_hist_p_0.95.pt'),
                    T.load('position_hist_p_0.60.pt'),
                    T.load('position_hist_p_0.55.pt'),
                    T.load('position_hist_p_0.525.pt'),
                    T.load('position_hist_p_0.50.pt')]

position_hists_p.reverse()

return_probs_p = [T.load('return_prob_p_0.95.pt'),
                    T.load('return_prob_p_0.60.pt'),
                    T.load('return_prob_p_0.55.pt'),
                    T.load('return_prob_p_0.525.pt'),
                    T.load('return_prob_p_0.50.pt')]

print(f'return_probs_p: {return_probs_p}')

position_hists_p = [pos_hist[0] for pos_hist in position_hists_p]

# figure parameters
num_assets = 3
nrows = num_assets
ncols = len(conf_levels)
fig, axes = plt.subplots(nrows, ncols, sharey='all', sharex='all', figsize=(10,7))


for prob_idx, prob_val in enumerate(conf_levels):
    
    for idx_asset in range(num_assets):
     
        temp = axes[idx_asset, prob_idx].imshow(position_hists_p[prob_idx][idx_asset].reshape(1,1),
                                                interpolation='none',
                                                cmap=cmap,
                                                aspect='auto',
                                                vmin=np.array(0.0),
                                                vmax=np.array(1.0)
                                                )
        axes[idx_asset, prob_idx].set_xticks([])
        axes[idx_asset, prob_idx].set_yticks([])
            
        if prob_idx == ncols - 1:
            axes[idx_asset, prob_idx].set_ylabel(row_titles[idx_asset],
                                                rotation='horizontal',
                                                fontsize=25,
                                                fontweight='semibold',
                                                labelpad=40,
                                                y=0.72)
            axes[idx_asset, prob_idx].yaxis.set_label_position("right")

        if idx_asset == 0:
            axes[idx_asset, prob_idx].set_title(column_titles_p[prob_idx] + '\n',
                                                rotation='horizontal',
                                                fontsize=22,
                                                fontweight='semibold')
            
fig.colorbar(temp, ax=axes, orientation='horizontal', shrink=0.8, pad=0.1)
fig.savefig('positions_p.pdf', bbox_inches = 'tight')
fig2.savefig('PnLs_p.pdf', bbox_inches = 'tight')

#####################################################################################


################################## VARYING ALPHA #######################################

# CVaR thresholds considered:
# alphas = np.array([0.01, 0.75, 0.85, 0.99])
# column_titles_alpha = [r'$\mathbf{CVaR: 0.01}$',
#                        r'$\mathbf{CVaR: 0.75}$',
#                        r'$\mathbf{CVaR: 0.85}$',
#                        r'$\mathbf{CVaR: 0.99}$']

# # column_titles_alpha.reverse()

# row_titles = [r'$\mathbold{\pi_{\theta}^{(0)}}$',
#               r'$\mathbold{\pi_{\theta}^{(1)}}$',
#               r'$\mathbold{\pi_{\theta}^{(2)}}$']

# # ======================================== P&L PLOTS =======================================

# PnLs_alpha = [T.load('term_wealth_0.01.pt') - 1.0,
#                   T.load('term_wealth_0.75.pt') - 1.0,
#                   T.load('term_wealth_0.85.pt') - 1.0,
#                   T.load('term_wealth_0.99.pt') - 1.0]

# for idx, PnL in enumerate(PnLs_alpha):
#     mean_PnL, std_PnL = T.mean(PnL), T.std(PnL)
#     num_bins = 50
#     domain = np.linspace(mean_PnL - 5*std_PnL, mean_PnL + 5*std_PnL, 1500)

#     if std_PnL >= 1.0e-5:
#         PnL_kde = stats.gaussian_kde(PnL)
#         ax2.plot(domain, PnL_kde(domain), color=colors[idx])
        
#     ax2.hist(PnL, density=True, bins=num_bins, color=colors[idx], alpha=0.5)
#     ax2.set(title=r'Distribution of P\&L', xlabel=r'P\&L', ylabel='Density')
    
# ax2.legend(column_titles_alpha, loc="upper left")
# ax2.set_ylim(top=12)
# ax2.set_xlim(left=-0.50, right=0.7)

# # ========================================= POLICY PLOTS =====================================

# position_hists_alpha = [T.load('position_hist_0.01.pt'),
#                         T.load('position_hist_0.75.pt'),
#                         T.load('position_hist_0.85.pt'),
#                         T.load('position_hist_0.99.pt')]

# # position_hists_alpha.reverse()

# # return_probs_alpha = [T.load('return_prob_0.01.pt'),
# #                     T.load('return_prob_0.75.pt'),
# #                     T.load('return_prob_0.85.pt'),
# #                     T.load('return_prob_0.99.pt')]

# position_hists_alpha = [pos_hist[0] for pos_hist in position_hists_alpha]

# # figure parameters
# num_assets = 3
# nrows = num_assets
# ncols = len(alphas)
# fig, axes = plt.subplots(nrows, ncols, sharey='all', sharex='all', figsize=(10,7))


# for alpha_idx, alpha_val in enumerate(alphas):
    
#     for idx_asset in range(num_assets):
        
#         temp = axes[idx_asset, alpha_idx].imshow(position_hists_alpha[alpha_idx][idx_asset].reshape(1,1),
#                                                  interpolation='none',
#                                                  cmap=cmap,
#                                                  aspect='auto',
#                                                  vmin=np.array(0.0),
#                                                  vmax=np.array(1.0)
#                                                 )
#         axes[idx_asset, alpha_idx].set_xticks([])
#         axes[idx_asset, alpha_idx].set_yticks([])
        
#         if alpha_idx == ncols - 1:
#             axes[idx_asset, alpha_idx].set_ylabel(row_titles[idx_asset],
#                                                   rotation='horizontal',
#                                                   fontsize=25,
#                                                   fontweight='semibold',
#                                                   labelpad=40,
#                                                   y=0.72)
#             axes[idx_asset, alpha_idx].yaxis.set_label_position("right")

#         if idx_asset == 0:
#             axes[idx_asset, alpha_idx].set_title(column_titles_alpha[alpha_idx] + '\n',
#                                                  rotation='horizontal',
#                                                  fontsize=22,
#                                                  fontweight='semibold')
            
# fig.colorbar(temp, ax=axes, orientation='horizontal', shrink=0.8, pad=0.1)

fig.savefig('positions_p.pdf', bbox_inches='tight')
fig2.savefig('PnLs_p.pdf', bbox_inches='tight')
plt.show() 