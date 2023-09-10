# imports

import numpy as np
import torch as T
from torch.nn import ReLU
from torch.distributions.normal import Normal
from envs import *
from models import * 
from tqdm import tqdm
import pdb


# helper functions 

def est_cdf(x, X, h):
    """ Produce estimate of the cdf of the terminal wealth given mini-batch.
    """
    
    return (1/len(X))*T.sum(Normal(0,1).cdf((x - X)/h))
    
def GetGradient(batch, h):
    """Calculate the pdf and the gradient of the distribution function of X^0
    """
    
    batch_no_grad = batch.detach()
    normal = T.distributions.Normal(0,1)
    z_score = (batch_no_grad.reshape(1, -1) - batch_no_grad.reshape(-1, 1))/h
    f_x = T.mean(T.exp(normal.log_prob(z_score)), axis = 0)/h
    grad_F_x = -T.mean(T.exp(normal.log_prob(z_score))*batch.reshape(-1,1), axis = 0)/h 

    return f_x, grad_F_x

#################################################################################

# agent class 
class Agent():
    
    def __init__(self, env, algo_params, policy):
        self.env = env
        self.policy = policy 
        self.algo_params = algo_params
        self.lamb = algo_params["init_lamb"]
        self.mu = algo_params["init_mu"]
        self.init_history()
            
    def init_history(self):
        
        self.loss_history = []
        self.constraint_prob_history = []
        self.lam_history = [self.algo_params["init_lamb"]]
        self.mu_history = [self.algo_params["init_mu"]]
        self.RDEU_history = []
        
    def update_history(self, RDEU, return_prob):
        """Store RDEU and probability of
        """
        
        self.RDEU_history.append(RDEU)
        self.constraint_prob_history.append(return_prob)
   
    def gamma(self, u):
        """Distortion function corresponding to the alpha-beta risk. Can be evaluated at arrays.
        """
        alpha = self.env.params["alpha"]
        beta = self.env.params["beta"]
        q = self.env.params["q"]
        
        norm_factor = q*alpha + (1-q)*(1-beta)
        
        return (1/norm_factor)*(q*(u <= alpha) + (1-q)*(u > beta))
    
    def est_RDEU(self, batch, len_partition=10_000):
        """Estimate RDEU given a mini-batch using Riemann sum with right endpoint.
        """
        partition = T.linspace(0, 1, len_partition)
        values = T.quantile(batch, q=partition)*self.gamma(partition)
        
        return (-1.0)*T.dot(values, (1/len_partition)*T.ones(len_partition))
    
    def compute_loss(self, batch):
        """Compute estimate of augmented Lagrangian from mini-batch of terminal wealth samples.
        """
        
        batch_no_grad = batch.detach()
        M, c, x0, p = self.algo_params["batch_size"], self.env.params["returns_req"], \
                        self.env.params["init_wealth"], self.env.params["goal_prob"]   
        
        # compute bandwidth of KDE using Silverman's rule.
        h = max(0.01, 1.06*(M**(-1/5))*T.std(batch_no_grad))
        
        # compute loss
        
        # distribution function evaluated at mini-batch
        F_x = T.stack([est_cdf(x, batch_no_grad, h) for x in batch_no_grad])
        
        # gradient of distribution function and pdf each evaluated at mini-batch
        f_x, grad_F_x = GetGradient(batch, h) 
        
        # risk measure weight
        RM_weight = self.gamma(F_x)
        
        # constraint error term: c[X,p]= (p - (1 - F((1+c)X_0)))_+
        constr_err = ReLU()(p - (1 - est_cdf((1+c)*x0, batch_no_grad, h)))
        
        LM_weight = (self.lamb + self.mu*constr_err)
        grad_F_x0 = -T.mean(T.exp(Normal(0,1).log_prob((1+c)*x0 - batch_no_grad))*batch)
        
        # compute loss
        loss = T.mean(RM_weight*(grad_F_x/f_x)) + LM_weight*grad_F_x0
        
        # probability of meeting goal        
        return_prob = 1.0 - est_cdf((1+c)*x0, batch_no_grad, h) 
        
        # risk of strategy
        RDEU = self.est_RDEU(batch_no_grad)
        
        return loss, RDEU, return_prob

    def update_multipliers(self, constr_err):
        """Update Lagrange multipliers. 
        """
        self.lamb += self.mu*constr_err
        self.mu *= self.algo_params["pen_strength_lr"]
          
    def update_policy(self):
        """Generate mini-batch of terminal wealth samples and update policy.
        """
        
        curr_state = self.env.reset(self.algo_params["batch_size"])
        
        terminal_wealth = self.env.params['init_wealth'] * T.ones(self.algo_params["batch_size"], 1)
        pi = T.zeros(self.algo_params["batch_size"], self.env.params["Ndt"], self.env.params["num_assets"])
        
        for i, _ in enumerate(range(self.env.params["Ndt"])):
            
            # take action
            action = self.policy(curr_state[:, :-1])
            
            # get new state and reward
            new_state, rew = self.env.step(curr_state, action) 
            
            terminal_wealth += rew.clone()
        
            # update state
            curr_state = new_state.clone()
            
            # store actions
            pi[:,i,:] = action
                                    
        # compute loss, RDEU and probability of posting desired return estimates using terminal_wealth
        loss, RDEU, return_prob = self.compute_loss(terminal_wealth)
        
        # store loss and RDEU to track training
        self.update_history(RDEU, return_prob) 
        
        # perform update using mini-batch of terminal wealth samples
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()
        
        # decay learning rate
        if self.policy.scheduler.get_last_lr()[0] >= 5e-4:
            self.policy.scheduler.step()

        return loss, RDEU, return_prob, terminal_wealth, pi

    def Train(self):
        """Main training loop.
        """
        for m in tqdm(range(self.algo_params["num_epochs"])):
            
            # at final epoch of training, store probability of meeting goal, distribution of term. wealth and 
            # position history over final mini-batch
            if m == self.algo_params["num_epochs"] - 1:
                self.algo_params["batch_size"] = 10_000
                
                # update policy     
                _, _, return_prob, term_wealth_samps, pi = self.update_policy() 
                self.term_wealth_dist = term_wealth_samps 
                self.return_prob = return_prob
                self.pi = pi
            
            else:
                # update policy     
                _, _, return_prob, term_wealth_samps, pi = self.update_policy() 
                
                # update Lagrange multipliers
                if m % self.algo_params["pen_update_freq"] == 0:
                    self.update_multipliers(constr_err=ReLU()(self.env.params["goal_prob"] 
                                                              - return_prob))    