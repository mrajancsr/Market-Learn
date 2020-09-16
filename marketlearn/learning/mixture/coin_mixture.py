   """Implementation of coin mixture via EM algorithm"""


   import pandas as pd
   import numpy as np
   from scipy.stats import bernoulli, binom
   from scipy.optimize import minimize

   
   class CoinEM:
       def __init__(self, n_components=2, max_iter=50):
           self.n_components = n_components
           self.max_iter = max_iter



def coin_em(trials, theta_A=None, theta_B=None, maxiter=0):
    # initial guess
    theta_A = theta_A or np.random.random()
    theta_B = theta_B or np.random.random()
    thetas = [(theta_A, theta_B)]
    # iterate
    for c in range(maxiter):
        print("#%d:\t%0.2f %0.2f" % (c, theta_A, theta_B))
        heads_A, tails_A, heads_B, tails_B = e_step(trials, theta_A, theta_B)
        theta_A, theta_B = m_step(heads_A, tails_A, heads_B, tails_B)
        thetas.append((theta_A, theta_B))
    return thetas, (theta_A, theta_B)

def e_step(trials, theta_A, theta_B):
    """produce expected value for  heads_A, heads_B, tails_B
    over the flips given the coin biases
    """
    heads_A, tails_A = 0, 0
    heads_B, tails_B = 0, 0
    for trial in trials:
        likelihood_A = coin_likelihood(trial, theta_A)
        likelihood_B = coin_likelihood(trial, theta_B)
        p_A = likelihood_A / (likelihood_A + likelihood_B)
        p_B = likelihood_B / (likelihood_A + likelihood_B)
        heads_A += p_A * trial.count("H")
        tails_A += p_A * trial.count("T")
        heads_B += p_B * trial.count("H")
        tails_B += p_B * trial.count("T")
    return heads_A, tails_A, heads_B, tails_B

def m_step(heads_A, tails_A, heads_B, tails_B):
    theta_A = heads_A / (heads_A + heads_B)
    theta_B = heads_B / (heads_A + heads_B)
    return theta_A, theta_B

def coin_likelihood(trial, bias):
    numHeads = trial.count("H")
    n = len(trial)
    return binom(n, bias).pmf(numHeads)
