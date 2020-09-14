   """Implementation of coin mixture via EM algorithm"""


   import pandas as pd
   import numpy as np
   from scipy.stats import bernoulli, binom
   from scipy.optimize import minimize

   
   class CoinEM:
       def __init__(self, n_components=2, max_iter=50):
           self.n_components = n_components
           self.max_iter = max_iter
        