#!/usr/bin/env python
# coding: utf-8

# In[1]:


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
# from hyperopt.pyll.stochastic import sample
# import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import os
import torch

from Get_GNG_Nodes import GrowingNeuralGas
from Geometry_Score import *


# ### Generating the dataset

# In[2]:


DATA_PATH = '../data/'
print(os.listdir(DATA_PATH))

embeddings = torch.load(DATA_PATH + 'bert_embeddings_128maxlength.pt', map_location=torch.device('cpu'))
emb_tensor = torch.stack(list(embeddings.values()))
X = emb_tensor.numpy()

# np.random.shuffle(X)
# X = X[:10]
X.shape


# In[3]:


param_hyperopt= {
    'e_b': hp.uniform('e_b', 0.1, 0.5),
    'e_n': hp.uniform('e_n', 0.005, 0.009),
    'age_max': scope.int(hp.quniform('age_max', 20, 200, 10)),
    'a': hp.uniform('a', 0.4, 0.7),
    'd': hp.uniform('d', 0.7, 0.99),
    'l': scope.int(hp.quniform('l', 200, 400, 10)),
#     'nodes': scope.int(hp.quniform('nodes', 100, 200, 5)),
}

param_hyperopt


# ### Hyperopt function

# In[4]:


def hyperopt(param_hyperopt, X, num_eval):    
    start = time.time()
    
    def objective_function(param_hyperopt):
        print(param_hyperopt)
        
        fmodel = GrowingNeuralGas(X)
        gng_out = fmodel.GNG(e_b=param_hyperopt['e_b'], 
                       e_n=param_hyperopt['e_n'], 
                       age_max=param_hyperopt['age_max'], 
                       a=param_hyperopt['a'], 
                       d=param_hyperopt['d'],
                       ncol = X.shape[1],
                       nrow = X.shape[0],
                       l=param_hyperopt['l'],
#                        num_nodes=param_hyperopt['nodes'],
                       num_nodes=100,

                       plot_evolution=True
                      )
    
        gng_out = np.matrix(gng_out)
        
        rltsdata = rlts(X, n=100, L_0=32, i_max=100, gamma=1.0/8)
        rltsgng = rlts(gng_out, n=100, L_0=32, i_max=100, gamma=1.0/8)
        
#         print(rltsdata.shape, rltsgng.shape)
        
        gm_score = geom_score(rltsdata, rltsgng)

        data_t = torch.from_numpy(X)
        gngout_t = torch.from_numpy(gng_out)
  
        loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.05)
        sinkhorn_loss = loss(data_t.float(), gngout_t.float())
        
        loss = gm_score + sinkhorn_loss
        print(loss)
        
        return {'loss': loss, 
                'status': STATUS_OK
               }

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_hyperopt, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.RandomState(1))
#     for x in trials.trials:
#         print(x)
#     loss = [x['result']['loss'] for x in trials.trials]
#     print(loss)

    
    return best_param


# ### Execute Hyperopt for 75 iterations

# In[5]:
# for i in trails.trails:
#     print(i)


num_eval = 100
results_hyperopt = hyperopt(param_hyperopt, X, num_eval)
print(results_hyperopt)


# In[ ]:





# In[ ]:




