#!/usr/bin/env python
# coding: utf-8

# In[67]:

import warnings
warnings.filterwarnings("ignore")
import environments_fully_observable 
#import environments_partially_observable
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
#from IPython.display import display, clear_output
import torch
import torch.nn as nn
import torch.nn.functional as F
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)
from Snake_agent_train import Snake
from Baseline import baseline_policy



# In[68]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# function to standardize getting an env for the whole notebook
def get_env(n, size):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    e = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    # or environments_partially_observable.OriginalSnakeEnvironment(n, size, 2)
    return e
env_ = get_env(2,7)
GAMMA = .9
ITERATIONS = 5000

#fig,axs=plt.subplots(1,min(len(env_.boards), 5), figsize=(10,3))
#for ax, board in zip(axs, env_.boards):
#    ax.get_yaxis().set_visible(False)
#    ax.get_xaxis().set_visible(False)
#    ax.imshow(board, origin="lower")


# In[69]:


class DQNModel_3(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNModel_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = input_shape[0], out_channels = 32, kernel_size = 5, stride = 1, padding = 0)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * (input_shape[1] - 4) * (input_shape[1] - 4), 64)  
        self.dense2 = nn.Linear(70, 64)  
        self.output_layer = nn.Linear(64, num_actions)
        self.input_shape = input_shape
        self.num_actions = num_actions

    def forward(self, inputs, inputs_mod):
        x = F.relu(self.conv1(inputs))
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = torch.cat((x, inputs_mod), dim=-1)  # Concatenate along the last dimension
        x = F.relu(self.dense2(x))
        x = self.output_layer(x)
        return x


# In[70]:


loaded_model = DQNModel_3((4,6,6), 4)
loaded_model.load_state_dict(torch.load('snake_agent_2.pth'))


# In[71]:


snake = Snake(0.99, 1, 1000, 6, 5000)  


# In[72]:


#loaded_model = DQNModel_3((4,6,6), 4)

#loaded_model.load_state_dict(torch.load('snake_agent_2.pth'))


# In[73]:


bsln = baseline_policy(200,6)


# In[74]:


import seaborn as sns


# In[75]:


def test_function():
    np.random.seed(1)
    env = get_env(5000,6)
    state = env.to_state()
    mod_inp = torch.tensor(snake.enrich(state), dtype = torch.float32)
    state = torch.permute(torch.tensor(state, dtype = torch.float32), (0,3,1,2))
    with torch.no_grad():
        action = [[a] for a in np.argmax(loaded_model(state, mod_inp).detach().numpy(), axis = 1)]
        rewards = env.move(action)
        for _ in range(224):
            state = env.to_state()
            mod_inp = torch.tensor(snake.enrich(state), dtype = torch.float32)
            state = torch.permute(torch.tensor(state, dtype = torch.float32), (0,3,1,2))
            action = [[a] for a in np.argmax(loaded_model(state, mod_inp).detach().numpy(), axis = 1)]
            rewards = np.hstack([rewards, env.move(action)])
    rewards = snake.filter_rewards(rewards)
    rew_baseline = bsln.main()

    print(f"Mean length of baseline : {np.round(np.mean([len(a) for a in rew_baseline]), 2)}")
    print(f"Mean length of agent : {np.round(np.mean([len(a) for a in rewards]), 2)}")
    
    data_sets = [[len(a) for a in rew_baseline], [len(a) for a in rewards]]
    
    rew_agent = [np.mean(a) for a in rewards]
    rew_baseline = [np.mean(a) for a in rew_baseline]
    print(f"Mean rewards agent : {np.round(np.mean(rew_agent), 4)}  -  SD : {np.round(np.std(rew_agent), 5)}")
    print(f"Mean rewards baseline : {np.round(np.mean(rew_baseline), 4)} -  SD : {np.round(np.std(rew_baseline), 5)}")
    
    data_sets.extend([[np.mean(a) for a in rew_baseline],[np.mean(a) for a in rewards]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i == 0:
            sns.kdeplot(data_sets[i], ax=ax, label = 'Baseline')
            sns.kdeplot(data_sets[i + 1], ax=ax, label = 'Agent')
            ax.set_title("Distributions of avg. lengths")
            ax.legend()
        else:
            sns.kdeplot(data_sets[i + 1], ax=ax, label = 'Baseline')
            sns.kdeplot(data_sets[i + 2], ax=ax, label = 'Agent')
            ax.set_title("Distributions of avg. rewards")
            ax.legend()
    
    plt.tight_layout()
    plt.show()    
    return rew_agent, rew_baseline


# In[76]:


a,b = test_function()

