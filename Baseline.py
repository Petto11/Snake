#!/usr/bin/env python
# coding: utf-8

# In[1]:


import environments_fully_observable 
#import environments_partially_observable
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
#from IPython.display import display, clear_output
from scipy.spatial.distance import cityblock
import seaborn as sns
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# function to standardize getting an env for the whole notebook
def get_env(n, size):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    e = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    # or environments_partially_observable.OriginalSnakeEnvironment(n, size, 2)
    return e


# In[3]:


class baseline_policy():
    def __init__(self, n, size):
        self.n = n
        self.size = size
        self.env = get_env(self.n, self.size)
        self.all_rewards = []
        self.terminated = False
        self.total_moves = 0
    
    def closest_corner(self):
        
        state = self.env.to_state()
        head_index = np.argwhere(state[:,:,:,3].reshape(-1,self.size * self.size) == 1)[:,1]
        self.head_pos = np.array([head_index // (self.size), head_index % (self.size)]).T
        self.corners = {'Up_left' : [1,1], 'Up_right' : [1,self.size - 2], 'Bottom_left': [self.size - 2, 1], \
                                'Bottom_right' : [self.size - 2, self.size - 2]}
        
        manhattan_distances_ul = np.abs(self.head_pos - self.corners['Up_left']).sum(axis=1)
        manhattan_distances_ur = np.abs(self.head_pos - self.corners['Up_right']).sum(axis=1)
        manhattan_distances_bl = np.abs(self.head_pos - self.corners['Bottom_left']).sum(axis=1)
        manhattan_distances_br = np.abs(self.head_pos - self.corners['Bottom_right']).sum(axis=1)
        
        self.temp = np.array([manhattan_distances_ul, manhattan_distances_ur,\
                       manhattan_distances_bl, manhattan_distances_br]).T
        
        self.best_corner = [list(self.corners.keys())[a] for a in np.argmin(self.temp, axis = 1)]
        
                
    def move_to_closest_corner(self):
        direction = np.array([a-b for a,b in zip(self.head_pos, [self.corners[c] for c in self.best_corner])])
        act_up = np.where(direction[:,1] < 0, 1 * abs(direction[:,1]), 3 * abs(direction[:,1]))  
        act_up = np.where(act_up == 0 , 5, act_up)
        act_d = np.where(direction[:,0] < 0, 5 * abs(direction[:,0]), 2 * abs(direction[:,0]))  
        act_d[act_d == 5] = -999
        act_d[act_d == 0] = 5
        act_d[act_d == -999] = 0
        self.actions_one_time = [[[a] for a in act_d], [[a] for a in act_up]]
        self.rewards = self.env.move(self.actions_one_time[0])
        self.rewards = np.hstack([self.rewards, self.env.move(self.actions_one_time[1])])
        self.boards = np.copy(self.env.boards)
        
    def find_indications(self):
        all_actions = [0,1,2,3]
        corners_indications = {'Up_left' : [0,1,2,3], 'Up_right' : [0,3,2,1], "Bottom_left":[2,1,0,3],\
                                   'Bottom_right' : [2,3,0,1]}
        
        indications = np.array([corners_indications[corner] for corner in self.best_corner])
        self.actions = np.tile(indications[:,0][:, np.newaxis], (1, self.size - 3))
        self.actions = np.hstack([self.actions, np.tile(indications[:,1][:, np.newaxis], (1, 1))])
        for _ in range(int((self.size - 3) // 2)):
            self.actions = np.hstack([self.actions, np.tile(indications[:,2][:, np.newaxis], (1, self.size - 4))])
            self.actions = np.hstack([self.actions, np.tile(indications[:,1][:, np.newaxis], (1, 1))])
            self.actions = np.hstack([self.actions, np.tile(indications[:,0][:, np.newaxis], (1, self.size - 4))])
            self.actions = np.hstack([self.actions, np.tile(indications[:,1][:, np.newaxis], (1, 1))])
        self.actions = np.hstack([self.actions, np.tile(indications[:,2][:, np.newaxis], (1, self.size - 3))])
        self.actions = np.hstack([self.actions, np.tile(indications[:,-1][:, np.newaxis], (1, self.size - 3))])
    
    def corner_move(self):
        actions_shape = self.actions.shape[1]
        self.rewards = np.hstack([self.rewards, self.env.move([[a] for a in self.actions[:,0]])])
        for i in range(1,actions_shape):
            rewards_t = self.env.move([[a] for a in self.actions[:,i]])
            self.rewards = np.hstack([self.rewards, rewards_t])    
        
    def head_position(self):
        state = self.env.to_state()
        self.head_index = tf.where(tf.reshape(state[:,:,:,3], (-1))).numpy()[0][0]
        self.head_pos = [self.head_index // (self.size), self.head_index % (self.size)]
        return self.head_pos

    def filter_before_one(self):
        filtered_rows = []
        for row in self.rewards:
            indices_of_1 = np.where(row == 1)[0]
            
            if len(indices_of_1) > 0:
                first_index_of_1 = indices_of_1[0]
                filtered_row = row[:first_index_of_1 + 1]
            else:
                filtered_row = row
            filtered_row = np.where(filtered_row == 0, -0.005, filtered_row)
            filtered_rows.append(filtered_row.tolist())
    
        return filtered_rows
        
        
    
    def main(self):
        
        self.closest_corner()
        self.move_to_closest_corner()
        self.find_indications()
        for _ in range(15):
            self.corner_move()
        return self.filter_before_one()

