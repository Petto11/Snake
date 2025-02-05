#!/usr/bin/env python
# coding: utf-8

# # Snake

# In[1]:


import environments_fully_observable 
#import environments_partially_observable
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
#from IPython.display import display, clear_output
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)
import time
import torch.optim as optim


# ## Environment definition

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


# In[253]:


class Snake():
    def __init__(self,gamma, epsilon, n , size, episodes, from_checkpoint = False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.rewards = []
        self.n = n
        self.size = size
        self.episodes = episodes
        self.agent = DQNModel_3((4, size, size), 4)
        #self.agent.build((None, 6, 6, 4))

        #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        #self.agent.compile(optimizer = optimizer, loss='mse')
        self.epsilon_decay = .99935
        self.epsilon_min = .1
        self.env = get_env(self.n, self.size)
        self.epsilon_values = []
        
        #if from_checkpoint:
        #    self.agent.load_weights('snake_v1')
        
        self.fruits_eaten = []
        self.eaten_bodies = []
        self.best_state_reward = []
        
    def pick_actions(self):
        
        random_numbers = np.random.rand(self.n)
        mask = random_numbers < self.epsilon
        state = self.env.to_state()
        mod_input = torch.tensor(snake.enrich(state), dtype = torch.float32)
        out = self.agent(torch.permute(torch.tensor(state, dtype = torch.float32), (0,3,1,2)), mod_input)
        
        out_copia = out.detach().numpy()
        mask_pred, pred_rewards = self.mask_illegal_actions()
        out_copia = np.where(mask_pred, out_copia, pred_rewards)
        
        actions = np.where(mask, np.random.choice([0,1,2,3], size = self.n), np.argmax(out_copia, axis = 1))
        actions = [[a] for a in actions]
        
        rewards = np.reshape(self.env.move(actions), (1,-1))
        terminated = (rewards < -0.05) | (rewards == 1)
        new_state = self.env.to_state()
        
        self.rewards.append(np.mean(rewards))
        self.fruits_eaten.append(np.sum(rewards == 0.5))
        self.eaten_bodies.append(np.sum(rewards == -0.2))
        
        return state, new_state, rewards, actions, terminated, out, out_copia
    
    def mask_illegal_actions(self):
        state = self.env.to_state()
        head_index = np.argwhere(state[:,:,:,3].reshape(-1,self.size * self.size) == 1)[:,1]
        state = state.reshape(self.n, self.size ** 2, 4)
        
        block_left = head_index - 1 #3
        block_right = head_index + 1 #1
        block_down = head_index + 6 #0
        block_up = head_index - 6 #2
        
        block_left = state[np.arange(self.n), block_left]
        block_right = state[np.arange(self.n), block_right]
        block_up = state[np.arange(self.n), block_up]
        block_down = state[np.arange(self.n), block_down]
        
        entity_reward_match = {0 : 0, 1 : 0.5, 2 : -0.2, -0.1 : -0.1}
        
        first_one_indices_l = np.argmax(block_left == 1, axis=1).astype(float)
        all_zeros_mask_l = np.all(block_left == 0, axis=1)
        first_one_indices_l[all_zeros_mask_l] = -.1
        block_left = np.array([entity_reward_match[a] for a in first_one_indices_l])
        
        first_one_indices_u = np.argmax(block_up == 1, axis=1).astype(float)
        all_zeros_mask_u = np.all(block_up == 0, axis=1)
        first_one_indices_u[all_zeros_mask_u] = -.1
        block_up = np.array([entity_reward_match[a] for a in first_one_indices_u])
        
        first_one_indices_d = np.argmax(block_down == 1, axis=1).astype(float)
        all_zeros_mask_d = np.all(block_down == 0, axis=1)
        first_one_indices_d[all_zeros_mask_d] = -.1
        block_down = np.array([entity_reward_match[a] for a in first_one_indices_d])
        
        first_one_indices_r = np.argmax(block_right == 1, axis=1).astype(float)
        all_zeros_mask_r = np.all(block_right == 0, axis=1)
        first_one_indices_r[all_zeros_mask_r] = -.1
        block_right = np.array([entity_reward_match[a] for a in first_one_indices_r])
        
        pred_rewards = np.array([block_down, block_right, block_up, block_left]).T
        
        mask = pred_rewards >= 0
        
        return mask,pred_rewards
        
    def train(self):
        start = time.time()
        temp = []
        
        plt.ion()  
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
        lines = {
            'average_reward': ax[0, 0].plot([], [], label='Average Reward')[0],
            'fruits_eaten': ax[0, 1].plot([], [], label='fruits_eaten')[0],
            'bodies_eaten': ax[1, 0].plot([], [], label='bodies_eaten')[0],
            'epsilon': ax[1, 1].plot([], [], label='Epsilon')[0]
        }
        
        ax[0, 0].set_xlim(0, self.episodes)
        ax[0, 0].set_ylim(-0.3, 0.7)
        ax[0, 0].set_xlabel('Episode')
        ax[0, 0].set_ylabel('Average Reward')
        ax[0, 0].set_title('Average Reward Over Time')
        ax[0, 0].legend()
        
        ax[0, 1].set_xlim(0, self.episodes)
        ax[0, 1].set_ylim(0, 300) 
        ax[0, 1].set_xlabel('Episode')
        ax[0, 1].set_ylabel('N° fruits')
        ax[0, 1].set_title('Fruits Eaten Over Time')
        ax[0, 1].legend()
    
        ax[1, 0].set_xlim(0, self.episodes)
        ax[1, 0].set_ylim(0, 150) 
        ax[1, 0].set_xlabel('Episode')
        ax[1, 0].set_ylabel('N° bodies')
        ax[1, 0].set_title('Bodies Eaten Over Time')
        ax[1, 0].legend()
        
        ax[1, 1].set_xlim(0, self.episodes)
        ax[1, 1].set_ylim(0, 1)  
        ax[1, 1].set_xlabel('Episode')
        ax[1, 1].set_ylabel('Epsilon')
        ax[1, 1].set_title('Epsilon Over Time')
        ax[1, 1].legend()
            
        for i in range(self.episodes):
            self.agent.train()
            optimizer.zero_grad()
            self.epsilon_values.append(self.epsilon)

            state, new_state, rewards, actions, terminated, out, out_copia = self.pick_actions()
            self.new_state = new_state
            with torch.no_grad():
                targets = np.where(terminated, rewards, rewards +  self.gamma * \
                          np.max(self.agent(torch.permute(torch.tensor(new_state, dtype = torch.float32), (0,3,1,2)), \
                          torch.tensor(snake.enrich(new_state), dtype = torch.float32)).detach().numpy(), axis = 1))
                
                
            q_target = out_copia
            q_target[np.arange(self.n), np.reshape(actions, (1,-1))] = targets
            self.out = out
            self.q_target = q_target
            loss = nn.MSELoss()(out, torch.tensor(q_target, dtype = torch.float32))
                

            if i % 10 == 0 and i != 0:
                lines['average_reward'].set_xdata(range(len(self.rewards)))
                lines['average_reward'].set_ydata(self.rewards)
                
                lines['fruits_eaten'].set_xdata(range(len(self.fruits_eaten))) 
                lines['fruits_eaten'].set_ydata(self.fruits_eaten)
                
                lines['bodies_eaten'].set_xdata(range(len(self.eaten_bodies)))  
                lines['bodies_eaten'].set_ydata(self.eaten_bodies)
                
                lines['epsilon'].set_xdata(range(len(self.epsilon_values)))  
                lines['epsilon'].set_ydata(self.epsilon_values)
                
                for k in range(2):
                    for j in range(2):
                        ax[k, j].set_xlim(0, i)
                        
                ax[0, 0].set_ylim(min(self.rewards) - 0.1, max(self.rewards) + 0.1)
                ax[0, 1].set_ylim(min(self.fruits_eaten) - 10, max(self.fruits_eaten) + 10)
                ax[1, 0].set_ylim(min(self.eaten_bodies) - 10, max(self.eaten_bodies) + 10)
                
                plt.tight_layout()
                clear_output(wait=True)
                display(fig)
        

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            #self.epsilon -= 1 / self.episodes
            loss.backward()
            optimizer.step()
            torch.save(self.agent.state_dict(), "snake_agent_3.pth")
            plt.ioff()
            plt.show()
        return self.rewards, self.eaten_bodies, self.fruits_eaten
    
    def enrich(self, state):
        head_indexes = np.argwhere(state[:,:,:,3].reshape(-1,36) == 1)[:,1]
        fruit_indexes = np.argwhere(state[:,:,:,1].reshape(-1,36) == 1)[:,1]
        
        size = state.shape[-2]
        hp_row = head_indexes // size
        hp_col = head_indexes % size
        
        fp_row = fruit_indexes // size
        fp_col = fruit_indexes % size
        
        hp = np.array([hp_row, hp_col]).T
        fp = np.array([fp_row, fp_col]).T

        temp = hp - fp
        mh_distances = np.sum(np.abs(temp), axis=1)
        sugg_actions = np.array([temp[:,0] < 0, temp[:,1] < 0 , temp[:,0] > 0, temp[:,1] > 0], dtype = np.int32)
        
        snake_length = np.sum(state[:,:,:,2].reshape(-1,36) == 1, axis = 1)
        
        #print(mh_distances.shape, sugg_actions.shape, snake_length.shape)
        return np.vstack([mh_distances, sugg_actions, snake_length]).astype(np.float32).T
    
    def filter_rewards(self, rewards):
        filtered_rows = []
        for row in rewards:
            indices_of_1 = np.where(row == 1)[0]
            
            if len(indices_of_1) > 0:
                first_index_of_1 = indices_of_1[0]
                filtered_row = row[:first_index_of_1 + 1]
            else:
                filtered_row = row
            filtered_row = np.where(filtered_row == 0, -0.005, filtered_row)
            
            if filtered_row[-1] == 1. and np.all(filtered_row > -0.1):
                filtered_rows.append(filtered_row.tolist())
        
        return filtered_rows
            
    def test(self):
        env = get_env(1000,6)
        state = env.to_state()
        loaded_model = DQNModel_3((4,6,6), 4)
        loaded_model.load_state_dict(torch.load('snake_agent_3.pth'))
        
        mod_inp = torch.tensor(self.enrich(state), dtype = torch.float32)
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
        return self.filter_rewards(rewards)
    


# In[209]:


import torch
import torch.nn as nn
import torch.nn.functional as F

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





snake = Snake(0.99, 1, 1000, 6, 5000)  




optimizer = optim.Adam(snake.agent.parameters(), lr = 1e-4)


# In[255]:


#agent_trained = snake.train()

