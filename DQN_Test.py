import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from collections import deque
import random
HIDDEN_SIZE = 200  # NN hidden layer size
class DQN(nn.Module):
    def __init__(self, lr):
        super().__init__()
       
        
        
        self.learning_rate = lr

        # PyTorch refers to fully connected layers as Linear layers.

        # Creating PrimaryNetwork layers
        self.Layer_1= nn.Linear(in_features= 4, out_features = int(HIDDEN_SIZE))

    
        self.Layer_2 = nn.Linear(in_features= int(HIDDEN_SIZE), out_features = HIDDEN_SIZE)

        self.Layer_3 = nn.Linear(in_features= HIDDEN_SIZE, out_features = int(HIDDEN_SIZE))

        self.Layer_4 = nn.Linear(in_features= HIDDEN_SIZE, out_features = int(HIDDEN_SIZE))

        self.Layer_5 = nn.Linear(in_features= int(HIDDEN_SIZE), out_features = 2)

        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)

        self.loss = nn.MSELoss()


    """
        All pytorch nn require forward method to be defined. 
        This will implement the forward pass to the network.
    """
    def forward(self, state, Q_val=True):
        assert type(state) == torch.Tensor
        t = F.relu(self.Layer_1(state) )
        t = F.relu(self.Layer_2(t))
        t = F.relu(self.Layer_3(t))
        t = F.relu(self.Layer_4(t))
        # To get actions we do not need to apply the relu funtions we need the raw output
        action = self.Layer_5(t)
        index =  torch.argmax(action)  # returns the index of the maximum value which will always be 0 or 1
        if Q_val:
            return action
        return index
   

env = gym.make('CartPole-v1')
#env = env.unwrapped
network = DQN(0.01)
targetNet = DQN(0.01)


targetNet.load_state_dict(torch.load("dqn_win.dat"))



avg =[]
for i_episode in range(20):
    
    observation = env.reset()
    for t in range(500):
        env.render()
        
        action = targetNet.forward(torch.tensor([observation], dtype= torch.float32),Q_val=False)
        observation, reward, done, info = env.step(int(action))
        #print("Not Done!")
        if done:
            avg.append(t+1)
            print("Game Episode finished after {} timesteps".format(t+1))
            break

print("Average Score: ",sum(avg)/len(avg))
env.close()


