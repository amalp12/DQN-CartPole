
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from collections import deque
import random
import matplotlib.pyplot as plt

EPISODES = 200  # number of episodes
MAX_EXPLORATION =1  # e-greedy threshold start value
MIN_EXPLORATION = 0.2  # e-greedy threshold end value
EXPLORATION_DECAY = 1/200  # e-greedy threshold decay
GAMMA = 0.93  # Q-learning discount factor
LEARNING_RATE = 10e-4  # NN optimizer learning rate
HIDDEN_SIZE = 200  # NN hidden layer size
BATCH_SIZE = 40  # Q-learning batch size
EXPLORATION = 1
EXPLORATION_DECAY =0.0002
TRAIN_EPISODES = 10000

"""
Any Class that implements a neural network from pytorch have to inherit from the nn.Module class
PyTorch refers to fully connected layers as Linear layers.

"""

def getMedian( l):
    l = sorted(l)
    return l[int(len(l)/2)]
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
        
        
class Memory():
    def __init__(self, limit):
        self.memory = deque()    
        self.max_length = limit
        self.counter = 0  
    def append(self, val):
        if (self.get_length()==self.max_length):
            self.delete_left()

        self.memory.append(val)
        
    def prepend(self,val):
        if (self.get_length()==self.max_length):
            self.delete_left()
        self.memory.prepend(val)

    def delete_right(self):
        return self.memory.pop()

    def delete_left(self):
        return self.memory.popleft()
    def get_length(self):
        return len(self.memory)

    
class Agent():

    def __init__(self,env, primaryN, targetN, gamma=.99, eplison = .3,eplison_decay=0.3 , state_dimention=4 , max_length=1000, weights = None, frames =5000 ):


         # Initializing all the constnt variables   
        self.state_dimention = state_dimention
        self.weights = weights
        self.env = env 
        self.frames = frames
        self.batch_size = BATCH_SIZE
        self.update = False
        self.train_cnt = 0
        self.discount_rate = GAMMA #(gamma)
        self.exploration_rate = EXPLORATION
        self.max_exploration_rate = MAX_EXPLORATION
        self.min_exploration_rate = MIN_EXPLORATION
        self.learning_rate =LEARNING_RATE 
        self.exploration_decay = EXPLORATION_DECAY# eplison 
       
        self.no_of_episodes = TRAIN_EPISODES
        # Initializing our network
        self.Q = DQN(self.learning_rate)
        self.target_net = DQN(self.learning_rate)
        self.target_net.load_state_dict(torch.load("target_net.dat"))
        self.max_score = 100
        self.max_median = 12
        # action  0 is left and 1 is right  
        self.action_space = [0,1]   
        # Creating memory elements
        self.mem_length = max_length
        self.mem_counter = 0
        self.state_memory=  Memory(max_length)
        self.next_state_memory=  Memory(max_length)
        self.action_memory = Memory(max_length)
        self.terminal_memory = Memory(max_length)
        self.reward_memory = Memory(max_length)
 

    def store(self, observation, action,reward ,next_observation, done):
        """
            Stores all the results of an iteration in the memory
        """
        #index = self.mem_counter % self.mem_length
        self.state_memory.append(observation)
        self.next_state_memory.append(next_observation)
        self.action_memory.append(action)
        self.terminal_memory.append(done)
        self.reward_memory.append(reward)
        self.mem_counter+=1

    def choose(self, observation):
        """
        Choose a random action or choose throught the neural network depending on the explaoration constant which decays with the number of iterations
    
        """
        observation = torch.tensor([observation], dtype= torch.float32)
        if self.exploration_rate> np.random.random():
            self.UpdateExploration()
            return torch.tensor(self.RandomMove(), dtype = torch.float32)
        else :
            self.UpdateExploration()
            return self.NetworkMove(observation)



    def NetworkMove(self, observation):

        assert type(observation) == torch.Tensor
        action = self.Q.forward(observation, Q_val=False)
        # torch.argmax(input, dim, keepdim=False) → LongTensor Returns the indices of the maximum values of a tensor across a dimension.
        #action =  torch.argmax(torch.tensor(action)).item()  # returns the index of the maximum value which will always be 0 or 1
        return action

        
    """
    here are several different loss functions under the nn package . 
    A simple loss is: nn.MSELoss which computes the mean-squared error between the input and the target.
    """
    
    def Save(self,primaryN,t):
        torch.save(primaryN.state_dict(), f"{t}-best.dat")
    
    def RandomMove(self):
        action = self.env.action_space.sample()
        return action
    """
    There is two ways to learn:
        one way is to play a bunch of random games until out memory fills up and then start learning (what we are going to do )

        the other way is to play and game and learn from that and iterate
    """
    def Rlist(self, a):
        l= [i for i in range(a) ]
        return l
   

    def Learn(self):
        
        # make a list with all possibel indexes
        l = self.state_memory.get_length()
        
        if l<self.batch_size:
            return
        li = self.Rlist(l)
        # return a number in the range of  l (memory size)  muliple calls will never return the same number
        #batch = np.random.choice(li,self.batch_size,  replace=False)
        batch =random.sample(li, self.batch_size)
        # getting the batch's variables
        b_state = torch.tensor([self.state_memory.memory[i] for i in batch], dtype= torch.float32, requires_grad=False)
        b_state_new = torch.tensor([self.next_state_memory.memory[i] for i in batch], dtype= torch.float32, requires_grad=False)
        b_reward = torch.tensor([self.reward_memory.memory[i] for i in batch], dtype= torch.float32, requires_grad=False)
        b_terminal = torch.tensor([self.terminal_memory.memory[i] for i in batch], dtype= torch.float32, requires_grad=False)

        b_action =  [self.action_memory.memory[i] for i in batch]

        Q_vals =[]# 
        Target_vals = torch.empty(size = (self.batch_size,2), dtype =torch.float32, requires_grad=False)
        #values = torch.stack(list(values),1)
        for i in range(self.batch_size):
            Q = self.Q.forward(b_state[i], Q_val=True)
            Q_next =self.target_net.forward(b_state_new[i], Q_val= True).detach()
            #Q_target = Q + self.learning_rate * (b_reward[i]* self.discount_rate  * Q_next -Q)
            Q_target = b_reward[i] + self.discount_rate * Q_next
            if b_terminal[i]:
                Q_target = b_reward[i]
               
            Q_vals.append(Q)
            Target_vals[i] = Q_target
           
        Q_vals = torch.stack(Q_vals,0)
       
        
        self.Q.optimizer.zero_grad()
        loss = self.Q.loss(Q_vals, Target_vals)
        if self.train_cnt%10==0:
            print("Loss: " ,loss.item())
        loss.backward()
        self.Q.optimizer.step()

        
        

    def Play(self):
        episode_rewards = []
        
        avg =[]
        observation = env.reset()
        for episode in range(TRAIN_EPISODES):
            
            
            current_reward = 0
            for t in range(501):
                #self.Learn()
                #env.render()
                #print(observation)
                # Action is to be chosen by the agent
                action = self.choose(observation)
                
                
                next_observation, reward, done, info = self.env.step(int(action))
                current_reward += reward
                if not done:
                    self.store(observation, action,reward ,next_observation, done)
                    observation = next_observation
                if done:
                    #avg.append(t)
                    episode_rewards.append(t)
                    self.store(observation, action,-100 ,next_observation, done)
                    print(f"DQN Training Episode finished ({self.train_cnt} of {TRAIN_EPISODES}).. Total Reward is : {t+1}.")
                    observation = env.reset()
                    
                    self.train_cnt+=1
                    
                    if t>100:
                        self.Save(self.Q,self.train_cnt)
                    elif t>=499:
                        self.Save(self.Q)    
                    
                    break
            
        
                    
                
            avg_score = t# sum(avg)/len(avg)
            #median_score = getMedian(avg)
            #print("Average Score: ", avg_score, "Median Score: ",median_score)
            if avg_score>self.max_score :
            
            #if self.train_cnt%25==0 :
                self.target_net= deepcopy(self.Q)
                #self.Save(self.Q, self.train_cnt)
                self.max_score = avg_score
                #self.max_median = median_score
            self.Learn()
            
        self.plot_graph(episode_rewards)
    def UpdateExploration(self):
        if(self.exploration_rate >self.min_exploration_rate and self.exploration_rate<= self.max_exploration_rate ):
            self.exploration_rate -= self.exploration_rate * self.exploration_decay
        else:
            self.exploration_rate = self.min_exploration_rate   

    def plot_graph(self,episode_rewards):
        plt.plot(np.arange(len(episode_rewards)), episode_rewards)#,s=2)
        plt.title("Total reward per episode (episodic)")
        plt.ylabel("reward")
        plt.xlabel("episode")
        plt.show()


def Save2Target(primaryN):
    return deepcopy(primaryN)
def UpdatePrimary( TargetN):
    return deepcopy(targetNet)

env = gym.make('CartPole-v1')
network = DQN(0.001)
targetNet = DQN(0.01)

agent = Agent(env, network, targetNet)
y = [1,2,3,4]
y= torch.tensor(y, dtype= torch.float)
x= torch.randn(4, dtype= torch.float)
y= torch.randn(2, dtype= torch.float)
z= torch.randn(2, dtype= torch.float)
g = torch.randn(4, dtype= torch.float)
f =network.forward(x)

agent.Play()



for i_episode in range(20):
    
    observation = env.reset()
    
    for t in range(500):
        env.render()
        #print(observation)
        # Action is to be chosen by the agent
        #action = env.action_space.sample()
        action = agent.Q.forward(torch.tensor([observation], dtype= torch.float32), Q_val=False)
        observation, reward, done, info = env.step(int(action))
        if done:
            print("Game Episode finished after {} timesteps".format(t+1))
            break
env.close()


