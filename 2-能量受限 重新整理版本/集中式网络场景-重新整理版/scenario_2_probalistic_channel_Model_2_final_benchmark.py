# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:13:23 2021

@author: xxx

This file is used to simulate the scenario that
we use Hungarian to find the optimal configuration

- 该代码用来求得 benchmark 性能
- 调整不同 U 的大小表示不同用户数量
- 该代码的含义是中心节点已知信道统计信息（信道噪声）从而用Hungarian算法按照直接求得最优分配方案（没有statistical learning 的过程）
- 因此选择较大的 budget 多次取平均可得 benchmark
"""
 

import numpy as np
import random 
import matplotlib.pyplot as plt
import time
import math
import copy
import Auctioning as Auction
from hungarian import Hungarian
from random import shuffle
# 生成随机数，然后求和，然后除法，得到向量
# auction algorithm


# In[0] 参数设置
alpha_0 = 1/30
alpha_1 = 3
budget_list = [   0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.4, 4.8  ] # J
budget_list = [   0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.4 ] # J
budget_list = [  2  ] # J
# budget_list = [ 200, 500, 1000, 3000, 8000, 10000, 20000   ]

U = 5 # 次要用户的数量
N = 20 # 设置好信道数量    
Tx_antenna_Num = 4
c_sensing = 2.5e-04 # J # 表示信道分配和探测的能量
transmission_rate = 1 # Mbits/s/Hz
duration = 95e-03 # s
bandwidth = 1 # MHz
transmission_distance = [ 121.81086664, 127.29765091, 136.53161459,  141.98326578,  146.85736952,  151.10445988,  156.91303438,   161.70360273, 159.16186613, 118.98220862]
# Channel_Noise = -174 # dBm/Hz
# transform to practical
# Channel_Noise = -104 # dBm
Channel_Noise_vector = [ -94 - 0.5 * i for i in range(20) ]
SU_threshold = np.zeros(U).tolist() # 设置次要用户能量阈值

Monte_Carlo_Num = 30
 
# In[0] 
def create_matrices(number_of_agents):
    """
    Create matrices of agents and their values for varying objects
    """
    matrix = []
    agent = 0
    while agent != number_of_agents:
        temp_matrix = []
        for prices in range(0,number_of_agents):
            temp_matrix.append(round(random.uniform(1, 15),0))
        matrix.append(temp_matrix)
        agent += 1
    prices_list = []
    for prices in range(0,number_of_agents):
        prices_list.append(round(random.uniform(1, 10),0))
    return matrix, prices_list

def assign_values(number_of_agents):
    """
    Assign the values to each of the agents
    """
    temp_list = []
    agents = []
    for i in range(0,number_of_agents):
        temp_list.append(i)
    shuffle(temp_list)
    for p in range(0,number_of_agents):
        agents.append([temp_list[p],0])
    return agents

def get_best_option(agent_number, matrix, costs):
    """
    Find the best option for the given agent
    """
    diff_list = []
    for i in range(0,len(costs)):
        diff_list.append(matrix[agent_number][i] - costs[i])
    highest_val = max(diff_list)
    ind_highest_val = diff_list.index(highest_val)
    diff_list[ind_highest_val] = -1000000000
    second_highest_val = max(diff_list)
    ind_second_highest_val = diff_list.index(max(diff_list))
    return highest_val, ind_highest_val, second_highest_val, ind_second_highest_val

def find_index_of_val(mylist, value):
    """
    Find index of given value in list
    """
    l = [i[0] for i in mylist]
    if value in l:
        return l.index(value) 
    else:
        return []
    # 如果有值，则进行交换，如果没有值，则进行满意

def check_happiness(agent_matrix,payoff_matrix,cost_list):
    """
    Check if agents are happy with their current situation
    """
    for i in range(0,len(agent_matrix)):
        if agent_matrix[i][1] == 0:
            high,ind_high,sec_high,sec_ind_high = get_best_option(i,payoff_matrix,cost_list)
            if ind_high == agent_matrix[i][0]:
                agent_matrix[i][1] = 1

def sophisticated_auction(epsilon, agents, payoff_matrix, cost_list):
    """
    Running the sophisticated version of the auction algorithm where we force agents to increase bids by epsilon each time they bid
    """
    item_Num = 0
    while sum(n for _, n in agents) != len(agents):
        for i in range(0,len(agents)):
            check_happiness(agents, payoff_matrix, cost_list)
            item_Num = item_Num + 1
            if agents[i][1] == 0:
                high,ind_high,sec_high,sec_ind_high = get_best_option(i, payoff_matrix, cost_list)
                switch_index = find_index_of_val( agents, int(ind_high) )
                if switch_index == []: # 如果为 switch_index 没有用户占用，则不需要交换
                    agents[i][1] = 1
                    agents[i][0] = ind_high
                else:
                    agents[switch_index][1] = 0
                    agents[i][1] = 1 
                    agents[switch_index][0] = agents[i][0]
                    agents[i][0] = ind_high
                    cost_list[ind_high] = cost_list[ind_high] + abs(((payoff_matrix[i][ind_high] - cost_list[ind_high]) - (payoff_matrix[i][sec_ind_high] - cost_list[sec_ind_high]))) + epsilon
    return item_Num
# '''

# In[0] 定义函数和类

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 


class algo:
    def __init__(self, network_info):
        self.network_info = network_info
 
        
        # some other parameters
        self.estimated_theta = 0
        self.accumulative_reward = 0
        self.accumulative_cost = 0
        self.mean_reward = 0
        self.mean_cost = 0
        self.chosen_time = 0
        self.index = 0
        
    def compute_index(self, instant_reward, time, instant_cost):
        self.accumulative_reward = self.accumulative_reward + instant_reward
        self.accumulative_cost = self.accumulative_cost + instant_cost
        self.chosen_time = self.chosen_time + 1
        self.mean_reward = (self.accumulative_reward)/(self.chosen_time)
        self.mean_cost = (self.accumulative_cost)/(self.chosen_time)
        

    
    def compute_index_update(self, time):
        self.index = alpha_0 * self.mean_reward/(self.mean_cost) +  np.sqrt( alpha_1 * np.log(time) / (self.chosen_time) )
        return self.index
        
    def initial(self):
        self.estimated_theta = 0
        self.accumulative_reward = 0
        self.mean_reward = 0
        self.chosen_time = 0
        self.index = 0
        self.index_classic_MAB = 0
        self.accumulative_cost = 0
        self.mean_cost = 0        
 
class network_info:
    def __init__(self, Tx_antenna_Num, c_sensing, transmission_rate, duration, bandwidth, transmission_distance, Channel_Noise):
        self.c_sensing = c_sensing
        self.transmission_rate = transmission_rate # transmission rate 
        self.duration = duration
        self.throughput = transmission_rate * duration
        self.transmission_distance = transmission_distance
        self.Tx_antenna_Num = Tx_antenna_Num
        self.Bandwidth = bandwidth
        self.Channel_Noise = Channel_Noise
        self.PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/(self.transmission_distance**(3.76)) )

    def PathLoss_Model(self):
        Channel_parameter_vector = []
        for i in range(self.Tx_antenna_Num):
            Channel_parameter_vector.append( self.PathLoss_Model_Parameter * 1/np.sqrt(2) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) )
            # 其他场景直接求信噪比比较好 或者信道系数 Rician
            # Channel_parameter_vector.append( self.PathLoss_Model_Parameter * ( np.sqrt( K_factor/(K_factor+1) ) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) + np.sqrt( 1/(K_factor+1) ) * np.random.normal()   )  )
            
        Channel_parameter_norm = np.linalg.norm( Channel_parameter_vector )
        self.Channel_gain_noise_ratio = Channel_parameter_norm**2/self.Channel_Noise

    def Energy_minimum_(self): # definition of the network cost
        self.Power_minimum = ( 2**(self.transmission_rate/self.Bandwidth) - 1 )/self.Channel_gain_noise_ratio
        self.Energy_minimum = self.Power_minimum * self.duration + self.c_sensing
        return self.Power_minimum, self.Energy_minimum     
 
# In[0]

Channel_Noise = [ 10**(Channel_Noise_vector[i]/10)/(10**3) for i in range(len(Channel_Noise_vector)) ] 

Channel_Noise_Set = [ Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise ]


network_SU_set = []
for u in range(U):
    network_SU_set.append([])
    for i in range(N):
        network_SU_set[u].append( network_info( Tx_antenna_Num, c_sensing, transmission_rate, duration, bandwidth, transmission_distance[u], Channel_Noise_Set[u][i]   ) )
 
    
# This is used for the static Hungarian Algorithm 
Channel_gain_noise_ratio_Matrix = np.zeros( (U,N) )
for u in range(U):
    for i in range(N):
        Channel_gain_noise_ratio_Matrix[u][i] = network_SU_set[u][i].PathLoss_Model_Parameter/network_SU_set[u][i].Channel_Noise/1e6
 
# find the instantaneous transmit power and energy
Power_Requirement = np.zeros((U,N))
Energy_Requirement = np.zeros((U,N))
 
# find channel gain any time
for u in range(U):
    for i in range(N):
        network_SU_set[u][i].PathLoss_Model()
        Power_Requirement[u][i], Energy_Requirement[u][i] = network_SU_set[u][i].Energy_minimum_()
 
policy_SU_set = []
for u in range(U):
    policy_SU_set.append([])
    for i in range(N):
        policy_SU_set[u].append( algo(network_SU_set[u][i]) )

# In[1]

epsilon_bertsekas = 0.001
average_reward_set = np.zeros(len(budget_list))
sum_reward_set = np.zeros(len(budget_list))
reward_difference_set = np.zeros(len(budget_list))
normalized_reward_difference_set = np.zeros(len(budget_list))
 
for index_, budget in enumerate(budget_list):
    
    simulation_Num_ = math.ceil(budget * 3000)
    reward_monte_set = np.zeros( (Monte_Carlo_Num, simulation_Num_) )
    reward_monte_mean = np.zeros(simulation_Num_)
    end_time_budget = np.zeros(Monte_Carlo_Num)    
    
    end_time_budget_SU_set = np.zeros((U,Monte_Carlo_Num))
    
    sum_reward = np.zeros(Monte_Carlo_Num)
    reward_difference = np.zeros(Monte_Carlo_Num)
    normalized_reward_difference = np.zeros(Monte_Carlo_Num)
    a_Num_chosen = np.zeros((1,N)).tolist()[0]
    print("this is the", budget, "budget")
    
    for monte in range(Monte_Carlo_Num):
        # print("This is the", monte, "simulation")
        
        for u in range(U):
            for i in range(N):
                policy_SU_set[u][i].initial()
    
        reward = np.zeros(N)
        index = np.zeros(N)
        chosen_arm = []
        
        reward_U_N = np.zeros((U,N))
        index_U_N = np.zeros((U,N))
        cost_U_N = np.zeros((U,N))
        
        reward_monte = np.zeros(simulation_Num_)
        reward_monte[0] = 0.0001
        # regret_monte = np.zeros(simulation_Num_)
        
        budget_residual = np.ones(U) * budget
    
        Power_Requirement = np.zeros((U,N))
        Energy_Requirement = np.zeros((U,N))
        
        U_temp = U
        Channel_gain_noise_ratio_Matrix_temp = copy.deepcopy(Channel_gain_noise_ratio_Matrix)
        
        for num in range(simulation_Num_):
            
            for u in range(U_temp):
                for i in range(N):
                    network_SU_set[u][i].PathLoss_Model()
                    # print('the user', u, 'on channel', i, 'channel gain is', network_SU_set[u][i].Channel_gain_noise_ratio )
                    Power_Requirement_temp, Energy_Requirement_temp = network_SU_set[u][i].Energy_minimum_()
                    cost_U_N[u][i] = copy.deepcopy( Energy_Requirement_temp )
                    # print('the user', u, 'on channel', i, 'minimum energy requirement is', Energy_Requirement_temp )
                    reward_U_N[u][i] = copy.deepcopy( network_SU_set[u][i].throughput )  

 
            temp_reward = 0
            temp_cost = 0
            
            temp_reward_SU = np.zeros(U)
            temp_cost_SU = np.zeros(U)
            
            if num < 1:
                for u in range(U):
                    for i in range(N):
                        policy_SU_set[u][i].compute_index( reward_U_N[u][i], num, cost_U_N[u][i] )
                
            else:
                    
                for u in range(U):
                    for i in range(N):
                        index_U_N[u][i] = policy_SU_set[u][i].compute_index_update( num )

                for u in range(U):
                    if budget_residual[u] < SU_threshold[u]:
                        index_U_N[u] = copy.deepcopy( np.zeros(N).tolist() )   
                        Channel_gain_noise_ratio_Matrix_temp[u] = copy.deepcopy( np.zeros(N).tolist() )  
                        
                # print(index_U_N)


                # """ Static Benchamrk # 
                # hungarian 算法模块 in known environment
                hungarian = Hungarian()
                hungarian.calculate(Channel_gain_noise_ratio_Matrix_temp.tolist(), is_profit_matrix=True)
                allocation_relationship = hungarian.get_results()
                Iter_Num = 0
                # """

                ''' Static auction Benchmark
                agents = assign_values(U_temp)
                cost_list = np.zeros(N).tolist()
                Iter_Num = 0
                Iter_Num = sophisticated_auction(epsilon_bertsekas, agents, Channel_gain_noise_ratio_Matrix_temp, cost_list)
                allocation_relationship_origin = []
                for u in range(U_temp):
                    allocation_relationship_origin.append( [ u, agents[u][0] ])  
                allocation_relationship = copy.deepcopy(allocation_relationship_origin)
                '''
 
                temp_SU = 0
                temp_channel = 0
                for j in range( len( allocation_relationship ) ):
                    temp_SU = allocation_relationship[j][0] # 表示第几个次要用户 
                    temp_channel = allocation_relationship[j][1] # 表示第几个信道
                    policy_SU_set[temp_SU][temp_channel].compute_index( reward_U_N[temp_SU][temp_channel], num, cost_U_N[temp_SU][temp_channel] )
                    if budget_residual[temp_SU] < SU_threshold[temp_SU]:
                        reward_U_N[temp_SU][temp_channel] = copy.deepcopy(0)
                        cost_U_N[temp_SU][temp_channel] = copy.deepcopy(0)
                        policy_SU_set[temp_SU][temp_channel].chosen_time = policy_SU_set[temp_SU][temp_channel].chosen_time - 1  # 拉动的次数减少1
                    temp_reward_SU[temp_SU] += copy.deepcopy(reward_U_N[temp_SU][temp_channel])
                    temp_cost_SU[temp_SU] += copy.deepcopy(cost_U_N[temp_SU][temp_channel])
                    budget_residual[temp_SU] = budget_residual[temp_SU] - temp_cost_SU[temp_SU]  #- Iter_Num * c_a # 因为加能量了和 benchmark 就不同了， 仅比较平均收益
                # print(budget_residual)
                
                temp_reward = sum(temp_reward_SU)
                temp_cost = sum(temp_cost_SU)
            
            if num>0:
                reward_monte[num] = temp_reward
            
            if reward_monte[num] == 0:
                break
             
        end_time_budget[monte] = num
        
        # for i in range(N):
        #     print(policy[i].chosen_time)
        # print("This is the space")
        
        reward_monte_set[monte] = reward_monte
        
        chosen_time_matrix = []
        for u in range(U):
            chosen_time_matrix.append([])
            for i in range(N):
                chosen_time_matrix[u].append( policy_SU_set[u][i].chosen_time )
        print(chosen_time_matrix)
        print('---'*20)
            
    reward_monte_mean = sum(reward_monte_set)/Monte_Carlo_Num
    average_reward_set[index_] = sum(reward_monte_mean)/budget
    sum_reward_set[index_] = sum(reward_monte_mean)
   
chosen_time_matrix = []
for u in range(U):
    chosen_time_matrix.append([])
    for i in range(N):
        chosen_time_matrix[u].append( policy_SU_set[u][i].chosen_time )
 
# sum(optimal_sum_reward_cost_ratio_SU_set)

# In[0] 能量效率
plt.plot(budget_list, average_reward_set )
 



























 

