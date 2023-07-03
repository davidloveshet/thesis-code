# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:13:23 2021

@author: xxx

This file is used to simulate the centralized network with energy constraints   
说明：
1. 可通过设置不同的用户数量 U =2,3,4,5,6 得到不同用户数量下的仿真结果
 
"""

# 备注：求拉动次数的时候，需要设置为 reward 和 cost 随机值


import numpy as np
import random 
import matplotlib.pyplot as plt
import time
import math
import copy
from hungarian import Hungarian
from random import shuffle
# 生成随机数，然后求和，然后除法，得到向量
# auction algorithm

start = time.time() # 记录时间
 
# In[0] 参数设置
U = 6 # 次要用户的数量
N = 10 # 设置信道数量
alpha_0 = 1/20
alpha_1 = 1/2
budget_list = [ 2, 5, 10, 20,  40,  60,  80,  100, 150, 200 , 250  ] 
# budget_list = [ 2, 5, 10, 20,  40,  60,  80,  100 ] 
Monte_Carlo_Num = 200
 
# In[1] 编写 Bertsekas 拍卖算法，该算法可用于检验 Hungarian 算法的准确性。
def create_matrices(number_of_agents):
    """
    创建矩阵
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
    赋值给每个次要用户
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
    为次要用户求得最好的信道
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
 
    l = [i[0] for i in mylist]
    if value in l:
        return l.index(value) 
    else:
        return []
    # 如果有值，则进行交换，如果没有值，则进行满意

def check_happiness(agent_matrix,payoff_matrix,cost_list):
    """
    检查各次要用户满意度
    """
    for i in range(0,len(agent_matrix)):
        if agent_matrix[i][1] == 0:
            high,ind_high,sec_high,sec_ind_high = get_best_option(i,payoff_matrix,cost_list)
            if ind_high == agent_matrix[i][0]:
                agent_matrix[i][1] = 1

def sophisticated_auction(epsilon, agents, payoff_matrix, cost_list):
    """
    拍卖过程
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
 
   
# In[1]
def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 


class algo:
    def __init__(self, channel_info):
        self.channel_info = channel_info
        self.cost = self.channel_info.cost 
        
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
        
        # self.index = self.mean_reward/self.mean_cost + np.sqrt(np.log(time) / (self.chosen_time))
        # self.index_classic_MAB = self.mean_reward + np.sqrt(np.log(time) / (self.chosen_time))
        # self.D_ONES = self.mean_reward + np.sqrt(np.log(time) / (self.chosen_time)) - (   self.cost - 1.5 * np.sqrt(np.log(time) / (self.chosen_time))  )
        # self.ONES = self.mean_reward - self.cost * ( self.mean_reward/self.cost - np.sqrt(np.log(time) / (self.chosen_time))  )+  np.sqrt(np.log(time) / (self.chosen_time))
        # return self.index

    
    def compute_index_update(self, time):
        self.index = alpha_0 * self.mean_reward/(self.mean_cost) +  np.sqrt( alpha_1 * np.log(time) / (self.chosen_time) )
        # self.index = alpha_0 * self.mean_reward/(self.mean_cost) +  1/(self.mean_cost) * np.sqrt( alpha_1 * np.log(time) / (self.chosen_time) )
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
 
class channel_info:
    def __init__(self, p, c_s, c_a, mu):
        self.p = p
        self.c_s = c_s
        self.c_a = c_a
        self.mu = mu
        self.mean_reward = self.mu
        self.cost = c_s + np.dot(self.p,self.c_a)
        
    def cost_random(self):
        self.cost_random_ = self.c_s + random_pick(self.c_a, self.p)
        return self.cost_random_
    
# In[2] 信道分配和探测的耗能 量纲为 J 
c_1_s = 2.5e-04
c_2_s = 2.5e-04
c_3_s = 2.5e-04
c_4_s = 2.5e-04
c_5_s = 2.5e-04
c_6_s = 2.5e-04
c_7_s = 2.5e-04
c_8_s = 2.5e-04
c_9_s = 2.5e-04
c_10_s = 2.5e-04

c_channel_sensing_cost = [ c_1_s, c_2_s, c_3_s, c_4_s, c_5_s, c_6_s, c_7_s, c_8_s, c_9_s, c_10_s ]

# In[3] 信道接入能耗
c_1_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ]  # 能耗最大为 0.019 J，来源于发射功率 200 mW * 95 ms = 19000e-06 J
c_2_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_3_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_4_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_5_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_6_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_7_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_8_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_9_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_10_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_11_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 
c_12_a = [ 1/4 * 1.9e-02, 1/2 * 1.9e-02, 3/4 * 1.9e-02, 1.9e-02 ] 

c_channel_cost = [ c_1_a, c_2_a, c_3_a, c_4_a, c_5_a, c_6_a, c_7_a, c_8_a, c_9_a, c_10_a, c_11_a, c_12_a ]
 
# In[4] 固定传输速率 * 传输时长
R_u_1 = 1.2 * 95e-03 # 1.2 Mbps 表示传输速率， 95e-03表示传输时长 
R_u_2 = 1.2 * 95e-03 #  
R_u_3 = 1.2 * 95e-03 # M/s
R_u_4 = 1.2 * 95e-03 # M/s
R_u_5 = 1.2 * 95e-03 # M/s
R_u_6 = 1.2 * 95e-03 # M/s
R_u_7 = 1.2 * 95e-03 # M/s
R_u_8 = 1.2 * 95e-03 # M/s
R_u_9 = 1.2 * 95e-03 # M/s
R_u_10 = 1.2 * 95e-03 # M/s

# 吞吐量
mu_1 = R_u_1  
mu_2 = R_u_2  
mu_3 = R_u_3  
mu_4 = R_u_4   
mu_5 = R_u_5 
mu_6 = R_u_6   
mu_7 = R_u_7   
mu_8 = R_u_8   
mu_9 = R_u_9   
mu_10 = R_u_10   

mu_SU_1_set = [ mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9, mu_10 ]

# In[5] 为不同次要用户产生不同的发射概率等级，随机生产存储
p_SU_set = []
 
p_SU_1 = [
       [0.06174471, 0.35042525, 0.37971375, 0.20811628],
       [0.26520542, 0.35568782, 0.19987372, 0.17923303],
       [0.44506214, 0.20935051, 0.02776782, 0.31781953],
       [0.57177777, 0.23385294, 0.11798038, 0.0763889 ],
       [0.13569405, 0.33520816, 0.07611595, 0.45298185],
       [0.02649764, 0.15105907, 0.48289203, 0.33955126],
       [0.15169642, 0.36479579, 0.48085015, 0.00265765],
       [0.2470291 , 0.2880954 , 0.25683072, 0.20804479],
       [0.21909338, 0.42521685, 0.12076967, 0.2349201 ],
       [0.31589767, 0.13476606, 0.29826421, 0.25107206]
       ]

p_SU_2 = [
       [0.18338951, 0.17312694, 0.09531618, 0.54816736],
       [0.0704549 , 0.17784292, 0.23768699, 0.51401519],
       [0.24578482, 0.03913634, 0.30889377, 0.40618506],
       [0.19519355, 0.33128528, 0.45194203, 0.02157914],
       [0.45190821, 0.37867284, 0.06390243, 0.10551653],
       [0.03615571, 0.32336649, 0.32794074, 0.31253707],
       [0.34886073, 0.15789666, 0.13347781, 0.3597648 ],
       [0.29435091, 0.19567684, 0.42998625, 0.07998601],
       [0.28721664, 0.03654805, 0.38581039, 0.29042492],
       [0.42372639, 0.19929034, 0.28386359, 0.09311968]
       ]

p_SU_3 = [
       [0.06084334, 0.17384631, 0.19039525, 0.57491511],
       [0.2424558 , 0.32689075, 0.2702551 , 0.16039835],
       [0.29904198, 0.09054811, 0.25534321, 0.35506671],
       [0.23013793, 0.09460166, 0.32906737, 0.34619304],
       [0.20690329, 0.28099179, 0.07679829, 0.43530663],
       [0.33121953, 0.05055977, 0.28945066, 0.32877003],
       [0.1084882 , 0.59693003, 0.24280109, 0.05178068],
       [0.03015215, 0.24129966, 0.35153093, 0.37701726],
       [0.05300668, 0.31245031, 0.2829692 , 0.35157381],
       [0.29538221, 0.25762475, 0.23505472, 0.21193832]]

p_SU_4 = [
       [0.09730962, 0.38887915, 0.28529084, 0.22852039],
       [0.01314042, 0.28175913, 0.26809688, 0.43700357],
       [0.18813798, 0.06867587, 0.16365664, 0.57952951],
       [0.35875416, 0.15459592, 0.13441669, 0.35223324],
       [0.20525672, 0.24522102, 0.28097974, 0.26854252],
       [0.15434565, 0.28783439, 0.15750141, 0.40031855],
       [0.07728072, 0.16792752, 0.1773763 , 0.57741546],
       [0.29152017, 0.19891043, 0.39628497, 0.11328443],
       [0.24187067, 0.08259983, 0.65462786, 0.02090163],
       [0.27273774, 0.29094668, 0.24389187, 0.1924237 ]
       ]

p_SU_5 = [
       [0.05107346, 0.05169562, 0.11518363, 0.78204728],
       [0.10252445, 0.15966677, 0.08823257, 0.64957621],
       [0.10491243, 0.16066294, 0.16422305, 0.57020158],
       [0.19049839, 0.11804792, 0.19267804, 0.49877565],
       [0.05542195, 0.31988261, 0.23249926, 0.39219618],
       [0.81121973, 0.06266485, 0.07368292, 0.0524325 ],
       [0.36503636, 0.00423233, 0.15704134, 0.47368997],
       [0.37228114, 0.08883578, 0.25625632, 0.28262675],
       [0.27036591, 0.2661399 , 0.21971602, 0.24377817],
       [0.26111373, 0.24519586, 0.43288094, 0.06080947]
       ]

p_SU_6 = [
       [0.07525058, 0.16554491, 0.34432987, 0.41487465],
       [0.12892919, 0.17141703, 0.15001509, 0.54963869],
       [0.14507854, 0.18398687, 0.12021666, 0.55071793],
       [0.18542521, 0.36861768, 0.09730517, 0.34865194],
       [0.3205763 , 0.29179786, 0.10906981, 0.27855603],
       [0.41502255, 0.0193601 , 0.27098502, 0.29463233],
       [0.35723613, 0.31903216, 0.30192554, 0.02180617],
       [0.55050605, 0.06390999, 0.31816956, 0.0674144 ],
       [0.8269601 , 0.02343547, 0.14909019, 0.00051424],
       [0.22204796, 0.41187935, 0.33495139, 0.0311213 ]
       ]

p_SU_set = [ p_SU_1, p_SU_2, p_SU_3, p_SU_4, p_SU_5, p_SU_6 ]

# 不同的次要用户有不同的能量损失
c_SU_set = np.zeros((U,N))

for u in range(U):
    for i in range(N):
        c_SU_set[u][i] = c_channel_sensing_cost[i] + np.dot( c_channel_cost[i], p_SU_set[u][i] ) 
 
# In[5] U 个次要用户，N条信道，每个组合定义一个 class
channel_SU_set = []
for u in range(U):
    channel_SU_set.append([])
    for i in range(N):
        channel_SU_set[u].append( channel_info( p_SU_set[u][i], c_channel_sensing_cost[i], c_channel_cost[i], mu_SU_1_set[i] ) )
    
mu_SU_set = []
policy_SU_set = []
for u in range(U):
    policy_SU_set.append([])
    for i in range(N):
        policy_SU_set[u].append( algo(channel_SU_set[u][i]) )


# 两个关键参数 policy_SU_set 和 channel_SU_set
# 假设不同用户先有相同的 budget，为了表示不同的 budget，可以在 threshold 中设置

# In[0] 求最优信道组合 和 Benchmark 
reward_optimal = np.zeros(N)
reward_cost_ratio = np.zeros(N)
reward_optimal_value_benchmark = np.zeros(len(budget_list))
reward_sum_benchmark = 0
cost_sum_benchmark = 0

reward_optimal_SU_set = np.zeros((U,N))
reward_cost_ratio_SU_set = np.zeros((U,N))

for u in range(U):
    for i in range(N):
        reward_optimal_SU_set[u][i] = copy.deepcopy( channel_SU_set[u][i].mu )
        reward_cost_ratio_SU_set[u][i] = copy.deepcopy( reward_optimal_SU_set[u][i]/channel_SU_set[u][i].cost )

print("The optimal channels are")
for u in range(U): # 找到最大值的位置
    print(reward_cost_ratio_SU_set[u].tolist().index(max(reward_cost_ratio_SU_set[u].tolist())) )

optimal_sum_reward_cost_ratio_SU_set = np.zeros(U).tolist()

for u in range(U):
    optimal_sum_reward_cost_ratio_SU_set[u] = max( reward_cost_ratio_SU_set[u].tolist() )


SU_threshold = np.zeros(U).tolist()

optimal_reward_sum = np.zeros(len(budget_list))
optimal_reward_SU = np.zeros(U)

optimal_reward_average = np.zeros(len(budget_list))

for index, budget in enumerate(budget_list):
    for u in range(U):
        optimal_reward_SU[u] = copy.deepcopy( (budget - SU_threshold[u]) * optimal_sum_reward_cost_ratio_SU_set[u] )
        optimal_reward_sum[index] += copy.deepcopy( optimal_reward_SU[u] )
        optimal_reward_average[index] = optimal_reward_sum[index]/budget

reward_optimal_value_benchmark = copy.deepcopy(optimal_reward_sum)
 
# In[1] 所提算法仿真

epsilon_bertsekas = 0.0001
average_reward_set = np.zeros(len(budget_list))
sum_reward_set = np.zeros(len(budget_list))
reward_difference_set = np.zeros(len(budget_list))
normalized_reward_difference_set = np.zeros(len(budget_list))


for index_, budget in enumerate(budget_list):
    
    simulation_Num_ = math.ceil(budget * 300)
    reward_monte_set = np.zeros( (Monte_Carlo_Num, simulation_Num_) )
    reward_monte_mean = np.zeros(simulation_Num_)
    end_time_budget = np.zeros(Monte_Carlo_Num)    
    
    end_time_budget_SU_set = np.zeros((U,Monte_Carlo_Num))
    
    sum_reward = np.zeros(Monte_Carlo_Num)
    reward_difference = np.zeros(Monte_Carlo_Num)
    normalized_reward_difference = np.zeros(Monte_Carlo_Num)
    a_Num_chosen = np.zeros((1,N)).tolist()[0]
    print("The budget is", budget)
    
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
        reward_monte[0] = 1e-04  # 该处将初始reward 设置为很小的数，是为了和后面判决条件 reward = 0 时 break 对应
        
        budget_residual = np.ones(U) * budget
    
        for num in range(simulation_Num_):
            
            for u in range(U):
                for i in range(N):
                    reward_U_N[u][i] = copy.deepcopy( channel_SU_set[u][i].mu ) + random.gauss(0,0.000001)
                    cost_U_N[u][i] = copy.deepcopy( channel_SU_set[u][i].cost_random() )
            
            temp_reward_SU = np.zeros(U) # 各个次要用户的吞吐量
            temp_cost_SU = np.zeros(U) # 各个次要用户的能耗

            temp_reward = 0 # 总的瞬时吞吐量
            temp_cost = 0 # 总的瞬时能耗
            
            if num < 1:
                for u in range(U):
                    for i in range(N):
                        policy_SU_set[u][i].compute_index( reward_U_N[u][i], num, cost_U_N[u][i] ) # 初始化
                
            else:
                for u in range(U):
                    for i in range(N):
                        index_U_N[u][i] = policy_SU_set[u][i].compute_index_update( num ) # 计算用户-信道标号
                
                
                # 当剩余能量小于阈值时，用户-信道标号为0 （表示不参与后续过程的感知接入过程）
                for u in range(U):
                    if budget_residual[u] < SU_threshold[u]:
                        index_U_N[u] = copy.deepcopy( np.zeros(N).tolist() )
                   
                
                """
                # Hungarian 算法求信道分配方案
                hungarian = Hungarian()
                hungarian.calculate(index_U_N.tolist(), is_profit_matrix=True)
                allocation_relationship = hungarian.get_results()
                """
                
                
                """
                添加 bertsekas 模块
                """
                
                # ''' origin auction 
                agents = assign_values(U)
                cost_list = np.zeros(N).tolist()
                Iter_Num = 0
                Iter_Num = sophisticated_auction(epsilon_bertsekas, agents, index_U_N, cost_list)
                allocation_relationship_origin = []
                for u in range(U):
                    allocation_relationship_origin.append( [ u, agents[u][0] ])  
                allocation_relationship = copy.deepcopy(allocation_relationship_origin)
                # '''
                
                # allocation_relationship = [ [0,3], [1,4], [2,6], [3,7], [4,5], [5,8] ]  
                
                
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
                    temp_reward_SU[temp_SU] = copy.deepcopy(reward_U_N[temp_SU][temp_channel])
                    temp_cost_SU[temp_SU] = copy.deepcopy(cost_U_N[temp_SU][temp_channel])
                    budget_residual[temp_SU] = budget_residual[temp_SU] - temp_cost_SU[temp_SU]  # - Iter_Num * c_a  因为加能量了和 benchmark 就不同了， 仅比较平均收益
                        
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
    reward_difference_set[index_] = reward_optimal_value_benchmark[index_] - sum_reward_set[index_] 
    normalized_reward_difference_set[index_] = reward_difference_set[index_]
    chosen_time_set = []
    # for i in range(len(policy)):
    #     print(policy[i].chosen_time)
    #     chosen_time_set.append(policy[i].chosen_time)
plt.plot(budget_list, normalized_reward_difference_set)
plt.title('relationship between regret and energy')
plt.show() 


chosen_time_matrix = []
for u in range(U):
    chosen_time_matrix.append([])
    for i in range(N):
        chosen_time_matrix[u].append( policy_SU_set[u][i].chosen_time )
 
# average throughput 
plt.plot(budget_list, average_reward_set)
plt.title('relationship between throughput/energy and energy')
plt.show() 

end = time.time() # 记录结束时间
print("程序运行时间: " + str(end-start) + "秒")










