# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 12:13:23 2021

@author: xxx
 
 

"""

# 备注：求拉动次数的时候，需要设置为 reward 和 cost 随机值


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
import Fast_fading_model_with_addtional_modeling_components as Fading_Parameter


start = time.time()

# In[0] 参数设置
alpha_0 = 1
alpha_1 =  1/8
budget_list = [  10,  20,  30,  40,   50, 60, 70, 80, 90, 100, 110 ,120, 130, 140, 150, 160, 170, 180      ] # J 次要用户能量
 
Monte_Carlo_Num = 20

U = 3 # 次要用户的数量  
N = 10 # 设置好信道数量    
Tx_antenna_Num = 4
c_allocation_probing_ACK = 1e-02 + 2.5e-04 # 表示分配、估计、ACK耗能

transmission_rate = 1 # Mbps 
duration = 95e-03 # s
bandwidth = 1 # MHz
Delta_min = 0.001  # 分布式中考虑数据精度，假设截断3位小数
epsilon_bertsekas = Delta_min/(U+1) # 对应文中 epsilon = Delta_min/(U+1)
g_max = 5 # 设置 g_max
transmission_distance = [ 315.1545, 324.23451, 333.535329,  345.2345543,  349.85736952,  353.32453458 ]
Channel_Noise_vector = [ -99 - 0.5 * i for i in range(N) ]
f_c = 6 # GHz
scenario = 'UMa' # RMa, UMa, UMi_Street_canyon, InH_Office   Rural, Urban, Street 
Channel_Noise = [ 10**(Channel_Noise_vector[i]/10)/(10**3) for i in range(len(Channel_Noise_vector)) ] 
Channel_Noise_Set = [ Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise ] # 不同用户的SNR


# In[0] Gale-Shapley 算法
def GS_matching(MP,WP):
    #MP是男性（次要用户）的择偶排序的集合 WP是女性（信道）的
    m = len(MP)
    n = len(WP)
    #给出男性和女性是否单身的数组用以评价
    isManFree = [True]*m
    isWomenFree = [True]*n
    #男性是否向女性求过婚的表格
    isManProposed = [[False for i in range(n)]for j in range(m)]
    #最后匹配得出的组合 返回结果
    match = [(-1,-1)]*m

    while(True in isManFree):
        #找到第一个单身男性的索引值
        indexM = isManFree.index(True)
        #对每个女生求婚  找到男性优先列表中还没找到对象的女性
        if(False in isManProposed[indexM]):
            indexW = -1  #找到还没被求婚的排名靠前的女性的索引
            for i in range(len(MP[indexM])):
                w = MP[indexM][i]
                if(not isManProposed[indexM][w]):
                    indexW = w
                    break
            isManProposed[indexM][indexW] = True
            if(isWomenFree[indexW]):#女性单身且她愿意接受这个男性
                isWomenAccept = False
                for i in range(len(WP[indexW])):
                    if(WP[indexW][i] == indexM):
                        isWomenAccept = True
                if(isWomenAccept):
                    isWomenFree[indexW] = False
                    isManFree[indexM] = False
                    match[indexM] = (indexM,indexW)
            else:
                indexM1 = -1   #与当前女性已匹配的男性的索引
                isWomenAccept = False
                for i in range(len(WP[indexW])):#找到这个人是否能被这个女性接受
                    if (WP[indexW][i] == indexM):
                        isWomenAccept = True
                if(isWomenAccept):
                    for j in range(m):
                        if(match[j][1] == indexW):
                            indexM1 = j
                            break
                    if(WP[indexW].index(indexM)<WP[indexW].index(indexM1)):
                        isManFree[indexM1] = True
                        isManFree[indexM] = False
                        match[indexM] = (indexM,indexW)
    # 删除没有进行配对的值  即任一方值为-1
    for i in range(len(match) - 1, -1, -1):
        # 倒序循环，从最后一个元素循环到第一个元素。不能用正序循环，因为正序循环删除元素后后续的列表的长度和元素下标同时也跟着变了，len(list)是动态的。
        if (match[i][0] == -1 or match[i][1] == -1):
                match.pop(i)
    print(match)
    return match

# if __name__=="__main__":
#     print("*************测试*******************")
    
#     matrix = np.array([
#     [ 0,1,2,3,4 ],
#     [ 0,1,2,3,4 ]
#         ])
    
    
#     MP = [[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4]]
#     WP = [[1,0,2,3],[2,1,0,3],[2,1,0,3],[3,2,1,0],[3,2,1,0]]

#     SM = GS_matching(MP,WP)
    
    



# In[0] 编写 Bertsekas 算法
# 根据文中分析，迭代次数 N_{Iter,thres} 最大值为 UN + UN(U+1)/Delta_min g_{max}，而 g_{max} 表示信道标号最大值
# 因此信道分配算法中消耗的能量可用常数形式表示，表现在信道分配、估计、ACK的能耗之中
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
 
    

# In[1] 定义函数和 class


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
 
        
        # some other parameters
        self.mean_cost_ = 1 
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
        self.index = alpha_0 * self.mean_reward/(self.mean_cost) +  np.sqrt( alpha_1 * np.log(time) / (self.chosen_time) )/self.mean_cost_ # or self.mean_cost 这个影响不大
        return self.index

    def compute_index_update_random(self):
        self.index = np.random.random()
        return self.index

 
    # EABS-TS
    def compute_index_update_EABS_TS(self, transmission_distance, residual_energy_):
        theta_temp = np.random.normal( self.mean_reward, 1/(1+self.chosen_time) )
        self.index = theta_temp - transmission_distance/residual_energy_
        return self.index
        
        
    def compute_index_update_MAB_without_energy(self, time):
        self.index = 12 * self.mean_reward +  np.sqrt( alpha_1 * np.log(time) / (self.chosen_time) )
        return self.index    
    
    def compute_index_update_greedy(self):
        self.index = alpha_0 * self.mean_reward/(self.mean_cost) 
        return self.index    
    
    # ONES
    def compute_c_term_temp(self, time, channel_Num):
        c_1 = 2 * max( 0, (self.channel_info.transmission_rate * self.channel_info.duration)**2 * (0.01 - 0.55e-03)**2/(0.55e-03)**4 ) # epsilon_max 设置为 0.01 J 
        d_0 = np.sqrt(8 * max(0, (self.channel_info.transmission_rate * self.channel_info.duration)**2 * (0.01 - 0.55e-03)/(0.55e-03)**2 ))
        d_1 = np.sqrt(2 * 0.01**2 * c_1)
        self.c_term_temp = np.sqrt( 2 * c_1/(5*c_1) * np.log( time * np.sqrt(channel_Num + 1) )/self.chosen_time ) # 备注：此处将 c_1 去除，因为不同场景、模型下的因子不同，因此不能完全相同
        self.c_term_temp_estimated = (d_0 + d_1)/300 * np.sqrt(np.log(time * np.sqrt(channel_Num + 1))/self.chosen_time )
    def compute_index_update_ONES(self, instant_reward, time, g_star, channel_Num):
        self.index = self.mean_reward - self.mean_cost * g_star + self.c_term_temp
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
    def __init__(self, Tx_antenna_Num, c_allocation_probing_ACK, transmission_rate, duration, bandwidth, transmission_distance, Channel_Noise):
        self.c_allocation_probing_ACK = c_allocation_probing_ACK
        self.transmission_rate = transmission_rate # transmission rate 
        self.duration = duration
        self.throughput = transmission_rate * duration
        self.transmission_distance = transmission_distance
        self.Tx_antenna_Num = Tx_antenna_Num
        self.Bandwidth = bandwidth
        self.Channel_Noise = Channel_Noise
        self.PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/(self.transmission_distance**(3.76)) )

    def PathLoss_Model(self):
        # channel noise is in the form of num
        # PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/(self.transmission_distance**(3.76)) )
        Channel_parameter_vector = []
        for i in range(self.Tx_antenna_Num):
            Channel_parameter_vector.append( self.PathLoss_Model_Parameter * 1/np.sqrt(2) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) )
            # 其他场景直接求信噪比比较好 或者信道系数
        Channel_parameter_norm = np.linalg.norm( Channel_parameter_vector )
        self.Channel_gain_noise_ratio = Channel_parameter_norm**2/self.Channel_Noise

    def Link_Parameter(self):
        return Fading_Parameter.main(scenario, self.transmission_distance, f_c)

    def PathLoss_Model_5G(self, scenario, f_c, link_parameter):
        # link_parameter = Fading_Parameter.main(scenario, self.transmission_distance, f_c)
        self.Channel_gain_noise_ratio = link_parameter**2/self.Channel_Noise

    def Energy_minimum_(self): # definition of the channel cost
        self.Power_minimum = ( 2**(self.transmission_rate/self.Bandwidth) - 1 )/self.Channel_gain_noise_ratio
        self.Energy_minimum = self.Power_minimum * self.duration # + self.c_allocation_probing_ACK 不在这里添加信道分配、探测、接入的能量，仅考虑传输过程的能耗
        return self.Power_minimum, self.Energy_minimum        



# In[0] 每个用户-信道组合定义一个class


channel_SU_set = []
for u in range(U):
    channel_SU_set.append([])
    for i in range(N):
        channel_SU_set[u].append( channel_info( Tx_antenna_Num, c_allocation_probing_ACK, transmission_rate, duration, bandwidth, transmission_distance[u], Channel_Noise_Set[u][i]   ) )
 
    
# In[1] 这是用来静态 static Hungarian Algorithm 使用的
# 假设各次要用户知道信道噪声及质量分布，可用信道噪声等表示信道质量。假设信道质量相同分布，信道噪声小则信道质量较好

Channel_gain_noise_ratio_Matrix = np.zeros( (U,N) )
for u in range(U):
    for i in range(N):
        Channel_gain_noise_ratio_Matrix[u][i] = channel_SU_set[u][i].PathLoss_Model_Parameter/channel_SU_set[u][i].Channel_Noise/1e7
 

# In[0] 每个用户-信道组合对应的算法定义一个class
policy_SU_set = []
for u in range(U):
    policy_SU_set.append([])
    for i in range(N):
        policy_SU_set[u].append( algo(channel_SU_set[u][i]) )

 
# In[0] 每个用户剩余能量阈值  
SU_threshold = np.zeros(U).tolist()

# In[0] load the channel parameter 5G
# Data_samplins 存储的是所需的最小能量
# b_1 = np.load(file = "Data_samplings_1.npy")   # 0.0035268431881289235
# b_2 = np.load(file = "Data_samplings_2.npy")   # 0.0033139947042351158
# b_3 = np.load(file = "Data_samplings_3.npy")    # 0.003591596309422101
# b_4 = np.load(file = "Data_samplings_4.npy")  # 0.0037611342790890645
# b_5 = np.load(file = "Data_samplings_5.npy")  # 0.0034461265162141984

# UMa 的数据
# Data_samplings_5G_UMa = np.load(file = "Data_samplings_5G_U_Ma.npy")     
# Data_samplings_5G_UMa_1 = np.load(file = "Data_samplings_5G_UMa_1.npy")    
# Data_samplings_5G_UMa_2 = np.load(file = "Data_samplings_5G_UMa_2.npy")    
# Data_samplings_5G_UMa_3 = np.load(file = "Data_samplings_5G_UMa_3.npy")   
# Data_samplings_5G_UMa_4 = np.load(file = "Data_samplings_5G_UMa_4.npy")   
# Data_samplings_5G_UMa_5 = np.load(file = "Data_samplings_5G_UMa_5.npy")   
# Data_samplings_5G_UMa_6 = np.load(file = "Data_samplings_5G_UMa_6.npy")   
# Data_samplings_5G_UMa_7 = np.load(file = "Data_samplings_5G_UMa_7.npy")   
# Data_samplings_5G_UMa_8 = np.load(file = "Data_samplings_5G_UMa_8.npy")    
# Data_samplings_5G_UMa_9 = np.load(file = "Data_samplings_5G_UMa_9.npy") 

# Data_samplings_5G_UMa = Data_samplings_5G_UMa.tolist()    
# Data_samplings_5G_UMa_1 = Data_samplings_5G_UMa_1.tolist() 
# Data_samplings_5G_UMa_2 = Data_samplings_5G_UMa_2.tolist() 
# Data_samplings_5G_UMa_3 = Data_samplings_5G_UMa_3.tolist() 
# Data_samplings_5G_UMa_4 = Data_samplings_5G_UMa_4.tolist() 
# Data_samplings_5G_UMa_5 = Data_samplings_5G_UMa_5.tolist() 
# Data_samplings_5G_UMa_6 = Data_samplings_5G_UMa_6.tolist() 
# Data_samplings_5G_UMa_7 = Data_samplings_5G_UMa_7.tolist() 
# Data_samplings_5G_UMa_8 = Data_samplings_5G_UMa_8.tolist() 
# Data_samplings_5G_UMa_9 = Data_samplings_5G_UMa_9.tolist()

# RMa 的数据
# Data_samplings_5G_RMa = np.load(file = "Data_samplings_5G_RMa.npy")      # 0.00033009311299963233
# Data_samplings_5G_RMa_1 = np.load(file = "Data_samplings_5G_RMa_1.npy")   # 0.00036760237810014984
# Data_samplings_5G_RMa_2 = np.load(file = "Data_samplings_5G_RMa_2.npy")   # 0.00035440906771119004
# Data_samplings_5G_RMa_3 = np.load(file = "Data_samplings_5G_RMa_3.npy")   # 0.0003582378976452796


# Data_samplings_5G_RMa = Data_samplings_5G_RMa.tolist()
# Data_samplings_5G_RMa_1 = Data_samplings_5G_RMa_1.tolist()
# Data_samplings_5G_RMa_2 = Data_samplings_5G_RMa_2.tolist()
# Data_samplings_5G_RMa_3 = Data_samplings_5G_RMa_3.tolist()



# UMi_Street_canyon 的数据

# Data_samplings_5G_UMi_Street_canyon = np.load(file = "Data_samplings_5G_UMi_Street_canyon.npy")  # 0.007809311582311418
Data_samplings_5G_UMi_Street_canyon_1 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_1.npy")  # 0.21789027582508214
Data_samplings_5G_UMi_Street_canyon_2 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_2.npy")  # 0.20969688464509287
Data_samplings_5G_UMi_Street_canyon_3 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_3.npy")  # 0.21091341136649977
Data_samplings_5G_UMi_Street_canyon_4 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_4.npy")  # 0.009194618475684042
Data_samplings_5G_UMi_Street_canyon_5 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_5.npy")  # 0.20207788862942508
Data_samplings_5G_UMi_Street_canyon_6 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_6.npy")  # 0.2208653720903688
Data_samplings_5G_UMi_Street_canyon_7 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_7.npy")  # 0.21001564692464939
Data_samplings_5G_UMi_Street_canyon_8 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_8.npy")  # 0.21001564692464939
Data_samplings_5G_UMi_Street_canyon_9 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_9.npy")  # 0.21001564692464939
Data_samplings_5G_UMi_Street_canyon_10 = np.load(file = "Data_samplings_5G_UMi_Street_canyon_10.npy")  # 0.21001564692464939

# Data_samplings_5G_UMi_Street_canyon = Data_samplings_5G_UMi_Street_canyon.tolist()
Data_samplings_5G_UMi_Street_canyon_1 = Data_samplings_5G_UMi_Street_canyon_1.tolist()
Data_samplings_5G_UMi_Street_canyon_2 = Data_samplings_5G_UMi_Street_canyon_2.tolist()
Data_samplings_5G_UMi_Street_canyon_3 = Data_samplings_5G_UMi_Street_canyon_3.tolist()
Data_samplings_5G_UMi_Street_canyon_4 = Data_samplings_5G_UMi_Street_canyon_4.tolist()
Data_samplings_5G_UMi_Street_canyon_5 = Data_samplings_5G_UMi_Street_canyon_5.tolist()
Data_samplings_5G_UMi_Street_canyon_6 = Data_samplings_5G_UMi_Street_canyon_6.tolist()
Data_samplings_5G_UMi_Street_canyon_7 = Data_samplings_5G_UMi_Street_canyon_7.tolist()
Data_samplings_5G_UMi_Street_canyon_8 = Data_samplings_5G_UMi_Street_canyon_8.tolist()
Data_samplings_5G_UMi_Street_canyon_9 = Data_samplings_5G_UMi_Street_canyon_9.tolist()
Data_samplings_5G_UMi_Street_canyon_10 = Data_samplings_5G_UMi_Street_canyon_10.tolist()

# In[1] 导入数据

ccc_user_1 = []
ddd_user_1 = []
for i in range(len(Data_samplings_5G_UMi_Street_canyon_7)):
    ccc_user_1.append(Data_samplings_5G_UMi_Street_canyon_7[i][0])
    ddd_user_1.append(Data_samplings_5G_UMi_Street_canyon_7[i][0][0])

sum(ddd_user_1)/len(ddd_user_1)    
 
# UMa 的数据


# RMa 的数据
# data_set = [ Data_samplings_5G_RMa, Data_samplings_5G_RMa_1, Data_samplings_5G_RMa_2, Data_samplings_5G_RMa_3 ]

# UMi_Street_canyon 的数据
data_set = [ Data_samplings_5G_UMi_Street_canyon_1, Data_samplings_5G_UMi_Street_canyon_2, Data_samplings_5G_UMi_Street_canyon_3, Data_samplings_5G_UMi_Street_canyon_4, Data_samplings_5G_UMi_Street_canyon_5, Data_samplings_5G_UMi_Street_canyon_6, Data_samplings_5G_UMi_Street_canyon_7, Data_samplings_5G_UMi_Street_canyon_8, Data_samplings_5G_UMi_Street_canyon_9 ]     
data_5G = []

for i in range( len(data_set) ):
    for j in range( len(data_set[i]) ):
        data_5G.append( data_set[i][j] )


 
# In[1] 算法仿真部分

 
average_reward_set = np.zeros(len(budget_list))
sum_reward_set = np.zeros(len(budget_list))
reward_difference_set = np.zeros(len(budget_list))
normalized_reward_difference_set = np.zeros(len(budget_list))


for index_, budget in enumerate(budget_list):
    
    simulation_Num_ = math.ceil( budget * 20 )
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
        g_star = np.zeros(U)
        
        U_temp = U
        Channel_gain_noise_ratio_Matrix_temp = copy.deepcopy( Channel_gain_noise_ratio_Matrix )
        
        # """
        shuffle(data_5G) # 每次蒙特卡洛仿真随机更换一次数据
        # """
        
        for num in range(simulation_Num_):
 
            for u in range(U_temp):
                for i in range(N):
                    ''' 原先在代码中产生5G信道系数并求得能耗，但耗时太久，因此先生成能耗存储在 data_5G 中
                    link_parameter = channel_SU_set[u][i].Link_Parameter()
                    channel_SU_set[u][i].PathLoss_Model_5G(scenario, f_c, link_parameter)
                    print('the user', u, 'on channel', i, 'channel gain is', channel_SU_set[u][i].Channel_gain_noise_ratio )
                    Power_Requirement_temp, Energy_Requirement_temp = channel_SU_set[u][i].Energy_minimum_()
                    cost_U_N[u][i] = copy.deepcopy( Energy_Requirement_temp )
                    print('the user', u, 'on channel', i, 'minimum energy requirement is', Energy_Requirement_temp )
                    '''
                    reward_U_N[u][i] = copy.deepcopy( channel_SU_set[u][i].throughput )  
                    cost_U_N[u][i] = data_5G[ num ][u][i] + policy_SU_set[u][i].channel_info.c_allocation_probing_ACK  # 能耗
 
            temp_reward = 0
            temp_cost = 0
            
            temp_reward_SU = np.zeros(U)
            temp_cost_SU = np.zeros(U)
            
            if num < 1:
                for u in range(U):
                    for i in range(N):
                        policy_SU_set[u][i].compute_index( reward_U_N[u][i], num, cost_U_N[u][i] )
                
            else:
                
                epsilon_ = 0.1  
                     
                # """
                g_star = np.zeros(U)
                for u in range(U):
                    for i in range(N):
                        # proposed 
                        index_U_N[u][i] = policy_SU_set[u][i].compute_index_update( num )
                        index_U_N[u][i] = min( round(index_U_N[u][i],3), g_max ) # 由于设置的精度为 0.001，因此 取三位 小数    
                        
                        # random 
                        # index_U_N[u][i] = policy_SU_set[u][i].compute_index_update_random()
                # """ 
                        # ONES
                        policy_SU_set[u][i].compute_c_term_temp( num, N )                
                    g_star[u] = max( [ policy_SU_set[u][i].mean_reward/(policy_SU_set[u][i].mean_cost) - policy_SU_set[u][i].c_term_temp for i in range(N) ]   )
                    for i in range(N):                  
                        index[i] = policy_SU_set[u][i].compute_index_update_ONES(reward[i], num, g_star[u], N)                        
 
  
                
                # 次要用户剩余能量小于阈值时停止信道接入过程
                for u in range(U):
                    if budget_residual[u] < SU_threshold[u]:
                        index_U_N[u] = copy.deepcopy( np.zeros(N).tolist() )   
                        Channel_gain_noise_ratio_Matrix_temp[u] = copy.deepcopy( np.zeros(N).tolist() ) 
 

                """ Static Benchamrk # 该处是为了求得 benchmark 带来的最优能量效率
                # hungarian 算法模块 in known environment
                hungarian = Hungarian()
                hungarian.calculate(Channel_gain_noise_ratio_Matrix_temp.tolist(), is_profit_matrix=True)
                allocation_relationship = hungarian.get_results()
                Iter_Num = 0
                """
                
                # """
                # 添加 Bertsekas 模块，注意我们在 auction 算法中的 allocation_relationship 和 hungarian 算法中的 allocation_relationship 交换了一下
                # """
                 
                # ''' 进行信道分配
                agents = assign_values(U_temp)
                cost_list = np.zeros(N).tolist()
                Iter_Num = 0
                Iter_Num = sophisticated_auction(epsilon_bertsekas, agents, index_U_N, cost_list)
                allocation_relationship_origin = []
                for u in range(U_temp):
                    allocation_relationship_origin.append( [ u, agents[u][0] ])  
                allocation_relationship = copy.deepcopy(allocation_relationship_origin)
                # '''
                    
                    
                
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
                    budget_residual[temp_SU] = budget_residual[temp_SU] - temp_cost_SU[temp_SU]   
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
    # reward_difference_set[index_] = reward_optimal_value_benchmark[index_] - sum_reward_set[index_] 
    # normalized_reward_difference_set[index_] = reward_difference_set[index_]
    # chosen_time_set = []
    # for i in range(len(policy)):
    #     print(policy[i].chosen_time)
    #     chosen_time_set.append(policy[i].chosen_time)
    
# In[0] 吞吐量损失与能量关系
# plt.plot(budget_list, normalized_reward_difference_set)


# In[0] 每条信道感知次数
chosen_time_matrix = []
for u in range(U):
    chosen_time_matrix.append([])
    for i in range(N):
        chosen_time_matrix[u].append( policy_SU_set[u][i].chosen_time )



# sum(optimal_sum_reward_cost_ratio_SU_set)

# In[0] 能量效率与能量关系 
plt.plot(budget_list , average_reward_set )
 
 
end = time.time()
total_time = end - start 
print(total_time)









