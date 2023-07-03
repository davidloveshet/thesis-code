# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:03:56 2022
Modified on 13/05/2023
@author: xxx

The simulation parameter is as follows.

We choose the user number to be 3, the channel number is 10.

由于在仿真中生成5G系数再进行learning时代码运行速度很慢，因此在该代码中先生成5G信道环境下的SNR并存储下来
可调整参数：
scenario 中含有三种场景，分别为 UMa RMa 和 UMi_Street_canyon
simulation_Num_ 为数据数量，产生后存储在 Data_samplings 中

* 注意的是这里仅生成次要用户传输时所需的最小的能量，没有添加信道分配、探测、ACK的能耗！！！需要在仿真中自己添加
"""

# generate channel parameter in 5G

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

    
# In[0] 注意在这里调整5G信道场景 

# '''
scenario = 'UMa' # RMa, UMa, UMi_Street_canyon, InH_Office   Rural, Urban, Street 
scenario = 'RMa'
scenario = 'UMi_Street_canyon'
# ''' 

Data_samplings = []
f_c = 6 # GHz


# In[1] 定义 class
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


    def PathLoss_Model(self):
        # channel noise is in the form of num
        PathLoss_Model_Parameter = np.sqrt( 10**(-3.53)/(self.transmission_distance**(3.76)) )
        Channel_parameter_vector = []
        for i in range(self.Tx_antenna_Num):
            Channel_parameter_vector.append( PathLoss_Model_Parameter * 1/np.sqrt(2) * ( np.random.normal() + complex(0, 1) * np.random.normal() ) )
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
        self.Energy_minimum = self.Power_minimum * self.duration # + self.c_allocation_probing_ACK 在这里不添加分配、探测、ACK的能耗
        return self.Power_minimum, self.Energy_minimum   
    
    
# In[1] 参数设置
U = 3
N = 10 
Tx_antenna_Num = 4
c_allocation_probing_ACK = 1e-02 + 2.5e-04 # 表示分配、估计、ACK耗能
transmission_rate = 1 # Mbits/s/Hz
duration = 95e-03 # s
bandwidth = 1 # MHz
simulation_Num_ = 100 # 数据量
transmission_distance = [ 315.1545, 324.23451, 333.535329,  345.2345543,  349.85736952,  353.32453458 ]
Channel_Noise_vector = [ -99 - 0.5 * i for i in range(N) ]    
Channel_Noise = [ 10**(Channel_Noise_vector[i]/10)/(10**3) for i in range(len(Channel_Noise_vector)) ] 

Channel_Noise_Set = [ Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise, Channel_Noise ]


channel_SU_set = []
for u in range(U):
    channel_SU_set.append([])
    for i in range(N):
        channel_SU_set[u].append( channel_info( Tx_antenna_Num, c_allocation_probing_ACK, transmission_rate, duration, bandwidth, transmission_distance[u], Channel_Noise_Set[u][i]   ) )
     
        
     
        
for num in range(simulation_Num_):
    
    cost_U_N = np.zeros((U,N))
    for u in range(U):
        for i in range(N):
            link_parameter = channel_SU_set[u][i].Link_Parameter()
            channel_SU_set[u][i].PathLoss_Model_5G(scenario, f_c, link_parameter)
            # print('the user', u, 'on channel', i, 'channel gain is', channel_SU_set[u][i].Channel_gain_noise_ratio )
            Power_Requirement_temp, Energy_Requirement_temp = channel_SU_set[u][i].Energy_minimum_()
            cost_U_N[u][i] = copy.deepcopy( Energy_Requirement_temp )    # 所需的最小能量
    Data_samplings.append(cost_U_N)
    print("This is", num,"-th sampling data")



np.save(file = "Data_samplings_5G_UMi_Street_canyon_12", arr = Data_samplings)


end = time.time()

total_time = end - start

 

















