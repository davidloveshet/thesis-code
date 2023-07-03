# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[11]
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import  ConnectionPatch

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black", alpha = 0.3)

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    
    
fig, ax = plt.subplots()


import scipy.io
# data = scipy.io.loadmat('N_5_U_2_M_2_K_1_regret.mat') # 读取mat文件
Sample_Num = 100000
x_Num = list(range(Sample_Num))
axins = ax.inset_axes( ( 0.5, 0.36, 0.35, 0.26 ) )

proposed_regret = scipy.io.loadmat('P_d_proposed_regret.mat')['Regret_as_time'][0]
sense_optimal_random_access_regret = scipy.io.loadmat('P_d_sense_optimal_random_access_regret.mat')['Regret_as_time'][0]
random_regret = scipy.io.loadmat('P_d_random_regret.mat')['Regret_as_time'][0]
MAB_regret = scipy.io.loadmat('P_d_MAB_regret.mat')['Regret_as_time'][0]
 
pure_exploitation_regret = scipy.io.loadmat('P_d_pure_exploitation_regret.mat')['Regret_as_time'][0]


plt.plot(x_Num[2:1000], proposed_regret[2:1000],'-', label = '$\pi^C$-IID', LineWidth = 2)
plt.plot(x_Num[2:1000], MAB_regret[2:1000],'-', label = 'UCB', LineWidth = 2)
plt.plot(x_Num[2:450], sense_optimal_random_access_regret[2:450],'-', label = 'sense optimal access random', LineWidth = 2)
plt.plot(x_Num[2:400], random_regret[2:400],'-', label = 'pure exploration', LineWidth = 2)
 
plt.plot(x_Num[2:500], pure_exploitation_regret[2:500],'-', label = 'pure exploitation', LineWidth = 2)


# U_2_M_3_K_1_regret = scipy.io.loadmat('U_2_M_3_K_1_regret.mat')['Regret_as_time'][0]
# U_2_M_3_K_2_regret = scipy.io.loadmat('U_2_M_3_K_2_regret.mat')['Regret_as_time'][0]
# U_3_M_2_K_1_regret = scipy.io.loadmat('U_3_M_2_K_1_regret.mat')['Regret_as_time'][0]
# U_3_M_2_K_2_regret = scipy.io.loadmat('U_3_M_2_K_2_regret.mat')['Regret_as_time'][0]
# U_4_M_2_K_1_regret = scipy.io.loadmat('U_4_M_2_K_1_regret.mat')['Regret_as_time'][0]
# U_4_M_2_K_2_regret = scipy.io.loadmat('U_4_M_2_K_2_regret.mat')['Regret_as_time'][0]
# U_4_M_1_K_1_regret = scipy.io.loadmat('U_4_M_1_K_1_regret.mat')['Regret_as_time'][0]


axins.plot(x_Num[2:1000], proposed_regret[2:1000],'-', label = '$\pi^C$-IID', LineWidth = 2)
axins.plot(x_Num[2:1000], MAB_regret[2:1000],'-', label = 'UCB', LineWidth = 2)

zone_and_linked(ax, axins, 400, 1000, x_Num, [ proposed_regret[2:1000], MAB_regret[2:1000]  ], 'top' )



# N_5_U_2_M_2_K_1_regret = scipy.io.loadmat('N_5_U_2_M_2_K_1_regret.mat')['save_result2'][0]
# N_6_U_2_M_2_K_1_regret = scipy.io.loadmat('N_6_U_2_M_2_K_1_regret.mat')['save_result2'][0]
# N_6_U_2_M_2_K_2_regret = scipy.io.loadmat('N_6_U_2_M_2_K_2_regret.mat')['save_result2'][0]
# N_7_U_2_M_2_K_1_regret = scipy.io.loadmat('N_7_U_2_M_2_K_1_regret.mat')['save_result2'][0]
# N_7_U_2_M_2_K_2_regret = scipy.io.loadmat('N_7_U_2_M_2_K_2_regret.mat')['save_result2'][0]
# N_8_U_3_M_2_K_1_regret = scipy.io.loadmat('N_8_U_3_M_2_K_1_regret.mat')['save_result2'][0]
# N_8_U_3_M_2_K_2_regret = scipy.io.loadmat('N_8_U_3_M_2_K_2_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_1_regret = scipy.io.loadmat('N_20_U_3_M_4_K_1_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_2_regret = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_3_regret = scipy.io.loadmat('N_20_U_3_M_4_K_3_regret.mat')['save_result2'][0]
# N_20_U_3_M_4_K_4_regret = scipy.io.loadmat('N_20_U_3_M_4_K_4_regret.mat')['save_result2'][0]



# plt.plot(x_Num,U_2_M_3_K_1_regret,'-', label = 'U=2,M=3,K=1', LineWidth = 2)
# plt.plot(x_Num,U_2_M_3_K_2_regret,'-', label = 'U=2,M=3,K=2', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_1_regret,'-', label = 'U=2,M=3,K=1', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_2_regret,'-', label = 'U=2,M=3,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_1_regret,'-', label = 'U=4,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_2_regret,'-', label = 'U=4,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_1_K_1_regret,'-', label = 'U=4,M=1,K=1', LineWidth = 2)
# plt.plot(x_Num,N_6_U_2_M_2_K_2_regret,'-', label = 'N=6,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_1_regret,'-', label = 'N=7,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_2_regret,'-', label = 'N=7,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_1_regret,'-', label = 'N=8,U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_2_regret,'-', label = 'N=8,U=3,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_1_regret,'--', label = 'N=20,U=3,M=4,K=1', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_2_regret,'--', label = 'N=20,U=3,M=4,K=2', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_3_regret,'--', label = 'N=20,U=3,M=4,K=3', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_4_regret,'--', label = 'N=20,U=3,M=4,K=4', LineWidth = 2)


plt.xlabel('slot number',fontsize=12)
plt.ylabel('normalized regret',fontsize=12)
# plt.ylim([ -10,115])
# plt.xlim([-500, 35000])
plt.legend(ncol=1,  framealpha = 0)
plt.grid()

# fig.savefig('centralized_regret_comparison.eps', dpi = 600, format = 'eps')



# In[00]
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import  ConnectionPatch

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black", alpha = 0.3)

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    
    
fig, ax = plt.subplots()


import scipy.io
# data = scipy.io.loadmat('N_5_U_2_M_2_K_1_regret.mat') # 读取mat文件
Sample_Num = 100000
x_Num = list(range(Sample_Num))
axins = ax.inset_axes( ( 0.47, 0.54, 0.35, 0.18 ) )


proposed_reward = scipy.io.loadmat('P_d_proposed_reward.mat')['Reward_average_as_time'][0]
sense_optimal_random_access_reward = scipy.io.loadmat('P_d_sense_optimal_random_access_reward.mat')['Reward_average_as_time'][0]
random_reward = scipy.io.loadmat('P_d_random_reward.mat')['Reward_average_as_time'][0]
MAB_reward = scipy.io.loadmat('P_d_MAB_reward.mat')['Reward_average_as_time'][0]
greedy_reward = scipy.io.loadmat('P_d_greedy_reward.mat')['Reward_average_as_time'][0]
pure_exploitation_reward = scipy.io.loadmat('P_d_pure_exploitation_reward.mat')['Reward_average_as_time'][0]

optimal_reward = scipy.io.loadmat('P_d_random_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)



plt.plot(x_Num[2:1000], proposed_reward[2:1000],'-', label = '$\pi^C$-IID', LineWidth = 2)
plt.plot(x_Num[2:1000], MAB_reward[2:1000],'-', label = 'UCB', LineWidth = 2)
plt.plot(x_Num[2:1000], sense_optimal_random_access_reward[2:1000],'-', label = 'sense optimal access random', LineWidth = 2)
plt.plot(x_Num[2:1000], random_reward[2:1000],'-', label = 'pure exploration', LineWidth = 2)
# plt.plot(x_Num, greedy_reward,'--', label = 'greedy', LineWidth = 2)
plt.plot(x_Num[2:1000], pure_exploitation_reward[2:1000],'-', label = 'pure exploitation', LineWidth = 2)

# plt.plot(x_Num[2:1000], proposed_reward[2:1000],'-', label = 'proposed', LineWidth = 2)
# plt.plot(x_Num[2:1000], MAB_reward[2:1000],'-', label = 'MAB', LineWidth = 2)
# plt.plot(x_Num[2:1000], sense_optimal_random_access_reward[2:1000],'-', label = 'sense optimal access random', LineWidth = 2)
# plt.plot(x_Num[2:1000], random_reward[2:1000],'-', label = 'pure exploration', LineWidth = 2)
# # plt.plot(x_Num, greedy_reward,'--', label = 'greedy', LineWidth = 2)
# plt.plot(x_Num[2:1000], pure_exploitation_reward[2:1000],'-', label = 'pure exploitation', LineWidth = 2)

plt.plot(x_Num[2:1000], optimal_reward[2:1000],'-', label = '$\pi^{*C}$-IID', LineWidth = 2)


axins.plot(x_Num[2:1000], proposed_reward[2:1000],'-', label = '$\pi^C$-IID', LineWidth = 2)
axins.plot(x_Num[2:1000], MAB_reward[2:1000],'-', label = 'UCB', LineWidth = 2)

zone_and_linked(ax, axins, 400, 1000, x_Num, [ proposed_reward[2:1000], MAB_reward[2:1000]  ], 'bottom' )



# U_2_M_3_K_1_reward = scipy.io.loadmat('U_2_M_3_K_1_reward.mat')['Reward_average_as_time'][0]
# U_2_M_3_K_2_reward = scipy.io.loadmat('U_2_M_3_K_2_reward.mat')['Reward_average_as_time'][0]
# U_3_M_2_K_1_reward = scipy.io.loadmat('U_3_M_2_K_1_reward.mat')['Reward_average_as_time'][0]
# U_3_M_2_K_2_reward = scipy.io.loadmat('U_3_M_2_K_2_reward.mat')['Reward_average_as_time'][0]
# U_4_M_2_K_1_reward = scipy.io.loadmat('U_4_M_2_K_1_reward.mat')['Reward_average_as_time'][0]
# U_4_M_2_K_2_reward = scipy.io.loadmat('U_4_M_2_K_2_reward.mat')['Reward_average_as_time'][0]
# U_4_M_1_K_1_reward = scipy.io.loadmat('U_4_M_1_K_1_reward.mat')['Reward_average_as_time'][0]

# U_2_M_3_K_1_reward_optimal = scipy.io.loadmat('U_2_M_3_K_1_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_2_M_3_K_2_reward_optimal = scipy.io.loadmat('U_2_M_3_K_2_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_3_M_2_K_1_reward_optimal = scipy.io.loadmat('U_3_M_2_K_1_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_3_M_2_K_2_reward_optimal = scipy.io.loadmat('U_3_M_2_K_2_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_4_M_2_K_1_reward_optimal = scipy.io.loadmat('U_4_M_2_K_1_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_4_M_2_K_2_reward_optimal = scipy.io.loadmat('U_4_M_2_K_2_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)
# U_4_M_1_K_1_reward_optimal = scipy.io.loadmat('U_4_M_1_K_1_reward_optimal.mat')['Reward_expect_all_status'][0] * np.ones(Sample_Num)


# N_5_U_2_M_2_K_1_reward = scipy.io.loadmat('N_5_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_6_U_2_M_2_K_1_reward = scipy.io.loadmat('N_6_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_6_U_2_M_2_K_2_reward = scipy.io.loadmat('N_6_U_2_M_2_K_2_reward.mat')['save_result1'][0]
# N_7_U_2_M_2_K_1_reward = scipy.io.loadmat('N_7_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_7_U_2_M_2_K_2_reward = scipy.io.loadmat('N_7_U_2_M_2_K_2_reward.mat')['save_result1'][0]
# N_8_U_3_M_2_K_1_reward = scipy.io.loadmat('N_8_U_3_M_2_K_1_reward.mat')['save_result1'][0]
# N_8_U_3_M_2_K_2_reward = scipy.io.loadmat('N_8_U_3_M_2_K_2_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_1_reward = scipy.io.loadmat('N_20_U_3_M_4_K_1_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_2_reward = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_3_reward = scipy.io.loadmat('N_20_U_3_M_4_K_3_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_4_reward = scipy.io.loadmat('N_20_U_3_M_4_K_4_reward.mat')['save_result1'][0]


# N_5_U_2_M_2_K_1_reward_optimal = scipy.io.loadmat('N_5_U_2_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_6_U_2_M_2_K_1_reward_optimal = scipy.io.loadmat('N_6_U_2_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_6_U_2_M_2_K_2_reward_optimal = scipy.io.loadmat('N_6_U_2_M_2_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_7_U_2_M_2_K_1_reward_optimal = scipy.io.loadmat('N_7_U_2_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_7_U_2_M_2_K_2_reward_optimal = scipy.io.loadmat('N_7_U_2_M_2_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)
# N_8_U_3_M_2_K_1_reward_optimal = scipy.io.loadmat('N_8_U_3_M_2_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num) 
# N_8_U_3_M_2_K_2_reward_optimal = scipy.io.loadmat('N_8_U_3_M_2_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_1_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_1_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_2_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_3_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_3_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_4_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_4_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  


# plt.plot(x_Num,U_2_M_3_K_1_reward,'-', label = 'U=2,M=3,K=1', LineWidth = 2)
# plt.plot(x_Num,U_2_M_3_K_2_reward,'-', label = 'U=2,M=3,K=2', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_1_reward,'-', label = 'U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,U_3_M_2_K_2_reward,'-', label = 'U=3,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_1_reward,'-', label = 'U=4,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,U_4_M_2_K_2_reward,'-', label = 'U=4,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,U_4_M_1_K_1_reward,'-', label = 'U=4,M=1,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward,'-.', label = 'N=7,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward,'-.', label = 'N=7,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward,'-.', label = 'N=8,U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward,'-.', label = 'N=8,U=3,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_1_reward,'--', label = '(20,3,4,1)', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_2_reward,'--', label = '(20,3,4,2)', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_3_reward,'--', label = '(20,3,4,3)', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_4_reward,'--', label = '(20,3,4,4)', LineWidth = 2)




# plt.plot(x_Num,N_5_U_2_M_2_K_1_reward_optimal,'r-', label = '(5,2,2,1),optimal')
# plt.plot(x_Num,N_6_U_2_M_2_K_1_reward_optimal,'g-', label = '(6,2,2,1),optimal')
# plt.plot(x_Num,N_6_U_2_M_2_K_2_reward_optimal,'c-', label = '(6,2,2,2),optimal')
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward_optimal,'m-', label = '(7,2,2,1),optimal')
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward_optimal,'y-', label = '(7,2,2,2),optimal')
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward_optimal,'b-', label = '(8,3,2,1),optimal')
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward_optimal,'k-', label = '(8,3,2,2),optimal')

# plt.plot(x_Num,U_2_M_3_K_1_reward_optimal,'--', label = 'U=2,M=3,K=1,optimal')
# plt.plot(x_Num,U_2_M_3_K_2_reward_optimal,'--', label = 'U=2,M=3,K=2,optimal')
# plt.plot(x_Num,U_3_M_2_K_1_reward_optimal,'--', label = 'U=3,M=2,K=1,optimal')
# plt.plot(x_Num,U_3_M_2_K_2_reward_optimal,'--', label = 'U=3,M=2,K=2,optimal')
# plt.plot(x_Num,U_4_M_2_K_1_reward_optimal,'--', label = 'U=4,M=2,K=1.optimal')
# plt.plot(x_Num,U_4_M_2_K_2_reward_optimal,'--', label = 'U=4,M=2,K=2,optimal')
# plt.plot(x_Num,U_4_M_1_K_1_reward_optimal,'--', label = 'U=4,M=1,K=1,optimal')
# plt.plot(x_Num,N_6_U_2_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward_optimal,'-')


# plt.plot(x_Num,N_20_U_3_M_4_K_1_reward_optimal,'-', label = '(20,3,4,1),optimal')
# plt.plot(x_Num,N_20_U_3_M_4_K_2_reward_optimal,'-', label = '(20,3,4,2),optimal')
# plt.plot(x_Num,N_20_U_3_M_4_K_3_reward_optimal,'-', label = '(20,3,4,3),optimal')
# plt.plot(x_Num,N_20_U_3_M_4_K_4_reward_optimal,'-', label = '(20,3,4,4),optimal')


plt.xlabel('slot number',fontsize=12)
plt.ylabel('average throughput',fontsize=12)
plt.legend(ncol = 1,  framealpha = 0)
# plt.xlim([-500,50000])
plt.grid()
# plt.xlim([-5000,175000])
# plt.ylim([1.4,3.2])

# plt.ylim([1.5,2.45])

# fig.savefig('3_temp.eps', dpi = 600, format = 'eps')


# fig.savefig('centralized_reward_comparison.eps', dpi = 600, format = 'eps')



 

