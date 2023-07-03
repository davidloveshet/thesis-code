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
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black", alpha = 0.5)

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
Sample_Num = 20000
x_Num = list(range(Sample_Num))
# axins = ax.inset_axes( ( 0.56, 0.08, 0.4, 0.19 ) )


 
N_20_U_3_M_4_K_2_regret = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret.mat')['save_result2'][0]
N_20_U_3_M_4_K_2_regret_DLF = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret_DLF.mat')['save_result2'][0]
N_20_U_3_M_4_K_2_regret_DLP = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret_DLP.mat')['save_result2'][0]

N_20_U_3_M_4_K_2_regret_random = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret_random.mat')['save_result2'][0]
N_20_U_3_M_4_K_2_regret_fixed_colli_alle = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret_fixed_colli_alle.mat')['save_result2'][0]
N_20_U_3_M_4_K_2_regret_greedy = scipy.io.loadmat('N_20_U_3_M_4_K_2_regret_greedy.mat')['save_result2'][0]


# plt.plot(x_Num,N_5_U_2_M_2_K_1_regret,'-', label = 'N=5,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_6_U_2_M_2_K_1_regret[0:20000],'-', label = 'N=6,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_6_U_2_M_2_K_2_regret,'-', label = 'N=6,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_1_regret,'-', label = 'N=7,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_2_regret,'-', label = 'N=7,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_1_regret,'-', label = 'N=8,U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_2_regret,'-', label = 'N=8,U=3,M=2,K=2', LineWidth = 2)
 

# plt.plot(x_Num,N_20_U_3_M_4_K_1_regret,'-', label = 'N=20,U=3,M=4,K=1', LineWidth = 2)
ax.plot(x_Num,N_20_U_3_M_4_K_2_regret,'--', label = '$\pi^D$-IID', LineWidth = 2)
ax.plot(x_Num,N_20_U_3_M_4_K_2_regret_DLF,'--', label = 'DLF', LineWidth = 2)
ax.plot(x_Num,N_20_U_3_M_4_K_2_regret_DLP,'--', label = 'DLP', LineWidth = 2)
ax.plot(x_Num,N_20_U_3_M_4_K_2_regret_random,'--', label = 'random', LineWidth = 2)
ax.plot(x_Num,N_20_U_3_M_4_K_2_regret_fixed_colli_alle,'--', label = 'fixed with collision alleviation', LineWidth = 2)
ax.plot(x_Num,N_20_U_3_M_4_K_2_regret_greedy,'--', label = '$\pi^D$-IID with greedy', LineWidth = 2)

# plt.plot(x_Num,N_20_U_3_M_4_K_3_regret,'-', label = 'N=20,U=3,M=4,K=3', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_4_regret,'-', label = 'N=20,U=3,M=4,K=4', LineWidth = 2)
 

plt.xlabel('slot number',fontsize=12)
plt.ylabel('normalized regret',fontsize=12)
# plt.ylim([ -500, 2800])
# plt.xlim([-500, 35000])
plt.legend(ncol=1,  framealpha = 0)
plt.grid()



# axins.plot(x_Num, N_20_U_3_M_4_K_2_regret_random_access, 'm-', alpha = 1 )
# axins.plot(x_Num, N_20_U_3_M_4_K_2_regret, 'b-', alpha = 1 )
# axins.plot(x_Num, N_20_U_3_M_4_K_2_regret_fixed_colli_alle, 'r-', alpha = 1 )
# zone_and_linked(ax, axins, 10000, 20000-1, x_Num, [  N_20_U_3_M_4_K_2_regret.tolist(), N_20_U_3_M_4_K_2_regret_fixed_colli_alle.tolist() ], 'bottom' )

# fig.savefig('decentralized_IID_figure_2_regret_comparison.eps', dpi = 600, format = 'eps')



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
Sample_Num = 20000
x_Num = list(range(Sample_Num))
axins = ax.inset_axes( ( 0.46, 0.34, 0.3, 0.29 ) )

# N_5_U_2_M_2_K_1_reward = scipy.io.loadmat('N_5_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_6_U_2_M_2_K_1_reward = scipy.io.loadmat('N_6_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_6_U_2_M_2_K_2_reward = scipy.io.loadmat('N_6_U_2_M_2_K_2_reward.mat')['save_result1'][0]
# N_7_U_2_M_2_K_1_reward = scipy.io.loadmat('N_7_U_2_M_2_K_1_reward.mat')['save_result1'][0]
# N_7_U_2_M_2_K_2_reward = scipy.io.loadmat('N_7_U_2_M_2_K_2_reward.mat')['save_result1'][0]
# N_8_U_3_M_2_K_1_reward = scipy.io.loadmat('N_8_U_3_M_2_K_1_reward.mat')['save_result1'][0]
# N_8_U_3_M_2_K_2_reward = scipy.io.loadmat('N_8_U_3_M_2_K_2_reward.mat')['save_result1'][0]
# N_20_U_3_M_4_K_1_reward = scipy.io.loadmat('N_20_U_3_M_4_K_1_reward.mat')['save_result1'][0]
N_20_U_3_M_4_K_2_reward = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward.mat')['save_result1'][0]
N_20_U_3_M_4_K_2_reward_DLF = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_DLF.mat')['save_result1'][0]
N_20_U_3_M_4_K_2_reward_DLP = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_DLP.mat')['save_result1'][0]

N_20_U_3_M_4_K_2_reward_random = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_random.mat')['save_result1'][0]
N_20_U_3_M_4_K_2_reward_fixed_colli_alle = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_fixed_colli_alle.mat')['save_result1'][0]
N_20_U_3_M_4_K_2_reward_greedy = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_greedy.mat')['save_result1'][0]
# N_20_U_3_M_4_K_2_reward_random_access = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_random_access.mat')['save_result1'][0]


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
N_20_U_3_M_4_K_2_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_2_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_3_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_3_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  
# N_20_U_3_M_4_K_4_reward_optimal = scipy.io.loadmat('N_20_U_3_M_4_K_4_reward_optimal.mat')['Reward_expect_all'][0] * np.ones(Sample_Num)  


# plt.plot(x_Num,N_5_U_2_M_2_K_1_reward,'-.', label = 'N=5,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_6_U_2_M_2_K_1_reward[0:20000],'-.', label = 'N=6,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_6_U_2_M_2_K_2_reward,'-.', label = 'N=6,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward,'-.', label = 'N=7,U=2,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward,'-.', label = 'N=7,U=2,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward,'-.', label = 'N=8,U=3,M=2,K=1', LineWidth = 2)
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward,'-.', label = 'N=8,U=3,M=2,K=2', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_1_reward,'-', label = 'N=20,U=3,M=4,K=1', LineWidth = 2)
plt.plot(x_Num, N_20_U_3_M_4_K_2_reward,'b--', label = '$\pi^D$-IID', LineWidth = 2)
plt.plot(x_Num, N_20_U_3_M_4_K_2_reward_DLF,'-', label = 'DLF', LineWidth = 2)
plt.plot(x_Num, N_20_U_3_M_4_K_2_reward_DLP,'-', label = 'DLP', LineWidth = 2)
plt.plot(x_Num, N_20_U_3_M_4_K_2_reward_random,'-', label = 'random', LineWidth = 2)
plt.plot(x_Num, N_20_U_3_M_4_K_2_reward_fixed_colli_alle,'r--', label = 'fixed with collision alleviation', LineWidth = 2)
plt.plot(x_Num, N_20_U_3_M_4_K_2_reward_greedy,'m--', label = '$\pi^D$-IID with greedy', LineWidth = 2)
# plt.plot(x_Num, N_20_U_3_M_4_K_2_reward_random_access,'-', label = 'proposed random access', LineWidth = 2)

# plt.plot(x_Num,N_20_U_3_M_4_K_3_reward,'-', label = 'N=20,U=3,M=4,K=3', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_4_reward,'-', label = 'N=20,U=3,M=4,K=4', LineWidth = 2)

axins.plot(x_Num, N_20_U_3_M_4_K_2_reward,'b--', label = '$\pi^D$-IID', LineWidth = 2)
axins.plot(x_Num, N_20_U_3_M_4_K_2_reward_fixed_colli_alle,'r--', label = 'fixed with collision alleviation', LineWidth = 2)
axins.plot(x_Num, N_20_U_3_M_4_K_2_reward_greedy,'m--', label = 'proposed with greedy', LineWidth = 2)

zone_and_linked(ax, axins, 8000, 20000-1, x_Num, [  N_20_U_3_M_4_K_2_reward, N_20_U_3_M_4_K_2_reward_fixed_colli_alle, N_20_U_3_M_4_K_2_reward_DLF, N_20_U_3_M_4_K_2_reward_greedy  ], 'bottom' )



# plt.plot(x_Num,N_5_U_2_M_2_K_1_reward_optimal,'r-', label = '(5,2,2,1),optimal')
# plt.plot(x_Num,N_6_U_2_M_2_K_1_reward_optimal,'g-', label = '(6,2,2,1),optimal')
# plt.plot(x_Num,N_6_U_2_M_2_K_2_reward_optimal,'c-', label = '(6,2,2,2),optimal')
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward_optimal,'m-', label = '(7,2,2,1),optimal')
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward_optimal,'y-', label = '(7,2,2,2),optimal')
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward_optimal,'b-', label = '(8,3,2,1),optimal')
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward_optimal,'k-', label = '(8,3,2,2),optimal')

# plt.plot(x_Num,N_5_U_2_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_6_U_2_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_6_U_2_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,N_7_U_2_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_7_U_2_M_2_K_2_reward_optimal,'-')
# plt.plot(x_Num,N_8_U_3_M_2_K_1_reward_optimal,'-')
# plt.plot(x_Num,N_8_U_3_M_2_K_2_reward_optimal,'-')


# plt.plot(x_Num,N_20_U_3_M_4_K_1_reward_optimal,'--', label = 'N=20,U=3,M=4,K=1,optimal', LineWidth = 2)
plt.plot(x_Num,N_20_U_3_M_4_K_2_reward_optimal,'--', label = '$\pi^{*D}$-IID', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_3_reward_optimal,'--', label = 'N=20,U=3,M=4,K=3,optimal', LineWidth = 2)
# plt.plot(x_Num,N_20_U_3_M_4_K_4_reward_optimal,'--', label = 'N=20,U=3,M=4,K=4,optimal', LineWidth = 2)


plt.xlabel('slot number',fontsize=12)
plt.ylabel('average throughput',fontsize=12)
plt.legend(ncol=2,  framealpha = 0, fontsize=9)
# plt.xlim([-500,50000])
# plt.ylim([-2,50])
plt.grid()


# fig.savefig('decentralized_IID_figure_2_reward_comparison.eps', dpi = 600, format = 'eps')



 

