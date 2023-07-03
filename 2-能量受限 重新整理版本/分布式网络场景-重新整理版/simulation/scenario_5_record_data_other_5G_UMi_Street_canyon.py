# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 01:28:03 2022

@author: xxx


"""


import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

 
fig, ax = plt.subplots()

N = 10
 

budget_list = [ 0,  10,  20,  30,  40,   50, 60, 70, 80, 90, 100, 110 ,120, 130, 140, 150, 160, 170, 180      ] # J 次要用户能量


# optimal_throuhgput_U_3 = np.array([133.05      , 143.        , 145.36666667, 148.2125    ,   148.78      , 150.9       , 151.234375  , 151.5775    ,    151.48958333, 151.35892857, 151.3796875 , 151.97638889,    152.1925    , 153.125     , 152.77395833, 152.09326923])

optimal_throughput_U_3_without_energy = 3.1112 * np.ones(len(budget_list))

optimal_throuhgput_U_3 = 2.81295 * np.ones(len(budget_list))

# average_throuhgput_U_3 = np.array([  0, 132.575     , 135.4       , 136.65833333, 142.39375   ,       142.225     , 146.07083333, 144.346875  , 145.545     ,       144.52291667, 144.60714286, 146.00625   , 147.04166667,       147.68      , 147.96477273, 147.52916667, 147.81826923])
average_throuhgput_U_3 = np.array([0, 2.574035  , 2.60443   , 2.67472833, 2.6756775 , 2.681472  ,
       2.6732225 , 2.69209786, 2.72032625, 2.70929556, 2.705031  ,
       2.70944409, 2.71834667, 2.70925462, 2.72704357, 2.73821733,
       2.735585  , 2.73728588, 2.74937972])

average_throuhgput_U_3_ONES = np.array([0, 2.51181   , 2.5443425 , 2.57577   , 2.61665875, 2.608702  ,
       2.63039333, 2.61338357, 2.65816062, 2.67177056, 2.6476985 ,
       2.66906909, 2.66158417, 2.66807577, 2.67774536, 2.68558733,
       2.68413656, 2.68310794, 2.69240611])


average_random = np.array([0, 1.812135  , 1.801205  , 1.81402833, 1.8216275 , 1.823622  ,  1.81735167, 1.81789429, 1.8181825 , 1.80494833, 1.8143585 ,  1.82240318, 1.82039875, 1.81267385, 1.82359357, 1.821404  ,   1.82088344, 1.81695941, 1.82371028])

average_random_without_allocation_energy = np.array([0, 1.66441176, 1.60360682, 1.58381362, 1.54244823, 1.57785894, 1.57518282, 1.57089628, 1.56120947, 1.57156694, 1.5811829 , 1.5799824 , 1.56979858, 1.55920852, 1.58080268, 1.57709763, 1.58098066, 1.55093345, 1.57109135])

average_EABS_UCB = np.array([ 0, 1.83446   , 1.835405  , 1.82653667, 1.799065  , 1.812697  ,  1.81766833, 1.80907286, 1.81467937, 1.82278722, 1.809371  ,  1.81467364, 1.81580708, 1.82063923, 1.81718107, 1.80842067,   1.81470844, 1.82025647, 1.81220472])

average_UCB_without_energy = np.array([0, 1.90571   , 1.906655  , 1.96967   , 1.9565275 , 1.960802  ,
       1.945285  , 1.94940143, 1.92517625, 1.92850111, 1.940281  ,
       1.92297364, 1.9297675 , 1.95524692, 1.95482929, 1.93534067,
       1.95925688, 1.92671235, 1.93187833])

 

average_throuhgput_U_3_greedy_01 = np.array([0, 2.416335  , 2.4868675 , 2.480295  , 2.49173375, 2.539447  , 2.5273975 , 2.53636571, 2.54243875, 2.55840389, 2.537926  ,  2.56189182, 2.54505083, 2.56693731, 2.58132036, 2.581974  ,  2.58593031, 2.57978147, 2.58043806])

# average_throuhgput_U_3_greedy_01_greedy = np.array([0, 115.2375    , 113.455     , 112.7975    , 112.599375  ,  115.378     , 116.39125   , 113.6915625 , 113.625     ,  112.975     , 112.56160714, 113.97640625, 114.14      ,    113.92175   , 112.98534091, 113.37864583, 113.315     ])


# plt.plot( budget_list, optimal_throuhgput_U_2, '-', label = 'U=2, Hungarian')
# plt.plot( budget_list, average_throuhgput_U_2, '--^', label = 'U=2, proposed')
# plt.plot( budget_list, optimal_throughput_U_3_without_energy, '-', label = 'Hungarian')
plt.plot( budget_list, optimal_throuhgput_U_3, '--', label = '$\pi^{*D}-$EC')
plt.plot( budget_list, average_throuhgput_U_3, '-^', label = '$\pi^{D}-$EC')
plt.plot( budget_list, average_random, '-*', label = 'random with channel allocation')
plt.plot( budget_list, average_random_without_allocation_energy, '-.', label = 'random without channel allocation')
# plt.plot( budget_list, average_throuhgput_U_3_TV, '--o', label = 'EABS-UCB')
plt.plot( budget_list, average_throuhgput_U_3_greedy_01, '-o', label = '$\epsilon$-greedy-decentralized')
plt.plot( budget_list, average_UCB_without_energy, '-x', label = 'UCB with Gale Shapley')
plt.plot( budget_list, average_throuhgput_U_3_ONES, '--*', label = 'decentralized ONES')
# plt.plot( budget_list, average_EABS_UCB, '--.', label = 'EABS-UCB-decentralized')
# plt.plot( budget_list, average_throuhgput_U_3_greedy_01_greedy, '--x', label = 'annealing greedy,$\epsilon$=0.1')



# plt.plot( budget_list, optimal_throuhgput_U_4, '-', label = 'U=4, Hungarian')
# plt.plot( budget_list, average_throuhgput_U_4, '--^', label = 'U=4, proposed')

# plt.plot( budget_list, optimal_throuhgput_U_5, '-', label = 'U=5, Hungarian')
# plt.plot( budget_list, average_throuhgput_U_5, '--^', label = 'U=5, proposed')

# plt.plot( budget_list, optimal_throuhgput_U_6, '-', label = 'U=6, Hungarian')
# plt.plot( budget_list, average_throuhgput_U_6, '--^', label = 'U=6, proposed')


# optimal_reward_average_U_3 = np.array([0.36975203, 0.36975203, 0.36975203, 0.36975203, 0.36975203, 0.36975203, 0.36975203, 0.36975203, 0.36975203, 0.36975203, 0.36975203]) * 100
# average_throuhgput_U_3 = np.array([0.29815995, 0.31169602, 0.32445394, 0.3433714 , 0.35522524, 0.3572314 , 0.36204172, 0.36403656, 0.36516788, 0.36590371, 0.36642088]) * 100

# optimal_reward_average_U_4_N_8 = np.array([0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464]) * 100
# average_throuhgput_U_4_N_8 = np.array([0.38935997, 0.40586003, 0.41907401, 0.43875134, 0.452966  , 0.45546518, 0.46135278, 0.46375612, 0.4651402 , 0.46595241, 0.46659456]) * 100


# optimal_reward_average_U_4_N_9 = np.array([0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464]) * 100
# average_throuhgput_U_4_N_9 = np.array([0.38651004, 0.40198403, 0.41622404, 0.43666137, 0.45089975, 0.4535386 , 0.45996198, 0.46269592, 0.46423106, 0.46522508, 0.46591248]) * 100

# optimal_reward_average_U_4_N_10 = np.array([0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464, 0.47042464]) * 100
# average_throuhgput_U_4_N_10 = np.array([0.38764984, 0.40221193, 0.41588203, 0.43616729, 0.45021577, 0.4526836 , 0.45904429, 0.46179153, 0.46337604, 0.46443393, 0.46518286]) * 100

# optimal_reward_average_U_5 = np.array([0.63944351, 0.63944351, 0.63944351, 0.63944351, 0.63944351, 0.63944351, 0.63944351, 0.63944351, 0.63944351, 0.63944351, 0.63944351]) * 100 # M/J
# average_throuhgput_U_5 = np.array([0.51931994, 0.54813195, 0.56966805, 0.59869335, 0.61664151, 0.61967081, 0.62700621, 0.63014292, 0.6319222 , 0.63304904, 0.63387818]) * 100 # M/J

# optimal_reward_average_U_6 = np.array([0.81388872, 0.81388872, 0.81388872, 0.81388872, 0.81388872, 0.81388872, 0.81388872, 0.81388872, 0.81388872, 0.81388872, 0.81388872]) * 100
# average_throuhgput_U_6 = np.array([0.65725989, 0.69769996, 0.72892603, 0.76665337, 0.78851075, 0.79202738, 0.80055981, 0.80404613, 0.80598595, 0.80723421, 0.80811196]) * 100


# sum_reward_set_N_6_M_1 = np.array([  414.959,  1265.607,  3406.171,  4263.86 ,  8559.036, 12856.465,       17155.422, 21454.871, 25754.918, 30054.481])
# sum_reward_set_N_6_M_2 = np.array([  383.698,  1174.007,  3226.343,  4063.799,  8241.519, 12421.151,       16600.783, 20780.415, 24958.135, 29137.767])
# sum_reward_set_N_6_M_3 = np.array([  378.095,  1130.077,  3037.241,  3806.113,  7695.005, 11621.937,       15592.225, 19585.048, 23599.078, 27631.328])

# optimal_reward = np.array([  430.18153661,  1290.54460983,  3441.45229287,  4301.81536608,        8603.63073217, 12905.44609825, 17207.26146434, 21509.07683042,       25810.89219651, 30112.70756259])

# plt.plot(budget[0:7], optimal_reward_average_U_2[0:7], '--^', label = 'U=2, Hungarian')
# plt.plot(budget[0:7], average_throuhgput_U_2[0:7], '--^', label = 'U=2, proposed')

# plt.plot(budget[0:7], optimal_reward_average_U_3[0:7], '--^', label = 'U=3, Hungarian')
# plt.plot(budget[0:7], average_throuhgput_U_3[0:7], '--^', label = 'U=3, proposed')

# plt.plot(budget[0:7], optimal_reward_average_U_4_N_9[0:7], '--^', label = 'U=4, Hungarian')

# plt.plot(budget[0:9], optimal_reward_average_U_4_N_10[0:9], '--^', label = 'U=4, Hungarian')


# plt.plot(budget[0:9], average_throuhgput_U_4_N_8[0:9], '--^', label = 'U=4, proposed')

# plt.plot(budget[0:9], average_throuhgput_U_4_N_9[0:9], '--^', label = 'U=4, proposed')


# plt.plot(budget[0:9], average_throuhgput_U_4_N_10[0:9], '--^', label = 'U=4, proposed')

# plt.plot(budget[0:7], optimal_reward_average_U_5[0:7], '--^', label = 'U=5, Hungarian')
# plt.plot(budget[0:7], average_throuhgput_U_5[0:7], '--^', label = 'U=5, proposed')

# plt.plot(budget[0:7], optimal_reward_average_U_6[0:7], '--^', label = 'U=6, Hungarian')
# plt.plot(budget[0:7], average_throuhgput_U_6[0:7], '--^', label = 'U=6, proposed')


# plt.plot(budget[0:4], sum_reward_set_M_1[0:4], 'b--.', label = 'N=10,M=1')
# plt.plot(budget[0:4], sum_reward_set_M_2[0:4], 'b--*', label = 'N=10,M=2')
# plt.plot(budget[0:4], sum_reward_set_M_3[0:4], 'b--o', label = 'N=10,M=3')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_1[0:4], 'r--.', label = 'N=8,M=1')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_2[0:4], 'r--*', label = 'N=8,M=2')
# plt.plot(budget[0:4], sum_reward_set_N_8_M_3[0:4], 'r--o', label = 'N=8,M=3')
# plt.plot(budget[0:4], optimal_reward[0:4], 'k-', label = 'maximum')





plt.xlabel('energy (J)',fontsize=12)
plt.ylabel('throughput/energy (M/J)',fontsize=12)
# plt.xlim([-1,450])
# plt.ylim([ -0.5, 0 ])
# plt.legend(ncol=2, fontsize=9, framealpha = 0)
plt.legend(  fontsize= 10, framealpha = 0)
plt.grid()
# fig.savefig('average_5G_decentralized_comparison_.eps', dpi = 600, format = 'eps')


























