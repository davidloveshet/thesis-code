# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 01:28:03 2022

@author: xxx



"""


import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

 
import numpy as np
import random 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()



budget_list = [   0,  3, 5, 10, 20, 25, 30, 35, 40,   50, 60, 70     ]

# optimal_reward_average_U_2 = np.array([0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168]) * 100
optimal_throuhgput_U_2 = 8.0788025 * np.ones( len(budget_list) )
average_throuhgput_U_2 = np.array([0, 7.39736667, 7.64182   , 7.75391   , 7.889755  , 7.912364  ,  8.04523667, 7.86600286, 7.9842775 , 8.031302  , 8.02655167,    7.97783   ])

optimal_throuhgput_U_3 = 11.40 * np.ones( len(budget_list) )
average_throuhgput_U_3 = np.array([0 ,10.5577    , 10.92122   , 10.92311   , 11.202405  , 11.207724  ,  11.18783667, 11.23226   , 11.2708025 , 11.281822  , 11.261935  ,   11.23551571])

optimal_throuhgput_U_4 = 14.2476275 * np.ones( len(budget_list) )
average_throuhgput_U_4 = np.array([  0, 13.1417    , 13.44442   , 13.65151   , 13.875705  , 13.864684  ,  13.89090333, 13.96826   , 13.9460025 , 13.961202  , 14.02105167,    14.08334429])


optimal_throuhgput_U_5 = 16.677822 * np.ones( len(budget_list) )
average_throuhgput_U_5 = np.array([0, 15.78903333, 15.88782   , 16.03981   , 16.187055  , 16.221444  ,  16.29693667, 16.42197429, 16.3219525 , 16.349122  , 16.457485  ,    16.42088714])

optimal_throuhgput_U_6 = 18.917922 * np.ones( len(budget_list) )
average_throuhgput_U_6 = np.array([0, 17.3787    , 17.78782   , 18.20011   , 18.259955  , 18.302324  ,  18.43887   , 18.52988857, 18.4637275 , 18.540962  , 18.53291833,   18.59910143])



plt.plot( budget_list, optimal_throuhgput_U_2, '-', label = 'U=2,$\pi^{*C}-$EC')
plt.plot( budget_list, optimal_throuhgput_U_3, '-', label = 'U=3,$\pi^{*C}-$EC')
plt.plot( budget_list, optimal_throuhgput_U_4, '-', label = 'U=4,$\pi^{*C}-$EC')
plt.plot( budget_list, optimal_throuhgput_U_5, '-', label = 'U=5,$\pi^{*C}-$EC')
plt.plot( budget_list, optimal_throuhgput_U_6, '-', label = 'U=6,$\pi^{*C}-$EC')



plt.plot( budget_list, average_throuhgput_U_2, '--^', label = 'U=2,$\pi^{D}-$EC')
plt.plot( budget_list, average_throuhgput_U_3, '--^', label = 'U=3,$\pi^{D}-$EC')
plt.plot( budget_list, average_throuhgput_U_4, '--^', label = 'U=4,$\pi^{D}-$EC')
plt.plot( budget_list, average_throuhgput_U_5, '--^', label = 'U=5,$\pi^{D}-$EC')
plt.plot( budget_list, average_throuhgput_U_6, '--^', label = 'U=6,$\pi^{D}-$EC')





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
# plt.xlim([-1,42])
plt.ylim([-4, 20 ])
plt.legend(ncol=2, fontsize=11, framealpha = 0)
# plt.legend( fontsize=12, framealpha = 0)
plt.grid()
fig.savefig('average_energy_efficiency_Rayleigh_.eps', dpi = 600, format = 'eps')


























