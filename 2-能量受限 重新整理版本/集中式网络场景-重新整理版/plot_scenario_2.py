# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 01:28:03 2022

@author: xxx

This is used to record the simulation data for

identical energy constraint, rate requirement

"""


import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

 
import numpy as np
import random 
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

N = 10
 

budget_list = [  0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.4, 4.8  ] # J

# optimal_reward_average_U_2 = np.array([0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168, 0.26497168]) * 100
optimal_throuhgput_U_2 = 301.637 * np.ones( len(budget_list) )
average_throuhgput_U_2 = np.array([0, 275.90625   , 285.38125   , 288.93541667, 289.2578125 ,  288.43      , 291.06458333, 292.903125  , 293.685625  ,  295.7609375 , 297.89642857, 296.90039063, 296.825     , 296.3609375 , 298.37244318, 297.55989583])
average_throuhgput_U_2 = np.array([0, 271.24333333, 281.9375    , 286.52611111, 288.25041667,  289.836     , 291.27833333, 294.161875  , 294.6855    ,  295.31430556, 295.47166667, 296.0409375 , 296.95342593 , 296.95342593 , 296.95342593 , 296.95342593])


optimal_throuhgput_U_3 = 410.946875 * np.ones( len(budget_list) )
average_throuhgput_U_3 = np.array([0, 380.525     , 389.525     , 393.63333333, 392.56041667, 396.38166667, 393.01805556, 400.39166667, 403.66      , 400.05972222, 403.65178571, 402.78333333, 405.07222222,  403.9425    , 403.86022727, 403.11215278])
average_throuhgput_U_3 = np.array([0, 364.565     , 372.2825    , 385.24166667, 384.7625    , 386.223     , 391.07583333, 393.436875  , 396.174     , 397.31      , 398.78642857, 402.78333333, 405.07222222,  403.9425    , 403.86022727, 403.11215278])



optimal_throuhgput_U_4 = 490.628125 * np.ones( len(budget_list) )
average_throuhgput_U_4 = np.array([0, 452.0125    , 462.9125    , 466.8625    , 469.134375  , 468.8825    , 466.69583333, 473.81875   , 477.855     , 477.89375   , 480.87321429, 481.13359375, 481.41527778,  482.163125  , 482.01931818, 482.61197917])
average_throuhgput_U_4 = np.array([0, 442.37      , 455.1225    , 460.76666667, 463.63625   ,  464.598    , 472.49083333, 472.904375  , 476.24      , 480.69625   , 480.87321429, 481.13359375, 481.41527778,  482.163125  , 482.01931818, 482.61197917])


optimal_throuhgput_U_5 = 563.659375 * np.ones( len(budget_list) )
average_throuhgput_U_5 = np.array([0, 531.21875   , 529.29375   , 534.23333333, 540.1171875 ,  545.76125   , 544.971875  , 548.334375  , 553.0475    ,    554.48072917, 554.60892857, 556.14648437, 558.57638889, 558.57638889, 558.57638889, 558.57638889    ])
average_throuhgput_U_5 = np.array([0, 499.655     , 519.6275    , 518.71666667, 528.52125   ,   531.991     , 535.39666667, 539.831875  , 543.291     ,   543.93458333, 545.72428571, 546.92109375, 549.27430556, 552.463125  , 553.5931818, 554.5931818  ])


optimal_throuhgput_U_6 = 612.35 * np.ones( len(budget_list) )  # 622.35 !!!
average_throuhgput_U_6 = np.array([0, 575.0375    , 577.921875  , 590.75833333, 593.1390625 ,    592.88125   , 599.37916667, 599.575     , 603.23125   ,    607.96458333, 609.56428571, 609.94257813, 610.24257813, 611.24257813, 613.52414773, 614.80381944])
average_throuhgput_U_6 = np.array([0, 545.765625  , 559.8125    , 568.96770833, 574.24296875   ,    582.3125    , 582.47708333, 590.3125    , 595.465     ,    599.98854167, 600.1321428,  601.65,       600.60069444,  601.65,      601.65,      601.65 ])


plt.plot( budget_list, optimal_throuhgput_U_2, '-', label = 'U=2,optimal')
plt.plot( budget_list, optimal_throuhgput_U_3, '-', label = 'U=3,optimal')
plt.plot( budget_list, optimal_throuhgput_U_4, '-', label = 'U=4,optimal')
plt.plot( budget_list, optimal_throuhgput_U_5, '-', label = 'U=5,optimal')
plt.plot( budget_list, optimal_throuhgput_U_6, '-', label = 'U=6,optimal')

plt.plot( budget_list, average_throuhgput_U_2, '--^', label = 'U=2,proposed')
plt.plot( budget_list, average_throuhgput_U_3, '--^', label = 'U=3,proposed')
plt.plot( budget_list, average_throuhgput_U_4, '--^', label = 'U=4,proposed')
plt.plot( budget_list, average_throuhgput_U_5, '--^', label = 'U=5,proposed')
plt.plot( budget_list, average_throuhgput_U_6, '--^', label = 'U=6, proposed')


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
plt.xlabel('energy (J)',fontsize=14)
plt.ylabel('throughput/energy (M/J)',fontsize=14)
# plt.xlim([-1,42])
# plt.ylim([0,80])
plt.legend(ncol=2, fontsize=11.3, framealpha = 0)
# plt.legend( fontsize=12, framealpha = 0)
plt.grid()
# fig.savefig('average_throughput_channel_Model_centralized.eps', dpi = 600, format = 'eps')


























