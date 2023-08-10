# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[11] s = 1,2,3 regret 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
feature_generation = np.array([   [1], [0]  ])
T_simulation = 10000
regret_x_label = [ round(0.001 * T_simulation), round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

# ---------------------------------------------------------------------------------------------------------
 #   proposed
cumulative_utility_regret_y_label_s_1_M_3 = np.array([  5.43333333,  20.9       ,  32.36666667,  68.23333333,
        94.66666667, 128.23333333, 180.53333333, 213.53333333,
       244.96666667, 261.83333333, 279.73333333, 294.4       ,
       314.03333333, 329.66666667, 343.66666667])
 

cumulative_utility_regret_y_label_s_2_M_3 = np.array([  4.61883833,  14.90558333,  27.53892833,  62.27348333,
        86.74787833, 126.38985167, 178.34382333, 220.253555  ,
       248.19517667, 273.98271333, 293.02606   , 316.37105   ,
       333.80785   , 348.01732667, 360.53209667])
 
cumulative_utility_regret_y_label_s_3_M_3 = np.array([  4.2977538,  17.4257724,  32.5661976,  68.0216058,  93.9057084,
       135.2922714, 196.453593 , 230.168121 , 262.5575436, 286.1195298,
       305.6977146, 332.6990352, 343.3380618, 362.7012882, 379.861191 ])

# ---------------------------------------------------------------------------------------------------------
#   random 

cumulative_utility_regret_y_label_random_s_1 = np.array([   7.86666667,   33.1       ,   64.76666667,  193.63333333,
        321.36666667,  641.73333333, 1269.7       , 1896.33333333,
       2524.66666667, 3156.86666667, 3799.26666667, 4436.33333333,
       5082.        , 5717.63333333, 6347.16666667])


cumulative_utility_regret_y_label_random_s_2 = np.array([   6.52107333,   26.960065  ,   50.14494833,  146.96558667,
        241.208055  ,  486.24777167,  969.196015  , 1449.48146   ,
       1933.24583667, 2412.35437333, 2898.50425   , 3380.44962667,
       3855.51031667, 4339.86421   , 4827.21349   ])

cumulative_utility_regret_y_label_random_s_3 = np.array([   5.7270386,   26.3055342,   50.4299006,  146.1962248,
        246.8151406,  490.1079804,  974.037321 , 1465.2983458,
       1951.4941776, 2441.493736 , 2931.3410322, 3418.2241652,
       3903.8689304, 4389.5056818, 4873.5326022])


# ---------------------------------------------------------------------------------------------------------
#   fixed 

cumulative_utility_regret_y_label_fixed_1 = np.array([  11.06666667,   49.33333333,   97.46666667,  288.3       ,
        478.06666667,  953.06666667, 1910.03333333, 2860.26666667,
       3810.96666667, 4750.9       , 5702.1       , 6650.46666667,
       7600.16666667, 8554.9       , 9499.86666666])

cumulative_utility_regret_y_label_fixed_2 = np.array([   7.30104167,   30.97236167,   60.36749167,  179.14446833,
        301.15511167,  601.21804167, 1196.170035  , 1803.959685  ,
       2402.339365  , 3008.673865  , 3611.38665833, 4218.56191167,
       4821.59807167, 5419.751395  , 6019.198185  ])


cumulative_utility_regret_y_label_fixed_3 = np.array([   2.147227,    8.511127,   16.430647,   51.597087,   82.780197,
        166.194427,  330.878017,  503.080437,  673.939367,  841.946327,
       1007.549147, 1173.434807, 1341.771747, 1511.664307, 1683.395327])


# ---------------------------------------------------------------------------------------------------------
#   greedy 


cumulative_utility_regret_y_label_greedy_s_1 = np.array([  6.63333333,  20.9       ,  34.86666667,  77.03333333,
       105.43333333, 163.7       , 256.23333333, 333.73333333,
       406.96666667, 471.56666667, 548.16666667, 621.83333333,
       689.3       , 756.66666667, 824.4       ])

cumulative_utility_regret_y_label_greedy_s_2 = np.array([  4.79087833,  16.75555333,  28.60944   ,  66.86035667,
        94.25196167, 143.05974667, 221.99937167, 283.54516167,
       346.94022833, 402.06514833, 458.04009833, 512.48032   ,
       560.73713833, 612.92652333, 672.82219667])
 
cumulative_utility_regret_y_label_greedy_s_3 = np.array([  4.4533158,  17.2871808,  30.7786488,  71.8753008,  98.9228186,
       152.6327204, 231.0585956, 292.6366348, 347.0432658, 398.6365816,
       447.549517 , 496.2814348, 544.6334042, 590.8782156, 641.4514218])
 
plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_1_M_3,'-*', label = 's=1,proposed' )
plt.plot(regret_x_label[0:6], cumulative_utility_regret_y_label_fixed_1[0:6],'-x', label = 's=1,fixed' )
plt.plot(regret_x_label[0:7], cumulative_utility_regret_y_label_random_s_1[0:7],'-o', label = 's=1,random' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_greedy_s_1,'--o', label = 's=1,greedy' )

plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_2_M_3,'-*', label = 's=2,proposed' )
plt.plot(regret_x_label[0:7], cumulative_utility_regret_y_label_fixed_2[0:7],'-x', label = 's=2,fixed' )
plt.plot(regret_x_label[0:7], cumulative_utility_regret_y_label_random_s_2[0:7],'--o', label = 's=2,random' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_greedy_s_2,'--o', label = 's=2,greedy' )

plt.plot(regret_x_label, cumulative_utility_regret_y_label_s_3_M_3,'-*', label = 's=3,proposed' )
plt.plot(regret_x_label[0:10], cumulative_utility_regret_y_label_fixed_3[0:10],'-x', label = 's=3,fixed' )
plt.plot(regret_x_label[0:7], cumulative_utility_regret_y_label_random_s_3[0:7],'--o', label = 's=3,random' )
plt.plot(regret_x_label, cumulative_utility_regret_y_label_greedy_s_3,'--o', label = 's=3,greedy' )

plt.legend(ncol=1,framealpha = 0,fontsize=11)
plt.xlabel('time',fontsize=16)
plt.ylabel('utility loss',fontsize=16)
# plt.title('s=2',fontsize=16)
plt.xlim([-500, 16200])
plt.grid()
# fig.savefig('regret_QoS.eps', dpi = 600, format = 'eps')

 

# In[] average throughput 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
feature_generation = np.array([   [1], [0]  ])
T_simulation = 10000
regret_x_label = [ round(0.001 * T_simulation), round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

# # --------------------------------------------------------------------------------------------------------- 
# proposed 

optimal_throughput_M_3_s_1 = 2.4 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_1 = np.array([1.85666667, 1.982,      2.07633333, 2.17255556, 2.21066667, 2.27176667,
 2.30973333, 2.32882222, 2.33875833, 2.34763333, 2.35337778, 2.35794286,
 2.36074583, 2.36337037, 2.36563333])


optimal_throughput_M_3_s_2 = 2.29 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_2 = np.array([1.76      , 1.96533333, 2.032     , 2.11066667, 2.1496    ,
       2.20043333, 2.23335   , 2.25178889, 2.26024167, 2.26739333,
       2.27121111, 2.27421429, 2.27567917, 2.27761852, 2.27981333])


optimal_throughput_M_3_s_3 = 1.97 * np.ones(len(regret_x_label)) # M bits 
average_throughput_M_3_s_3 = np.array([1.65      , 1.80733333, 1.805     , 1.84488889, 1.85706667,
       1.87426667, 1.89263333, 1.90535556, 1.91119167, 1.91796667,
       1.92177778, 1.9264619 , 1.930125  , 1.93344444, 1.93594   ])


 
# ---------------------------------------------------------------------------------------------------------
# random 
 
average_throughput_random_s_1 = np.array([1.61333333, 1.738     , 1.75233333, 1.75455556, 1.75726667,
       1.75826667, 1.76515   , 1.76788889, 1.76883333, 1.76862667,
       1.76678889, 1.7662381 , 1.76475   , 1.76470741, 1.76528333])


average_throughput_random_s_2 = np.array([1.55666667, 1.69866667, 1.74133333, 1.75388889, 1.76106667,
       1.75783333, 1.75981667, 1.76151111, 1.76148333, 1.76244   ,
       1.76181667, 1.76211429, 1.76308333, 1.76280741, 1.76223667])

average_throughput_random_s_3 = np.array([1.51666667, 1.68933333, 1.72766667, 1.756     , 1.74753333,
       1.75743333, 1.76221667, 1.75984444, 1.76039167, 1.76015333,
       1.7597    , 1.7605619 , 1.76110833, 1.76134815, 1.76158333])


# ---------------------------------------------------------------------------------------------------------
#   fixed 


average_throughput_fixed_1 = np.array([1.29333333, 1.41333333, 1.42533333, 1.439     , 1.44386667,
       1.44693333, 1.44498333, 1.44657778, 1.44725833, 1.44982   ,
       1.44965   , 1.44993333, 1.44997917, 1.44945556, 1.45001333])

average_throughput_fixed_2 = np.array([1.30666667, 1.42933333, 1.44666667, 1.45411111, 1.44893333,
       1.45016667, 1.45345   , 1.45013333, 1.4509    , 1.44972   ,
       1.44955556, 1.44878095, 1.44873333, 1.44925556, 1.44954   ])

average_throughput_fixed_3 = np.array([1.34333333, 1.44066667, 1.45333333, 1.44533333, 1.455     ,
       1.45453333, 1.45581667, 1.4527    , 1.45161667, 1.45177333,
       1.45244444, 1.45286667, 1.45275   , 1.45241481, 1.45188667])


# ---------------------------------------------------------------------------------------------------------
#   greedy 
average_throughput_greedy_s_1 = np.array([1.73666667, 1.982     , 2.05133333, 2.14322222, 2.18913333,
       2.2363    , 2.27188333, 2.28875556, 2.29825833, 2.30568667,
       2.30863889, 2.31116667, 2.3138375 , 2.31592593, 2.31756   ])

average_throughput_greedy_s_2 = np.array([1.75666667, 1.94733333, 2.00533333, 2.07866667, 2.1192    ,
       2.17093333, 2.20675   , 2.22285556, 2.22895833, 2.23376667,
       2.23645556, 2.23841429, 2.24025833, 2.24112593, 2.24137333])

average_throughput_greedy_s_3 = np.array([1.58666667, 1.77866667, 1.80833333, 1.83255556, 1.84873333,
       1.8682    , 1.89238333, 1.90567778, 1.91510833, 1.92162667,
       1.9264    , 1.92976667, 1.93223333, 1.93430741, 1.93540667])
 
plt.plot(regret_x_label, optimal_throughput_M_3_s_1,'-', label = 's=1,optimal', linewidth = 2 )
plt.plot(regret_x_label, average_throughput_M_3_s_1,'-^', label = 's=1,proposed' )
plt.plot(regret_x_label[3:15], average_throughput_random_s_1[3:15],'--^', label = 's=1,random' )
plt.plot(regret_x_label, average_throughput_greedy_s_1,'--o', label = 's=1,greedy' )
plt.plot(regret_x_label[3:15], average_throughput_fixed_1[3:15],'-x', label = 's=1,fixed' )

plt.plot(regret_x_label, optimal_throughput_M_3_s_2,'-', label = 's=2,optimal', linewidth = 2 )
plt.plot(regret_x_label, average_throughput_M_3_s_2,'-*', label = 's=2,proposed' )
plt.plot(regret_x_label[3:15], average_throughput_random_s_2[3:15],'--^', label = 's=2,random' )
plt.plot(regret_x_label, average_throughput_greedy_s_2,'--o', label = 's=2,greedy' )
plt.plot(regret_x_label[3:15], average_throughput_fixed_2[3:15],'-x', label = 's=2,fixed' )

plt.plot(regret_x_label, optimal_throughput_M_3_s_3,'-', label = 's=3,optimal', linewidth = 2 )
plt.plot(regret_x_label, average_throughput_M_3_s_3,'-o', label = 's=3,proposed' )
plt.plot(regret_x_label[3:15], average_throughput_random_s_3[3:15],'--^', label = 's=3,random' )
plt.plot(regret_x_label, average_throughput_greedy_s_3,'--o', label = 's=3,greedy' )
plt.plot(regret_x_label[3:15], average_throughput_fixed_3[3:15],'-x', label = 's=3,fixed' )

 
plt.legend(ncol= 1, framealpha = 0, fontsize = 9.5)
plt.xlabel('time',fontsize = 16)
plt.ylabel('average throughput (Mbps)',fontsize = 16)
# plt.title('s=1',fontsize = 14)
# plt.ylim([0,2.6])
plt.xlim([-500, 15000])
plt.grid()
# fig.savefig('QoS_average_throughput.eps', dpi = 600, format = 'eps')
 



# In[] average cost 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

regret_x_label = [ round(0.001 * T_simulation), round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

# ---------------------------------------------------------------------------------------------------------
# proposed
average_cost_optimal_s_1 = 1.5387 * np.ones(len(regret_x_label))
average_cost_M_3_s_1 = np.array([0.8852    , 1.04528   , 1.12742   , 1.25148667, 1.313988  ,
       1.397996  , 1.451426  , 1.47218   , 1.484011  , 1.4947716 ,
       1.50093567, 1.50857829, 1.51285025, 1.51544956, 1.5185936 ])

average_cost_optimal_s_2 = 1.0409 * np.ones(len(regret_x_label))
average_cost_M_3_s_2 = np.array([0.87356667, 0.96719333, 1.00131   , 1.07263778, 1.096064  ,
       1.11944233, 1.1377175 , 1.13895267, 1.14372225, 1.143809  ,
       1.14055544, 1.13678276, 1.13265504, 1.12888963, 1.12581547])


average_cost_optimal_s_3 = 0.4957 * np.ones(len(regret_x_label))
average_cost_M_3_s_3 = np.array([0.6605    , 0.67382   , 0.64947   , 0.61541667, 0.590224  ,
       0.557217  , 0.5261815 , 0.51864567, 0.51511675, 0.5112412 ,
       0.50956183, 0.50695614, 0.50566513, 0.50435489, 0.5033764 ])

# ---------------------------------------------------------------------------------------------------------
# random 
average_cost_M_3_random_s_1 = np.array([0.7314    , 0.77272667, 0.78293667, 0.78446556, 0.78638067,
       0.790544  , 0.79352883, 0.794     , 0.79462333, 0.7943544 ,
       0.79339556, 0.79304962, 0.79214875, 0.79212833, 0.7924439 ])

average_cost_M_3_random_s_2 = np.array([0.6985    , 0.75910667, 0.78069333, 0.78636111, 0.786452  ,
       0.785901  , 0.78778817, 0.78839044, 0.78844567, 0.78927107,
       0.78902139, 0.78949838, 0.78970292, 0.78969237, 0.78941223])

average_cost_M_3_random_s_3 = np.array([0.6717    , 0.75790667, 0.77539667, 0.78839778, 0.78529533,
       0.791789  , 0.7934645 , 0.791616  , 0.79178292, 0.79168527,
       0.79135689, 0.79203619, 0.79225242, 0.79225385, 0.79190083])



# ---------------------------------------------------------------------------------------------------------
#   fixed 
average_cost_M_3_fixed_1 = np.array([0.20126667, 0.22173333, 0.22322   , 0.22679   , 0.227312  ,
       0.22773267, 0.22779983, 0.22802578, 0.22811425, 0.2284702 ,
       0.22849539, 0.22856171, 0.22854229, 0.228439  , 0.22851413])

average_cost_M_3_fixed_2 = np.array([0.20706667, 0.22522667, 0.2274    , 0.22901889, 0.228096  ,
       0.22821167, 0.2289195 , 0.22841911, 0.2285315 , 0.22842453,
       0.22835722, 0.22823019, 0.22821733, 0.22834478, 0.22836773])

average_cost_M_3_fixed_3 = np.array([0.2091    , 0.22747333, 0.2294    , 0.22827556, 0.22933667,
       0.22904867, 0.22931817, 0.22892478, 0.22877283, 0.22874507,
       0.22883444, 0.22889105, 0.22888458, 0.22881341, 0.22874387])


# ---------------------------------------------------------------------------------------------------------
#   greedy 

average_cost_M_3_greedy_s_1 = np.array([0.88436667, 1.05743333, 1.12218   , 1.23109   , 1.286854  ,
       1.35380833, 1.3969115 , 1.41791711, 1.42928108, 1.43702913,
       1.44139483, 1.44470519, 1.44846921, 1.45153889, 1.45381007])

average_cost_M_3_greedy_s_2 = np.array([0.82236667, 0.96194   , 0.99684667, 1.03678   , 1.065484  ,
       1.09599933, 1.11613617, 1.11824944, 1.11326742, 1.10885287,
       1.10457144, 1.10134414, 1.0975785 , 1.09413593, 1.09220853])

average_cost_M_3_greedy_s_3 = np.array([0.62366667, 0.67      , 0.65600333, 0.60939222, 0.58094067,
       0.555604  , 0.54089517, 0.53496389, 0.53285142, 0.53186587,
       0.530998  , 0.53001948, 0.52940092, 0.52889226, 0.52837587])

plt.plot(regret_x_label, average_cost_optimal_s_1,'-', label = 's=1,optimal' ) 
plt.plot(regret_x_label, average_cost_M_3_s_1,'-^', label = 's=1,proposed' )
plt.plot(regret_x_label, average_cost_M_3_fixed_1,'-x', label = 's=1,fixed' )
plt.plot(regret_x_label[3:15], average_cost_M_3_random_s_1[3:15],'--^', label = 's=1,random' )
plt.plot(regret_x_label, average_cost_M_3_greedy_s_1,'--o', label = 's=1,greedy' )


plt.plot(regret_x_label, average_cost_optimal_s_2,'-', label = 's=2,optimal' ) 
plt.plot(regret_x_label, average_cost_M_3_s_2,'-*', label = 's=2,proposed' )
plt.plot(regret_x_label, average_cost_M_3_fixed_2,'-x', label = 's=2,fixed' )
plt.plot(regret_x_label[3:15], average_cost_M_3_random_s_2[3:15],'--^', label = 's=2,random' )
plt.plot(regret_x_label, average_cost_M_3_greedy_s_2,'--o', label = 's=2,greedy' )


plt.plot(regret_x_label, average_cost_optimal_s_3,'-', label = 's=3,optimal' ) 
plt.plot(regret_x_label, average_cost_M_3_s_3,'-o', label = 's=3,proposed' )
plt.plot(regret_x_label, average_cost_M_3_fixed_3,'-x', label = 's=3,fixed' )
plt.plot(regret_x_label[3:15], average_cost_M_3_random_s_3[3:15],'--^', label = 's=3,random' )
plt.plot(regret_x_label, average_cost_M_3_greedy_s_3,'--o', label = 's=3,greedy' )

plt.legend(ncol=1,framealpha = 0,fontsize = 9.5)
plt.xlabel('time',fontsize = 16)
plt.ylabel('average price',fontsize = 16)
# plt.ylim([0.05, 1.6])
plt.xlim([-500, 15000])
plt.grid()
# fig.savefig('QoS_average_cost.eps', dpi = 600, format = 'eps')























































 
