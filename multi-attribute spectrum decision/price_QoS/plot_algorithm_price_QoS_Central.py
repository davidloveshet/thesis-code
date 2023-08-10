# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[11] we plot the relationship between the regret and time slots 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]
  
cumulative_regret_y_label_U_3_M_1_K_1_Homo = np.array([  45.2353709 ,   81.9593425 ,  202.07105483,  294.75639507,
        370.6394314 ,  466.16679513,  697.7410112 ,  868.71386393,
        991.29339287, 1101.62269067, 1196.090539  , 1278.65404663,
       1358.09239   , 1430.24103137, 1493.47012613])
 
cumulative_regret_y_label_U_3_M_1_K_1_Homo_random = np.array([   59.76863125,   118.28747487,   352.95959137,   587.21716013,
         821.846737  ,  1175.28284813,  2347.41869587,  3524.04378637,
        4705.73753725,  5882.64661687,  7059.6516515 ,  8231.25706587,
        9405.08287575, 10588.35826563, 11766.91462825])

cumulative_regret_y_label_U_3_M_1_K_1_Homo_fixed = np.array([   68.051202  ,   135.122002  ,   403.405202  ,   671.688402  ,
         939.971602  ,  1342.396402  ,  2683.812402  ,  4025.228402  ,
        5366.644402  ,  6708.060402  ,  8049.476402  ,  9390.892402  ,
       10732.308402  , 12073.724402  , 13415.14040199])

cumulative_regret_y_label_U_3_M_2_K_2_Homo = np.array([  32.9737106 ,   57.64038593,  137.62383733,  199.73591607,
        254.657902  ,  326.28986   ,  518.44387787,  657.14013173,
        775.3826228 ,  876.23089147,  966.75694133, 1041.69599127,
       1116.82123827, 1183.1064984 , 1244.22436253])
 
cumulative_regret_y_label_U_3_M_2_K_2_Homo_random = np.array([   63.64258853,   125.06615533,   371.42817643,   618.36981983,
         864.2709288 ,  1233.29376867,  2468.16536437,  3699.31487663,
        4931.04912613,  6164.22835007,  7394.4838804 ,  8628.55004347,
        9857.26726577, 11089.29409363, 12325.6236613 ])

cumulative_regret_y_label_U_3_M_2_K_2_Homo_fixed = np.array([   98.731451,   195.739371,   583.771051,   971.802731,
        1359.834411,  1941.881931,  3882.040331,  5822.198731,
        7762.357131,  9702.515531, 11642.673931, 13582.832331,
       15522.990731, 17463.149131, 19403.307531])


cumulative_regret_y_label_U_3_M_1_K_1_Hete = np.array([  47.5479396 ,   85.44741693,  202.30974467,  279.63557347,
        351.98887987,  432.057262  ,  622.49394387,  766.30827973,
        888.40134893,  986.7439984 , 1079.48338747, 1155.7799612 ,
       1230.3259064 , 1294.57919053, 1349.76008173])


cumulative_regret_y_label_U_3_M_1_K_1_Hete_random = np.array([   62.6440824,   130.5303488,   400.8783148,   669.6583236,
         941.6597684,  1348.1593568,  2710.4055144,  4068.9144736,
        5429.3049236,  6794.027892 ,  8124.5102884,  9483.7443572,
       10813.555994 , 12167.6702692, 13547.9998176])

cumulative_regret_y_label_U_3_M_1_K_1_Hete_fixed = np.array([  31.5055 ,   59.2249 ,  191.40714,  324.88418,  455.55652,
        644.3639 , 1285.0337 , 1931.05622, 2569.63212, 3209.25418,
       3872.10438, 4515.8163 , 5147.7672 , 5780.76676, 6436.3888 ])

cumulative_regret_y_label_U_3_M_2_K_2_Hete = np.array([  70.90422164,  121.25246424,  249.61909688,  330.93710216,
        393.90535   ,  468.37213172,  645.41739728,  750.87488016,
        836.70768676,  901.21792536,  955.42228284, 1003.20044328,
       1045.82314752, 1083.14001096, 1116.97749888])

cumulative_regret_y_label_U_3_M_2_K_2_Hete_random = np.array([  130.1288974,   258.4756408,   768.057909 ,  1268.3684322,
        1767.4182692,  2531.0802302,  5047.5323368,  7585.5962   ,
       10118.856759 , 12646.3309714, 15174.6956858, 17696.0665658,
       20207.3122716, 22721.2521792, 25256.228713 ])

cumulative_regret_y_label_U_3_M_2_K_2_Hete_fixed = np.array([  147.489253  ,   293.256613  ,   876.326053  ,  1459.395493  ,
        2042.464933  ,  2917.069093  ,  5832.416293  ,  8747.763493  ,
       11663.110693  , 14578.457893  , 17493.805093  , 20409.152293  ,
       23324.499493  , 26239.84669299, 29155.19389299])

 
plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_1_K_1_Homo,'-o', label = 'Homo,M=1,proposed' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_1_K_1_Homo_random[0:7],'-^', label = 'Homo,M=1,random' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_1_K_1_Homo_fixed[0:7],'-^', label = 'Homo,M=1,fixed' )
plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_2_K_2_Homo,'-*', label = 'Homo,M=2,proposed' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_2_K_2_Homo_random[0:7],'-^', label = 'Homo,M=2,random' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_2_K_2_Homo_fixed[0:7],'-^', label = 'Homo,M=2,fixed' )

plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_1_K_1_Hete,'-o', label = 'Hete,M=1,proposed' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_1_K_1_Hete_random[0:7],'-*', label = 'Hete,M=1,random' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_1_K_1_Hete_fixed[0:7],'-*', label = 'Hete,M=1,fixed' )
plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_2_K_2_Hete,'--*', label = 'Hete,M=2,proposed' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_2_K_2_Hete_random[0:7],'--*', label = 'Hete,M=2,random' )
plt.plot(regret_x_label[0:7], cumulative_regret_y_label_U_3_M_2_K_2_Hete_fixed[0:7],'--*', label = 'Hete,M=2,fixed' )


plt.legend(ncol=2,framealpha = 0,fontsize=9.5)
plt.xlabel('time',fontsize=14)
plt.ylabel('utility loss',fontsize=14)
# plt.title('s=1',fontsize=16)
plt.xlim([-500, 24000])
plt.grid()
fig.savefig('utility_regret_Price_QoS.eps', dpi = 600, format = 'eps')
# fig.savefig('utility_regret_Price_QoS_for_TII.eps', dpi = 600, format = 'eps')

# In[1] Homogeneous throughput and time slots 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
  

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


average_throughput_U_3_M_1_K_1_1 = np.array([0.64916667, 0.66333333, 0.69770833, 0.71604167, 0.73133929,
       0.7469375 , 0.77496875, 0.78827778, 0.79758333, 0.80384167,
       0.80893403, 0.81289583, 0.81573177, 0.81838657, 0.82054792])

average_throughput_U_3_M_1_K_1_2 = np.array([0.627225  , 0.6453375 , 0.6782625 , 0.69597375, 0.70546696,
       0.71695313, 0.73930188, 0.74978375, 0.75634688, 0.76076075,
       0.76463563, 0.76763045, 0.77023633, 0.77231271, 0.77405038])

average_throughput_U_3_M_1_K_1_3 = np.array([0.6001    , 0.6159125 , 0.64147917, 0.65811125, 0.66864732,
       0.68190813, 0.70567594, 0.71715813, 0.72482531, 0.72992625,
       0.73328865, 0.73573321, 0.73776922, 0.73919611, 0.74059625])
 
average_throughput_U_3_M_1_K_1_1_random = np.array([0.5671875 , 0.55703125, 0.56171875, 0.5734375 , 0.57154018,
       0.56960937, 0.57226562, 0.56997396, 0.57251953, 0.57279687,
       0.57334635, 0.57408482, 0.57581055, 0.57547743, 0.57567969])

# average_throughput_U_3_M_1_K_1_2_random = np.array([0.596875  , 0.571875  , 0.57526042, 0.5790625 , 0.57723214,
#        0.57546875, 0.57359375, 0.57580729, 0.5753125 , 0.57471875,
#        0.57386719, 0.57487723, 0.57464844, 0.57529514, 0.57611719])

# average_throughput_U_3_M_1_K_1_3_random = np.array([0.5609375 , 0.5734375 , 0.58463542, 0.5821875 , 0.57823661,
#        0.576875  , 0.57769531, 0.57460937, 0.57568359, 0.57584375,
#        0.5753776 , 0.57541295, 0.57553711, 0.57467882, 0.57577344])

average_throughput_U_3_M_2_K_2_2_random = np.array([1.14583333, 1.148125  , 1.15652778, 1.15641667, 1.15714286,
       1.1561875 , 1.15398958, 1.15425   , 1.15386458, 1.152125  ,
       1.15236111, 1.1520119 , 1.15258594, 1.15279398, 1.15305208])


average_throughput_U_3_M_1_K_1_1_fixed = np.array([0.475     , 0.475     , 0.47109375, 0.4678125 , 0.47466518,
       0.47085938, 0.47621094, 0.4784375 , 0.48119141, 0.48240625,
       0.48027344, 0.48098214, 0.47998047, 0.48003472, 0.47975781])

average_throughput_U_3_M_1_K_1_2_fixed = np.array([0.5796875 , 0.6015625 , 0.584375  , 0.58671875, 0.58582589,
       0.58625   , 0.58640625, 0.58908854, 0.58998047, 0.59023437,
       0.58977865, 0.58922991, 0.58895508, 0.58939236, 0.58870312])

average_throughput_U_3_M_1_K_1_3_fixed = np.array([0.5171875 , 0.54296875, 0.53697917, 0.53828125, 0.53370536,
       0.53546875, 0.53648438, 0.53588542, 0.53648438, 0.5340625 ,
       0.53321615, 0.53285714, 0.53201172, 0.53184028, 0.53145312])

# --------------------------------------------------------------------------------------
average_throughput_U_3_M_2_K_2_1_fixed = np.array([1.105     , 1.10125   , 1.1125    , 1.1195    , 1.12428571,
       1.12170833, 1.12272917, 1.1215    , 1.12082292, 1.12111667,
       1.12136806, 1.12129762, 1.12114063, 1.12101852, 1.12040833])

average_throughput_U_3_M_2_K_2_2_fixed = np.array([0.8675    , 0.89583333, 0.90166667, 0.90433333, 0.9022619 ,
       0.90570833, 0.90841667, 0.91101389, 0.90928125, 0.910375  ,
       0.91024306, 0.9097619 , 0.91007292, 0.90944444, 0.90974583])

average_throughput_U_3_M_2_K_2_3_fixed = np.array([0.71666667, 0.73416667, 0.76458333, 0.76433333, 0.7660119 ,
       0.76891667, 0.76335417, 0.76205556, 0.76204167, 0.76155   ,
       0.76114583, 0.76029167, 0.75992188, 0.75971296, 0.76002083])

# --------------------------------------------------------------------------------------
average_throughput_U_3_M_2_K_2_1 = np.array([1.3125    , 1.34041667, 1.37875   , 1.39608333, 1.40166667,
       1.41266667, 1.43639583, 1.443125  , 1.45038542, 1.45695833,
       1.4615    , 1.46557143, 1.46799479, 1.46953241, 1.47179583])

average_throughput_U_3_M_2_K_2_2 = np.array([1.26916667, 1.28      , 1.30263889, 1.308     , 1.31547619,
       1.32366667, 1.32804167, 1.33766667, 1.337875  , 1.33956667,
       1.34097222, 1.34244048, 1.34360938, 1.34489815, 1.34583333])

average_throughput_U_3_M_2_K_2_3 = np.array([1.235     , 1.24375   , 1.22944444, 1.21616667, 1.21065476,
       1.21341667, 1.19645833, 1.18608333, 1.17915625, 1.1741    ,
       1.17086111, 1.16834524, 1.16585938, 1.16489815, 1.162125  ])


average_throughput_U_3_M_2_K_2_1_optimal = np.array([1.4025    , 1.4075    , 1.44791667, 1.44525   , 1.43714286,
       1.43575   , 1.434625  , 1.43629167, 1.4369375 , 1.43765   ,
       1.43689583, 1.43855357, 1.43860938, 1.43833333, 1.437525  ])

average_throughput_U_3_M_2_K_2_2_optimal = np.array([1.4975    , 1.53125   , 1.55625   , 1.55025   , 1.55464286,
       1.55725   , 1.5591875 , 1.552875  , 1.55325   , 1.555125  ,
       1.55514583, 1.55476786, 1.55454687, 1.55344444, 1.5531875 ])

average_throughput_U_3_M_2_K_2_3_optimal = np.array([1.0175    , 1.04625   , 1.00291667, 0.994     , 0.99821429,
       1.000375  , 1.0130625 , 1.016875  , 1.0143125 , 1.01135   ,
       1.01314583, 1.01419643, 1.01351563, 1.012625  , 1.0117375 ])

 
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1,'-^',label = 'M=1,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2,'-^',label = 'M=1,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3,'-^',label = 'M=1,user 3')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_random,'--^',label = 'M=1,user 1,2,3,random')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_fixed,'--^',label = 'M=1,user 1,fixed')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2_fixed,'--o',label = 'M=1,user 2,fixed')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3_fixed,'--*',label = 'M=1,user 3,fixed')


# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1_optimal,'-',label = 'M=2,user 1,optimal')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2_optimal,'-',label = 'M=2,user 2,optimal')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3_optimal,'-',label = 'M=2,user 3,optimal')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1,'-o',label = 'M=2,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2,'-o',label = 'M=2,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3,'-o',label = 'M=2,user 3')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2_random,'--^',label = 'M=2,user 1,2,3,random')
plt.plot(regret_x_label[1:15], average_throughput_U_3_M_2_K_2_1_fixed[1:15],'--^',label = 'M=2,user 1,fixed')
plt.plot(regret_x_label[1:15], average_throughput_U_3_M_2_K_2_2_fixed[1:15],'--o',label = 'M=2,user 2,fixed')
plt.plot(regret_x_label[2:15], average_throughput_U_3_M_2_K_2_3_fixed[2:15],'--*',label = 'M=2,user 3,fixed')

# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_Hete,'-^',label = 'Hete,M=K=1,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2_Hete,'-o',label = 'Hete,M=K=1,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3_Hete,'-*',label = 'Hete,M=K=1,user 3')

# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1_Hete,'-^',label = 'Hete,M=K=2,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2_Hete,'-o',label = 'Hete,M=K=2,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3_Hete,'-*',label = 'Hete,M=K=2,user 3')


plt.legend(ncol=1,framealpha = 0,fontsize=10)
plt.xlabel('time',fontsize=14)
plt.ylabel('average throughput (Mbps)',fontsize=14)
plt.xlim([-800, 32000])
plt.grid()

fig.savefig('Homo_average_throughput.eps', dpi = 600, format = 'eps')


# In[1]  Heterogeneous throughput and time slots
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]
 
average_throughput_U_3_M_1_K_1_1_Hete = np.array([0.64967188, 0.66775   , 0.7069401 , 0.72774219, 0.74117187,
       0.75534766, 0.7783957 , 0.79121641, 0.79891113, 0.80454922,
       0.80947695, 0.81318996, 0.81575625, 0.81828802, 0.82035227])

average_throughput_U_3_M_1_K_1_2_Hete = np.array([0.67390625, 0.68733594, 0.73654427, 0.763275  , 0.78117299,
       0.79764609, 0.8312418 , 0.84744167, 0.85652402, 0.86285562,
       0.8678263 , 0.87121551, 0.87379658, 0.87633698, 0.87827195])

average_throughput_U_3_M_1_K_1_3_Hete = np.array([0.54507812, 0.55989063, 0.58271354, 0.59094687, 0.59865513,
       0.60530547, 0.61933281, 0.62769505, 0.6307625 , 0.63196469,
       0.63334401, 0.63448382, 0.63495166, 0.63506554, 0.63542344])

average_throughput_U_3_M_2_K_2_1_Hete = np.array([1.3285    , 1.3575    , 1.4205    , 1.43655   , 1.44903571,
       1.46465   , 1.4821125 , 1.49518333, 1.504475  , 1.51389   ,
       1.51670417, 1.52120357, 1.52444375, 1.52508889, 1.527325  ])

average_throughput_U_3_M_2_K_2_2_Hete = np.array([1.5325    , 1.583     , 1.65725   , 1.70265   , 1.72885714,
       1.7506    , 1.77445   , 1.79146667, 1.7994375 , 1.805635  ,
       1.81074167, 1.81400714, 1.81669687, 1.81875   , 1.820555  ])

average_throughput_U_3_M_2_K_2_3_Hete = np.array([1.246     , 1.245     , 1.282     , 1.3004    , 1.30664286,
       1.324025  , 1.330025  , 1.34465   , 1.3554875 , 1.36476   ,
       1.37234167, 1.37226429, 1.37281875, 1.37570556, 1.37878   ])





# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1,'-^',label = 'M=K=1,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2,'-o',label = 'M=K=1,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3,'-*',label = 'M=K=1,user 3')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1,'-^',label = 'M=K=2,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2,'-o',label = 'M=K=2,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3,'-*',label = 'M=K=2,user 3')



plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_Hete,'-^',label = 'M=K=1,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2_Hete,'-o',label = 'M=K=1,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3_Hete,'-*',label = 'M=K=1,user 3')

plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1_Hete,'-^',label = 'M=K=2,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2_Hete,'-o',label = 'M=K=2,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3_Hete,'-*',label = 'M=K=2,user 3')


plt.legend(ncol=2,framealpha = 0,fontsize=12, loc = (3/16, 3/9) )
plt.xlabel('time',fontsize=14)
plt.ylabel('average throughput (Mbps)',fontsize=14)
plt.grid()
# plt.ylim([0.5,2.3])
# fig.savefig('Hete_average_throughput.eps', dpi = 600, format = 'eps')
 


# In[2] Homogeneous price and time slots 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

average_price_U_3_M_1_K_1_1 = np.array([0.35916667, 0.37659583, 0.41848264, 0.44194625, 0.45901518,
       0.48191729, 0.53501906, 0.56430194, 0.589345  , 0.607142  ,
       0.62127844, 0.63340122, 0.64206419, 0.65015201, 0.65714077])

average_price_U_3_M_1_K_1_2 = np.array([0.32725417, 0.3336    , 0.35507569, 0.36139583, 0.36559048,
       0.36819646, 0.37119885, 0.37136931, 0.36900349, 0.36723113,
       0.36567694, 0.36339574, 0.36187495, 0.35994141, 0.35888533])

average_price_U_3_M_1_K_1_3 = np.array([0.28639167, 0.2848875 , 0.28122778, 0.28268708, 0.28266935,
       0.28134417, 0.27984031, 0.27792792, 0.2763737 , 0.27574129,
       0.27514035, 0.27451863, 0.2738588 , 0.27350769, 0.27290081])


average_price_U_3_M_2_K_2_1 = np.array([0.75860833, 0.78497083, 0.84234583, 0.87399083, 0.88549167,
       0.90296833, 0.94514937, 0.96786264, 0.98318156, 0.99396592,
       1.00227653, 1.01029881, 1.01544786, 1.01992384, 1.02357704])

average_price_U_3_M_2_K_2_2 = np.array([0.657325  , 0.67036667, 0.66815417, 0.65980333, 0.66027738,
       0.65734417, 0.63659083, 0.62560944, 0.61699896, 0.61264883,
       0.60988444, 0.60741429, 0.60551526, 0.60440333, 0.60346692])

average_price_U_3_M_2_K_2_3 = np.array([0.56886667, 0.56134583, 0.53413333, 0.51613667, 0.50421012,
       0.49773833, 0.47594833, 0.46340194, 0.45509865, 0.4488785 ,
       0.44398333, 0.43993048, 0.43696714, 0.43440417, 0.431677  ])



average_price_U_3_M_1_K_1_1_random = np.array([0.28654688, 0.28377344, 0.2856901 , 0.29055312, 0.29001228,
       0.28910547, 0.29030937, 0.28913151, 0.29026113, 0.29068172,
       0.2909918 , 0.29157366, 0.29267178, 0.29255564, 0.2926757 ])

# average_price_U_3_M_1_K_1_2_random = np.array([0.31240625, 0.29775   , 0.2935599 , 0.29505   , 0.29400446,
#        0.292325  , 0.29098672, 0.29220286, 0.29218203, 0.29150656,
#        0.29151211, 0.29189163, 0.29156836, 0.29191649, 0.29229633])

# average_price_U_3_M_1_K_1_3_random = np.array([0.28417188, 0.29004687, 0.29573177, 0.29495938, 0.29263728,
#        0.29206719, 0.29321523, 0.29098359, 0.29182598, 0.29179656,
#        0.29174727, 0.29162333, 0.29181533, 0.29134939, 0.29201742])


average_price_U_3_M_2_K_2_2_random = np.array([0.58274167, 0.58186042, 0.58723611, 0.58700583, 0.58791548,
       0.58770479, 0.58546844, 0.58553583, 0.58556635, 0.58471708,
       0.58509792, 0.58460643, 0.58500326, 0.585211  , 0.58531115])




average_price_U_3_M_1_K_1_1_fixed = np.array([0.14725   , 0.14725   , 0.14603906, 0.14502187, 0.14714621,
       0.14596641, 0.14762539, 0.14831562, 0.14916934, 0.14954594,
       0.14888477, 0.14910446, 0.14879395, 0.14881076, 0.14872492])

average_price_U_3_M_1_K_1_2_fixed = np.array([0.30723438, 0.31882812, 0.30971875, 0.31096094, 0.31048772,
       0.3107125 , 0.31079531, 0.31221693, 0.31268965, 0.31282422,
       0.31258268, 0.31229185, 0.31214619, 0.31237795, 0.31201266])

average_price_U_3_M_1_K_1_3_fixed = np.array([0.23273437, 0.24433594, 0.24164062, 0.24222656, 0.24016741,
       0.24096094, 0.24141797, 0.24114844, 0.24141797, 0.24032812,
       0.23994727, 0.23978571, 0.23940527, 0.23932812, 0.23915391])



average_price_U_3_M_2_K_2_1_fixed = np.array([0.54351667, 0.54179583, 0.54744722, 0.550835  , 0.5531619 ,
       0.55195875, 0.55255146, 0.55186722, 0.55156615, 0.55171183,
       0.55187674, 0.55183345, 0.55175036, 0.55169426, 0.55140608])

average_price_U_3_M_2_K_2_2_fixed = np.array([0.367525  , 0.37920833, 0.38345   , 0.38404333, 0.3826869 ,
       0.38394958, 0.38461417, 0.38575431, 0.38511719, 0.38557625,
       0.38553368, 0.38531905, 0.3854626 , 0.38518667, 0.38538921])

average_price_U_3_M_2_K_2_3_fixed = np.array([0.3142    , 0.321575  , 0.33467917, 0.33467333, 0.33532917,
       0.3365425 , 0.33420687, 0.33362611, 0.33361125, 0.33342383,
       0.3332584 , 0.33290363, 0.33274359, 0.33267306, 0.33280321])



plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1,'-^',label = 'M=1,user 1')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2,'-^',label = 'M=1,user 2')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3,'-^',label = 'M=1,user 3')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1_random,'--.',label = 'M=1,user 1,2,3,random')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1_fixed,'--*',label = 'M=1,user 1,fixed')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2_fixed,'--*',label = 'M=1,user 2,fixed')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3_fixed,'--*',label = 'M=1,user 3,fixed')

plt.plot(regret_x_label, average_price_U_3_M_2_K_2_1,'-o',label = 'M=2,user 1')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_2,'-o',label = 'M=2,user 2')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_3,'-o',label = 'M=2,user 3')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_2_random,'--.',label = 'M=2,user 1,2,3,random')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_1_fixed,'--x',label = 'M=2,user 1,fixed')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_2_fixed,'--x',label = 'M=2,user 2,fixed')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_3_fixed,'--x',label = 'M=2,user 3,fixed')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2_random,'--o',label = 'M=1,user 2,random')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3_random,'--*',label = 'M=1,user 3,random')




# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1_Hete,'-^',label = 'Hete,M=K=1,user 1')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2_Hete,'-o',label = 'Hete,M=K=1,user 2')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3_Hete,'-*',label = 'Hete,M=K=1,user 3')

# plt.plot(regret_x_label, average_price_U_3_M_2_K_2_1_Hete,'-^',label = 'Hete,M=K=2,user 1')
# plt.plot(regret_x_label, average_price_U_3_M_2_K_2_2_Hete,'-o',label = 'Hete,M=K=2,user 2')
# plt.plot(regret_x_label, average_price_U_3_M_2_K_2_3_Hete,'-*',label = 'Hete,M=K=2,user 3')


# plt.legend(ncol=1,framealpha = 0,fontsize=8, loc = (1/16, 6/10) )
plt.legend(ncol=1,framealpha = 0,fontsize = 10 )
plt.xlabel('time',fontsize=14)
plt.ylabel('average price',fontsize=14)
plt.xlim([-800, 32000 ])
plt.grid()
fig.savefig('Homo_average_price.eps', dpi = 600, format = 'eps')


# In[2] Heterogeneous price and time slots 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]
 
average_price_U_3_M_1_K_1_1_Hete = np.array([0.36338333, 0.3852    , 0.40790278, 0.44410833, 0.46669881,
       0.48894167, 0.544795  , 0.57695472, 0.59837688, 0.61161525,
       0.62492868, 0.63490655, 0.64369667, 0.65036083, 0.65834129])

average_price_U_3_M_1_K_1_2_Hete = np.array([0.303275  , 0.31866667, 0.34704028, 0.36488167, 0.37324583,
       0.38462958, 0.40497854, 0.41601708, 0.42288042, 0.4284305 ,
       0.43237076, 0.43580018, 0.43882078, 0.44102722, 0.44323083])

average_price_U_3_M_1_K_1_3_Hete = np.array([0.22880833, 0.23037917, 0.23335556, 0.23328667, 0.23343155,
       0.2346275 , 0.23383188, 0.23528319, 0.23501792, 0.23379342,
       0.23243826, 0.23143262, 0.23075177, 0.22995157, 0.22849321])


average_price_U_3_M_2_K_2_1_Hete = np.array([0.724925  , 0.769735  , 0.85672167, 0.8985965 , 0.92108179,
       0.9402865 , 0.96609213, 0.978903  , 0.98671213, 0.9935382 ,
       0.99682304, 1.00032096, 1.00311094, 1.00464711, 1.00645565])

average_price_U_3_M_2_K_2_2_Hete = np.array([0.715675  , 0.74522   , 0.7858325 , 0.8075365 , 0.82064286,
       0.8315605 , 0.84380825, 0.851609  , 0.85553337, 0.85837155,
       0.86073992, 0.86223479, 0.86341784, 0.8644045 , 0.8652412 ])

average_price_U_3_M_2_K_2_3_Hete = np.array([0.51151   , 0.506735  , 0.50439833, 0.501552  , 0.49867214,
       0.49774925, 0.487361  , 0.48653133, 0.48680787, 0.4875725 ,
       0.488743  , 0.48734764, 0.48597025, 0.486357  , 0.48649165])


# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1,'-^',label = 'M=K=1,user 1')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2,'-o',label = 'M=K=1,user 2')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3,'-*',label = 'M=K=1,user 3')
# plt.plot(regret_x_label, average_price_U_3_M_2_K_2_1,'-^',label = 'M=K=2,user 1')
# plt.plot(regret_x_label, average_price_U_3_M_2_K_2_2,'-o',label = 'M=K=2,user 2')
# plt.plot(regret_x_label, average_price_U_3_M_2_K_2_3,'-*',label = 'M=K=2,user 3')

plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1_Hete,'-^',label = 'M=1,user 1')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2_Hete,'-o',label = 'M=1,user 2')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3_Hete,'-*',label = 'M=1,user 3')

plt.plot(regret_x_label, average_price_U_3_M_2_K_2_1_Hete,'-^',label = 'M=2,user 1')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_2_Hete,'-o',label = 'M=2,user 2')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_3_Hete,'-*',label = 'M=2,user 3')


plt.legend(ncol=2,framealpha = 0,fontsize=11, loc = (4/16, 4.8/9) )
plt.xlabel('time',fontsize=14)
plt.ylabel('average price',fontsize=14)
# plt.ylim([0.3,1.75])
plt.grid()
# fig.savefig('Hete_average_price.eps', dpi = 600, format = 'eps')

 


























