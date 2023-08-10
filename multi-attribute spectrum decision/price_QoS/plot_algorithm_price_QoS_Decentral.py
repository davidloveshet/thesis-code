# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 20:51:34 2021

@author: xxx
"""

# In[11] regret and time slots 
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]

cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_1 = np.array([  78.045   ,  152.93375 ,  458.50625 ,  701.089375,  906.6575  ,
       1181.5525  , 1590.848125, 1726.45625 , 1813.77875 , 1869.3     ,
       1918.313125, 1957.0425  , 1993.986875, 2029.12875 , 2057.655   ])
 
cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_1_random = np.array([   81.10666667,   165.92      ,   498.04      ,   822.305     ,
        1154.87      ,  1649.36166667,  3291.27333333,  4934.35166667,
        6581.95833333,  8204.91      ,  9844.885     , 11504.73166667,
       13151.75166667, 14777.62833333, 16418.26333333])

cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_1_fixed = np.array([   65.6,   129.6,   385.6,   641.6,   897.6,  1281.6,  2561.6,
        3841.6,  5121.6,  6401.6,  7681.6,  8961.6, 10241.6, 11521.6,
       12801.6])

# -----------------------------------------------------------------------------------------------------------
cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_1 = np.array([ 109.43125,  199.235  ,  465.7025 ,  628.70125,  774.93625,
        976.9675 , 1390.97375, 1447.69875, 1499.055  , 1549.155  ,
       1587.5175 , 1622.3675 , 1658.1675 , 1689.74875, 1718.58   ])


cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_1_random = np.array([  130.428,   261.352,   780.989,  1303.946,  1822.227,  2599.884,
        5200.369,  7797.948, 10391.943, 12990.306, 15602.485, 18198.099,
       20815.007, 23407.233, 26005.618])

cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_1_fixed = np.array([  103.59,   204.39,   607.59,  1010.79,  1413.99,  2018.79,
        4034.79,  6050.79,  8066.79, 10082.79, 12098.79, 14114.79,
       16130.79, 18146.79, 20162.79])

# -----------------------------------------------------------------------------------------------------------
cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_2 = np.array([  65.04148062,  128.30766262,  370.37701037,  626.04404487,
        887.94287025, 1235.08590262, 2019.73561437, 2368.4375975 ,
       2526.0041195 , 2645.59027788, 2694.81633425, 2743.516798  ,
       2784.05577263, 2816.837807  , 2847.42364138])


cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_2_random = np.array([   71.3216    ,   140.71510638,   412.29496137,   687.939015  ,
         967.95881887,  1387.53753937,  2783.78119163,  4171.7018035 ,
        5573.74078813,  6968.20486275,  8362.88792625,  9757.950484  ,
       11151.718693  , 12543.6616135 , 13935.84701212])

cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_2_fixed = np.array([   55.953735,   110.668135,   329.525735,   548.383335,
         767.240935,  1095.527335,  2189.815335,  3284.103335,
        4378.391335,  5472.679335,  6566.967335,  7661.255335,
        8755.543335,  9849.831335, 10944.119335])




cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_2 = np.array([  95.29197425,  171.35810337,  422.60872612,  600.57767738,
        790.74119725, 1031.68984075, 1400.84100212, 1747.64418525,
       1896.48553287, 1978.6042575 , 2011.3341995 , 2039.46644325,
       2071.31448013, 2094.65501825, 2116.6551195 ])

# cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_3 = np.array([  35.831114,   68.018306,  193.712402,  287.320657,  373.351157,
#         464.75326 ,  576.090869,  676.796051,  766.717958,  851.852798,
#         922.183321,  990.331262, 1057.786245, 1112.211732, 1167.264181])

# cumulative_regret_y_label_U_3_M_2_K_2_Homo = np.array([  34.0011519,   58.3473156,  140.2077935,  205.0160685,
#         258.046009 ,  328.9804365,  521.4744721,  669.0590301,
#         788.8111665,  895.1144684,  979.2313966, 1050.9070276,
#        1125.2949588, 1191.9276263, 1264.7836929])

# cumulative_regret_y_label_U_3_M_1_K_1_Hete = np.array([  44.44650387,   82.12661362,  193.41998613,  276.00859875,
#         341.4676315 ,  427.478061  ,  631.11661712,  775.97938387,
#         895.58216062,  995.361557  , 1074.6133765 , 1154.23659925,
#        1234.67758225, 1293.89739762, 1353.46418387])

# cumulative_regret_y_label_U_3_M_2_K_2_Hete = np.array([  71.5885022,  122.611221 ,  248.9624712,  324.2779342,
#         385.768852 ,  458.7599012,  634.74041  ,  734.1541978,
#         814.6182336,  880.1267364,  942.1090892,  992.6815496,
#        1029.2394516, 1064.9724462, 1098.4567822])

plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_1,'-o', label = '$s=1$,M=1,proposed' )
plt.plot(regret_x_label[0:8], cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_1_random[0:8],'--o', label = '$s=1$,M=1,random' )
plt.plot(regret_x_label[0:8], cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_1_fixed[0:8],'--o', label = '$s=1$,M=1,fixed' )
plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_2,'-o', label = '$s=2$,M=1,proposed' )
plt.plot(regret_x_label[0:8], cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_2_random[0:8],'--o', label = '$s=2$,M=1,random' )
plt.plot(regret_x_label[0:8], cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_2_fixed[0:8],'--o', label = '$s=2$,M=1,fixed' )

plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_1,'-*', label = '$s=1$,M=2,proposed' )
plt.plot(regret_x_label[0:8], cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_1_random[0:8],'-o', label = '$s=1$,M=2,random' )
plt.plot(regret_x_label[0:8], cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_1_fixed[0:8],'-o', label = '$s=1$,M=2,fixed' )



plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_2_K_2_Decen_s_2,'-*', label = '$s=2$,M=2,proposed' )
# plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_1_K_1_Decen_s_3,'-o', label = '$s_3$,M=K=1' )
# plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_2_K_2_Homo,'-*', label = 'Homo,U=3,M=2,K=2' )
# plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_1_K_1_Hete,'--o', label = 'Hete,U=3,M=1,K=1' )
# plt.plot(regret_x_label, cumulative_regret_y_label_U_3_M_2_K_2_Hete,'--*', label = 'Hete,U=3,M=2,K=2' )
plt.legend( ncol= 2, framealpha = 0, fontsize= 8 ) # ,loc = 'upper right'
plt.xlabel('time',fontsize=14)
plt.ylabel('utility loss',fontsize=14)
plt.xlim([-800, 18000])
# plt.title('s=1',fontsize=16)
# plt.ylim([-500,3000])
plt.grid()
# fig.savefig('utility_regret_Price_QoS_decentral.eps', dpi = 600, format = 'eps')

 


# In[1] throughput
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


average_throughput_U_3_M_1_K_1_1 = np.array([0.46875   , 0.471875  , 0.45963542, 0.486875  , 0.51266741,
       0.54246094, 0.62974609, 0.67717448, 0.7031543 , 0.72025781,
       0.73236979, 0.7409096 , 0.74771484, 0.75244792, 0.75624609])

average_throughput_U_3_M_1_K_1_2 = np.array([0.47734375, 0.47851562, 0.4734375 , 0.50695313, 0.5280692 ,
       0.54738281, 0.62814453, 0.67565104, 0.70225586, 0.71867187,
       0.72996094, 0.73943638, 0.74614746, 0.75162326, 0.75607031])

average_throughput_U_3_M_1_K_1_3 = np.array([0.46328125, 0.4875    , 0.50442708, 0.5275    , 0.54821429,
       0.57398437, 0.64662109, 0.68736979, 0.71121094, 0.72755469,
       0.73813802, 0.74540179, 0.75062012, 0.75511285, 0.75870313])


average_throughput_U_3_M_1_K_1_1_random = np.array([0.475     , 0.45625   , 0.45625   , 0.45770833, 0.45669643,
       0.45291667, 0.45078125, 0.45600694, 0.45486979, 0.4566875 ,
       0.45755208, 0.45813988, 0.45817708, 0.45880787, 0.45888542])

average_throughput_U_3_M_1_K_1_2_random = np.array([0.48333333, 0.47083333, 0.46875   , 0.46770833, 0.46205357,
       0.46104167, 0.45619792, 0.45430556, 0.45429688, 0.45497917,
       0.45539931, 0.45401786, 0.45394531, 0.45521991, 0.45582292])

average_throughput_U_3_M_1_K_1_3_random = np.array([0.44791667, 0.47395833, 0.46597222, 0.46333333, 0.46517857,
       0.46260417, 0.46270833, 0.46413194, 0.46356771, 0.4631875 ,
       0.46348958, 0.46297619, 0.46153646, 0.46179398, 0.46132292])


average_throughput_U_3_M_1_K_1_1_fixed = np.array([0.55      , 0.53125   , 0.53472222, 0.5375    , 0.53125   ,
       0.53354167, 0.535625  , 0.535625  , 0.53703125, 0.53591667,
       0.53600694, 0.53366071, 0.53335938, 0.5325463 , 0.53422917])

average_throughput_U_3_M_1_K_1_2_fixed = np.array([0.4875    , 0.52291667, 0.52847222, 0.52291667, 0.53095238,
       0.521875  , 0.52333333, 0.52493056, 0.53067708, 0.529625  ,
       0.52895833, 0.53053571, 0.53117188, 0.53064815, 0.53035417])

average_throughput_U_3_M_1_K_1_3_fixed = np.array([0.55416667, 0.54583333, 0.52708333, 0.53958333, 0.53482143,
       0.531875  , 0.526875  , 0.52875   , 0.53260417, 0.53283333,
       0.53145833, 0.53321429, 0.5328125 , 0.53261574, 0.5335    ])




# average_throughput_U_3_M_2_K_2_1 = np.array([1.2892625 , 1.31755625, 1.35441042, 1.37787125, 1.39371964,
#         1.407345  , 1.43234938, 1.44533   , 1.45299687, 1.46073113,
#         1.46385344, 1.46529259, 1.46911805, 1.46967278, 1.47136763])

# average_throughput_U_3_M_2_K_2_2 = np.array([1.2766625 , 1.29734375, 1.3144875 , 1.3155875 , 1.31650893,
#         1.319835  , 1.32642375, 1.32908625, 1.33447625, 1.3369225 ,
#         1.33961542, 1.34459107, 1.3459093 , 1.34957063, 1.35045094])

# average_throughput_U_3_M_2_K_2_3 = np.array([1.211575  , 1.22049375, 1.21722708, 1.21046625, 1.20803393,
#         1.20264438, 1.18809469, 1.18161854, 1.17448047, 1.16886125,
#         1.16651417, 1.163065  , 1.16016148, 1.15782306, 1.15658337])


average_throughput_U_3_M_1_K_1_s_2_1 = np.array([0.45      , 0.46875   , 0.47447917, 0.47171875, 0.45970982,
       0.47328125, 0.53675781, 0.595     , 0.63580078, 0.66353125,
       0.68302083, 0.6974442 , 0.7084375 , 0.71717014, 0.7243125 ])

average_throughput_U_3_M_1_K_1_s_2_2 = np.array([0.4703125 , 0.47734375, 0.48020833, 0.475     , 0.47578125,
       0.47867188, 0.53660156, 0.59333333, 0.63410156, 0.66104688,
       0.68169271, 0.69645089, 0.70771484, 0.7166059 , 0.72423437])

average_throughput_U_3_M_1_K_1_s_2_3 = np.array([0.4953125 , 0.4859375 , 0.50859375, 0.50078125, 0.49642857,
       0.49773438, 0.5484375 , 0.59765625, 0.63552734, 0.6614375 ,
       0.68289062, 0.69776786, 0.7084082 , 0.71830729, 0.72510937])

# average_throughput_U_3_M_2_K_2_1_Hete = np.array([1.3486    , 1.377475  , 1.46710417, 1.5075975 , 1.53094821,
#         1.55384125, 1.5800925 , 1.59532333, 1.60431312, 1.60985525,
#         1.61343042, 1.61655661, 1.61941406, 1.62171472, 1.62368238])

# average_throughput_U_3_M_2_K_2_2_Hete = np.array([1.3685    , 1.436075  , 1.54585   , 1.5950025 , 1.61973393,
#         1.6421925 , 1.67855187, 1.69563292, 1.70511125, 1.71196975,
#         1.71686062, 1.72046518, 1.72347375, 1.72585472, 1.72776   ])

# average_throughput_U_3_M_2_K_2_3_Hete = np.array([1.113275  , 1.1510875 , 1.20474167, 1.22564   , 1.23789464,
#         1.24763   , 1.26413687, 1.26895833, 1.27601625, 1.27800575,
#         1.27951854, 1.28077982, 1.28406344, 1.28464958, 1.28563863])


average_throughput_U_3_M_2_K_2_1 = np.array([1.0140625 , 1.0640625 , 1.13723958, 1.1828125 , 1.20412946,
       1.23132813, 1.27941406, 1.30132812, 1.31353516, 1.3198125 ,
       1.32481771, 1.32792411, 1.32958008, 1.33144097, 1.33251563])

average_throughput_U_3_M_2_K_2_2 = np.array([1.09375   , 1.1078125 , 1.1390625 , 1.185     , 1.20089286,
       1.21914063, 1.25398437, 1.28450521, 1.29669922, 1.30621875,
       1.31289063, 1.31520089, 1.31859375, 1.32108507, 1.32297656])

average_throughput_U_3_M_2_K_2_3 = np.array([1.13125   , 1.15703125, 1.21145833, 1.2284375 , 1.24386161,
       1.25046875, 1.26699219, 1.28921875, 1.30167969, 1.30776563,
       1.31330729, 1.31996652, 1.32400391, 1.32704861, 1.32853906])


average_throughput_U_3_M_2_K_2_1_random = np.array([0.9125    , 0.896875  , 0.89520833, 0.89175   , 0.88928571,
       0.892625  , 0.89703125, 0.9005    , 0.9006875 , 0.9008125 ,
       0.90025   , 0.90077679, 0.899625  , 0.89950694, 0.8987875 ])


average_throughput_U_3_M_2_K_2_1_fixed = np.array([0.925     , 0.929375  , 0.92458333, 0.927375  , 0.92919643,
       0.9314375 , 0.93084375, 0.92891667, 0.93135938, 0.9302    ,
       0.92985417, 0.93063393, 0.931     , 0.92973611, 0.930325  ])


average_throughput_U_3_M_2_K_2_s_2_1 = np.array([0.9765625 , 1.009375  , 1.09427083, 1.1284375 , 1.14274554,
       1.16046875, 1.22722656, 1.25013021, 1.26714844, 1.27765625,
       1.28851563, 1.29723214, 1.30413086, 1.30832465, 1.31233594])

average_throughput_U_3_M_2_K_2_s_2_2 = np.array([1.0296875 , 1.10234375, 1.1421875 , 1.1578125 , 1.16997768,
       1.17976562, 1.23113281, 1.25458333, 1.2715625 , 1.28376563,
       1.29257812, 1.29877232, 1.30351562, 1.30758681, 1.31083594])

average_throughput_U_3_M_2_K_2_s_2_3 = np.array([1.13278846, 1.14486058, 1.18613301, 1.20226154, 1.21427129,
       1.22045288, 1.24514014, 1.25806715, 1.27107897, 1.281645  ,
       1.29003309, 1.29657816, 1.30194171, 1.30625887, 1.3097438 ])




plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1,'-^',label = 'M=1,s=1,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2,'-o',label = 'M=1,s=1,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3,'-*',label = 'M=1,s=1,user 3')

plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_s_2_1,'-^',label = 'M=1,s=2,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_s_2_2,'-o',label = 'M=1,s=2,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_s_2_3,'-*',label = 'M=1,s=2,user 3')

plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1,'-^',label = 'M=2,s=1,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2,'-o',label = 'M=2,s=1,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3,'-*',label = 'M=2,s=1,user 3')

plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_s_2_1,'-^',label = 'M=2,s=2,user 1')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_s_2_2,'-o',label = 'M=2,s=2,user 2')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_s_2_3,'-*',label = 'M=2,s=2,user 3')

plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_random,'--^',label = 'M=1,s=1,2,user 1,2,3,random')

plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1_random,'--^',label = 'M=2,s=1,2,user 1,2,3,random')

plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_fixed,'--^',label = 'M=1,s=1,2,user 1,2,3,fixed')
plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1_fixed,'--^',label = 'M=2,s=1,2,user 1,2,3,fixed')



# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2_random,'--o',label = 'M=1,s=1,user 2,random')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3_random,'--*',label = 'M=1,s=1,user 3,random')


# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2_fixed,'--o',label = 'M=1,s=1,user 2,fixed')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3_fixed,'--*',label = 'M=1,s=1,user 3,fixed')


# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1,'-^',label = 'M=K=2,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2,'-o',label = 'M=K=2,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3,'-*',label = 'M=K=2,user 3')


    
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_Hete,'-^',label = 'M=K=1,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2_Hete,'-o',label = 'M=K=1,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3_Hete,'-*',label = 'M=K=1,user 3')

# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1_Hete,'-^',label = 'M=K=2,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2_Hete,'-o',label = 'M=K=2,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3_Hete,'-*',label = 'M=K=2,user 3')


plt.legend(ncol=1,framealpha = 0,fontsize=8)
plt.xlabel('time',fontsize=14)
plt.ylabel('average throughput (Mbps)',fontsize=14)
plt.grid()
plt.xlim([-800, 30000])
# plt.ylim([0.3, 2])


fig.savefig('Decentralized_average_throughput_1.eps', dpi = 600, format = 'eps')

 # In[1] price
import numpy as np
import random 
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
 

T_simulation = 16000 # the overall simulation time
regret_x_label = [ round(0.005 * T_simulation), round(0.01 * T_simulation), round(0.03 * T_simulation), round(0.05 * T_simulation),  round(0.07 * T_simulation), round(0.1 * T_simulation), round(0.2 * T_simulation), round(0.3 *T_simulation), round(0.4 *T_simulation), round(0.5 *T_simulation), round(0.6 *T_simulation), round(0.7 *T_simulation), round(0.8 *T_simulation), round(0.9 *T_simulation),  T_simulation  ]


average_price_U_3_M_1_K_1_1 = np.array([0.35569531, 0.37630469, 0.40438672, 0.41897188, 0.4269308 ,
       0.43483711, 0.44470273, 0.44808021, 0.44936973, 0.45062625,
       0.45145905, 0.45207879, 0.45257798, 0.45268576, 0.45279699])

average_price_U_3_M_1_K_1_2 = np.array([0.36303906, 0.38283594, 0.40514583, 0.41796719, 0.42563839,
       0.4330793 , 0.44273926, 0.44614635, 0.44837568, 0.44938625,
       0.45017812, 0.45107824, 0.45168467, 0.45217157, 0.45246785])

average_price_U_3_M_1_K_1_3 = np.array([0.34360938, 0.36304688, 0.40054427, 0.41681094, 0.42502734,
       0.4338793 , 0.44394707, 0.44709727, 0.44936152, 0.45067641,
       0.45145775, 0.4516923 , 0.45171929, 0.45201979, 0.45212156])


average_price_U_3_M_1_K_1_s_2_1 = np.array([0.316625  , 0.33076563, 0.35877083, 0.36163281, 0.36777232,
       0.37290859, 0.38905977, 0.39650286, 0.40229297, 0.40637891,
       0.40964674, 0.41167277, 0.41440908, 0.41649384, 0.41844023])

average_price_U_3_M_1_K_1_s_2_2 = np.array([0.33548437, 0.34082812, 0.34959375, 0.35929219, 0.36499107,
       0.36368906, 0.37744961, 0.38625521, 0.39291328, 0.39787234,
       0.40339297, 0.40710379, 0.41054863, 0.41346224, 0.41620328])

average_price_U_3_M_1_K_1_s_2_3 = np.array([0.33860937, 0.338     , 0.35263542, 0.361375  , 0.36729464,
       0.37101484, 0.37960898, 0.38989115, 0.39458652, 0.40004297,
       0.40603906, 0.41034688, 0.412921  , 0.416723  , 0.41891453])


average_price_U_3_M_2_K_2_1 = np.array([0.66570313, 0.68529688, 0.68670833, 0.68746094, 0.69126004,
       0.69632969, 0.70144141, 0.70618125, 0.70865703, 0.70979234,
       0.71092383, 0.71162734, 0.71156191, 0.71182943, 0.71213836])

average_price_U_3_M_2_K_2_2 = np.array([0.63425   , 0.66725   , 0.68462109, 0.68560234, 0.69012054,
       0.69312852, 0.6948709 , 0.69739167, 0.69915879, 0.70031375,
       0.70000215, 0.70022059, 0.70032944, 0.7007237 , 0.70125445])

average_price_U_3_M_2_K_2_3 = np.array([0.63789063, 0.65964844, 0.67427604, 0.68148828, 0.68554632,
       0.6889375 , 0.69350957, 0.69685352, 0.69856641, 0.69919602,
       0.70024674, 0.70061735, 0.70138511, 0.70174071, 0.70230355])


average_price_U_3_M_2_K_2_s_2_1 = np.array([0.65559375, 0.66915625, 0.67873958, 0.67876875, 0.68171429,
       0.68329766, 0.68930469, 0.69274661, 0.69550215, 0.69693953,
       0.69866706, 0.70010993, 0.7013623 , 0.70151181, 0.70235844])

average_price_U_3_M_2_K_2_s_2_2 = np.array([0.6215    , 0.65632031, 0.68519531, 0.68486719, 0.68907701,
       0.69014141, 0.69282695, 0.69475156, 0.69812266, 0.69947172,
       0.69961849, 0.70012455, 0.7007291 , 0.70150365, 0.70161477])

average_price_U_3_M_2_K_2_s_2_3 = np.array([0.62510938, 0.63791406, 0.66945573, 0.6806375 , 0.68044754,
       0.68641875, 0.69560234, 0.69683307, 0.69809004, 0.69941703,
       0.70038424, 0.70078694, 0.70093574, 0.70085339, 0.70110672])


average_price_U_3_M_1_K_1_1_random = np.array([0.29466667, 0.2908125 , 0.29648958, 0.29508333, 0.29356845,
       0.29296771, 0.29252031, 0.29358264, 0.29333802, 0.29416687,
       0.29408229, 0.29459375, 0.29463893, 0.29466898, 0.2949675 ])

average_price_U_3_M_1_K_1_2_random = np.array([0.30195833, 0.29853125, 0.30030556, 0.29723958, 0.2931369 ,
       0.29198125, 0.29196823, 0.29007917, 0.29111094, 0.29104042,
       0.29142517, 0.29130804, 0.29114271, 0.29177106, 0.29216698])

average_price_U_3_M_1_K_1_3_random = np.array([0.27427083, 0.29517708, 0.29605208, 0.29595   , 0.2959494 ,
       0.29520729, 0.29317813, 0.29433924, 0.29395078, 0.29315167,
       0.29376076, 0.29374568, 0.29335794, 0.29318519, 0.29318844])

average_price_U_3_M_2_K_2_s_2_1_random = np.array([0.587825  , 0.5921375 , 0.58410417, 0.5845875 , 0.58126071,
       0.58234125, 0.58548281, 0.58612542, 0.58614562, 0.58597137,
       0.58606833, 0.58622295, 0.58577359, 0.58574083, 0.58571156])

average_price_U_3_M_2_K_2_s_2_1_fixed = np.array([0.423925  , 0.42358125, 0.42225417, 0.42326125, 0.42410268,
       0.42451812, 0.42331719, 0.42244875, 0.42368484, 0.42323375,
       0.42315333, 0.42347437, 0.42364047, 0.42309653, 0.42327875])


average_price_U_3_M_1_K_1_1_fixed = np.array([0.2415    , 0.23260417, 0.23393056, 0.234675  , 0.23177679,
       0.23309375, 0.23388542, 0.23364236, 0.23427135, 0.2339425 ,
       0.23416285, 0.2331747 , 0.23306953, 0.23268889, 0.23341604])

average_price_U_3_M_1_K_1_2_fixed = np.array([0.21554167, 0.2304375 , 0.23167361, 0.2295125 , 0.23248214,
       0.22826875, 0.22965208, 0.23005486, 0.2323276 , 0.23208375,
       0.2316125 , 0.23234167, 0.23256276, 0.23232593, 0.23221729])

average_price_U_3_M_1_K_1_3_fixed = np.array([0.24420833, 0.24066667, 0.2305625 , 0.23575417, 0.23396726,
       0.23350208, 0.23115625, 0.23215   , 0.23363646, 0.23354583,
       0.23277986, 0.23355179, 0.23342708, 0.23328032, 0.23362125])

# average_throughput_U_3_M_1_K_1_1_Hete = np.array([0.64967188, 0.66775   , 0.7069401 , 0.72774219, 0.74117187,
#         0.75534766, 0.7783957 , 0.79121641, 0.79891113, 0.80454922,
#         0.80947695, 0.81318996, 0.81575625, 0.81828802, 0.82035227])

# average_throughput_U_3_M_1_K_1_2_Hete = np.array([0.67390625, 0.68733594, 0.73654427, 0.763275  , 0.78117299,
#         0.79764609, 0.8312418 , 0.84744167, 0.85652402, 0.86285562,
#         0.8678263 , 0.87121551, 0.87379658, 0.87633698, 0.87827195])

# average_throughput_U_3_M_1_K_1_3_Hete = np.array([0.54507812, 0.55989063, 0.58271354, 0.59094687, 0.59865513,
#         0.60530547, 0.61933281, 0.62769505, 0.6307625 , 0.63196469,
#         0.63334401, 0.63448382, 0.63495166, 0.63506554, 0.63542344])

# average_throughput_U_3_M_2_K_2_1_Hete = np.array([1.3486    , 1.377475  , 1.46710417, 1.5075975 , 1.53094821,
#         1.55384125, 1.5800925 , 1.59532333, 1.60431312, 1.60985525,
#         1.61343042, 1.61655661, 1.61941406, 1.62171472, 1.62368238])

# average_throughput_U_3_M_2_K_2_2_Hete = np.array([1.3685    , 1.436075  , 1.54585   , 1.5950025 , 1.61973393,
#         1.6421925 , 1.67855187, 1.69563292, 1.70511125, 1.71196975,
#         1.71686062, 1.72046518, 1.72347375, 1.72585472, 1.72776   ])

# average_throughput_U_3_M_2_K_2_3_Hete = np.array([1.113275  , 1.1510875 , 1.20474167, 1.22564   , 1.23789464,
#         1.24763   , 1.26413687, 1.26895833, 1.27601625, 1.27800575,
#         1.27951854, 1.28077982, 1.28406344, 1.28464958, 1.28563863])





plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1,'-^',label = 'M=1,s=1,user 1')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2,'-o',label = 'M=1,s=1,user 2')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3,'-*',label = 'M=1,s=1,user 3')

plt.plot(regret_x_label, average_price_U_3_M_1_K_1_s_2_1,'-^',label = 'M=1,s=2,user 1')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_s_2_2,'-o',label = 'M=1,s=2,user 2')
plt.plot(regret_x_label, average_price_U_3_M_1_K_1_s_2_3,'-*',label = 'M=1,s=2,user 3')

plt.plot(regret_x_label, average_price_U_3_M_2_K_2_1,'-^',label = 'M=2,s=1,user 1')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_2,'-o',label = 'M=2,s=1,user 2')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_3,'-*',label = 'M=2,s=1,user 3')

plt.plot(regret_x_label, average_price_U_3_M_2_K_2_s_2_1,'-^',label = 'M=2,s=2,user 1')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_s_2_2,'-o',label = 'M=2,s=2,user 2')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_s_2_3,'-*',label = 'M=2,s=2,user 3')

plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1_random,'--^',label = 'M=1,s=1,2,user 1,2,3,random')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_s_2_1_random,'--^',label = 'M=2,s=1,2,user 1,2,3,random')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2_random,'--o',label = 'M=1,s=1,user 2,random')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3_random,'--*',label = 'M=1,s=1,user 3,random')

plt.plot(regret_x_label, average_price_U_3_M_1_K_1_1_fixed,'--x',label = 'M=1,s=1,2,user 1,2,3,fixed')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_2_fixed,'--.',label = 'M=1,s=1,user 2,fixed')
# plt.plot(regret_x_label, average_price_U_3_M_1_K_1_3_fixed,'--',label = 'M=1,s=1,user 3,fixed')
plt.plot(regret_x_label, average_price_U_3_M_2_K_2_s_2_1_fixed,'--x',label = 'M=2,s=1,2,user 1,2,3,fixed')

# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1,'-^',label = 'M=K=2,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2,'-o',label = 'M=K=2,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3,'-*',label = 'M=K=2,user 3')



# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_1_Hete,'-^',label = 'M=K=1,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_2_Hete,'-o',label = 'M=K=1,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_1_K_1_3_Hete,'-*',label = 'M=K=1,user 3')

# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_1_Hete,'-^',label = 'M=K=2,user 1')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_2_Hete,'-o',label = 'M=K=2,user 2')
# plt.plot(regret_x_label, average_throughput_U_3_M_2_K_2_3_Hete,'-*',label = 'M=K=2,user 3')


plt.legend(ncol=1,framealpha = 0,fontsize=8)
plt.xlabel('time',fontsize=14)
plt.ylabel('average price',fontsize=14)
# plt.ylim([0.19, 1.08])
plt.xlim([-800, 31000])
plt.grid()
# plt.ylim([0.5,2.3])
fig.savefig('Decentralized_average_price_1.eps', dpi = 600, format = 'eps')