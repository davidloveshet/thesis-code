# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:55:36 2021

@author: xxx

输入判决矩阵，AHP 方法求得各 w_s

"""

import numpy as np
class AHP:
    #传入的np.ndarray是的判断矩阵
    def __init__(self,array):
        self.array = array
        # 记录矩阵大小
        self.n = array.shape[0]
        # 初始化RI值,用于一致性检验 
        RI_list = [0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45]
        self.RI = RI_list[self.n-1]
        
	#获取最大特征值和对应的特征向量
    def get_eig(self):
        #numpy.linalg.eig() 计算矩阵特征值与特征向量
        eig_val ,eig_vector = np.linalg.eig(self.array)
        #获取最大特征值
        max_val = np.max(eig_val)
        max_val = round(max_val.real, 4)
        #通过位置来确定最大特征值对应的特征向量
        index = np.argmax(eig_val)
        max_vector = eig_vector[:,index]
        max_vector = max_vector.real.round(4) 
        #添加最大特征值属性
        self.max_val = max_val
        #计算权重向量W
        weight_vector = max_vector/sum(max_vector)
        weight_vector = weight_vector.round(4)
        #打印结果
        print("最大的特征值: "+str(max_val))
        print("对应的特征向量为: "+str(max_vector))
        print("归一化后得到权重向量: "+str(weight_vector))
        return weight_vector
    
    #测试一致性
    def test_consitst(self):
        #计算CI值
        CI = (self.max_val-self.n)/(self.n-1) 
        CI = round(CI,4) 
        #打印结果
        print("判断矩阵的CI值为" +str(CI))
        print("判断矩阵的RI值为" +str(self.RI))
        #分类讨论
        if self.n == 2:
            print("仅包含两个子因素，不存在一致性问题")
        else:
            #计算CR值
            CR = CI/self.RI 
            CR = round(CR,4)
            #CR < 0.10才能通过检验
            if  CR < 0.10 :
                print("判断矩阵的CR值为" +str(CR) + "，通过一致性检验")
                return True
            else:
                print("判断矩阵的CR值为" +str(CR) + "，未通过一致性检验")
                return False
            
            
            
# The followings are test examples and we obtain the weights             
            
            
# matrix_s = np.array([  [1, 9/7, 3, 9 ],   [7/9, 1, 2,3],   [1/3, 1/2, 1, 2],  [1/9, 1/3, 1/2, 1 ]     ])           


# matrix_s_1 = np.array([ [1, 2, 4, 8], [ 1/2, 1, 2, 4], [1/4, 1/2, 1, 2 ], [1/8, 1/4, 1/2, 1]  ]) 

# matrix_s_2 = np.array([ [1, 9/7, 3, 9], [7/9, 1, 7/3, 7], [1/3, 3/7, 1, 4], [ 1/9, 1/7, 1/4, 1] ])

# matrix_s_3 = np.array([  [1,1,7/3,7/5], [1,1,7/3,7/5], [3/7, 3/7, 1, 3/5], [ 5/7, 5/7, 5/3, 1]  ])

# # matrix_s_3 = np.array([  [1,79/14,79/68,79/73], [14/79,1,14/68,14/73], [68/79, 68/14, 1, 68/73], [ 73/79, 73/14, 73/68, 1]  ])

# AHP_s_1 = AHP(matrix_s)   
# AHP_s_1.get_eig()
# AHP_s_1.test_consitst()
         
            
# matrix_s = np.array([  [1, 9/7, 9/5, 9/3, 9/1 ],   [7/9, 1, 7/5, 7/3, 7/1],   [ 5/9, 5/7, 1, 5/3, 5/1 ],  [ 3/9, 3/7, 3/5, 1, 3/1 ], [1/9, 1/7, 1/5, 1/3, 1]     ])      
# AHP_s_1 = AHP(matrix_s)            
# AHP_s_1.get_eig()
# AHP_s_1.test_consitst()           
            
# matrix_s = np.array([  [1, 9/3, 9/5, 9/3, 9/3 ],   [3/9, 1, 3/5, 7/3, 7/1],   [ 5/9, 5/3, 1, 5/3, 5/1 ],  [ 3/9, 3/7, 3/5, 1, 1/1 ], [3/9, 1/7, 1/5, 1/1, 1]     ])      
# AHP_s_1 = AHP(matrix_s)            
# AHP_s_1.get_eig()
# AHP_s_1.test_consitst()          
            
            
# matrix_s = np.array([  [1, 7/9, 7/9, 7/5, 7/3 ],   [7/9, 1, 5/4, 5/5, 5/3],   [ 7/9, 4/5, 1, 7/5, 7/3 ],  [ 5/7, 1, 5/7, 1, 1/1 ], [3/7, 3/5, 3/7, 1/1, 1]     ])      
# AHP_s_1 = AHP(matrix_s)            
# AHP_s_1.get_eig()
# AHP_s_1.test_consitst()           
            
            
matrix_s = np.array([  [1, 1, 5],  
                       [1, 1, 5],
                       [ 1/5, 1/5, 1] ])      
AHP_s_1 = AHP(matrix_s)            
AHP_s_1.get_eig()
AHP_s_1.test_consitst()          
            
  
            
  
# matrix_s = np.array([  [1,1],  
#                         [1,1] ])      
# AHP_s_1 = AHP(matrix_s)            
# AHP_s_1.get_eig()
# AHP_s_1.test_consitst()          
  
    
matrix_s = np.array([  [1, 1, 0],  
                       [1, 1, 0],
                       [ 0, 0, 1] ])      
AHP_s_1 = AHP(matrix_s)            
AHP_s_1.get_eig()
AHP_s_1.test_consitst()     
    
  
    
matrix_s = np.array([  [1, 1, 1/3],  
                       [1, 1, 1/3],
                       [3, 3, 1] ])      
AHP_s_1 = AHP(matrix_s)            
AHP_s_1.get_eig()
AHP_s_1.test_consitst()   
    
  


























            
            