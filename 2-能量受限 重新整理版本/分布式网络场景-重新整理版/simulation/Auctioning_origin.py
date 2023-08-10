import random
from random import shuffle
from hungarian import Hungarian

"""
The following is code impliments both the naive and sophisticated auctioning algorithm which is based on the MIT paper "Auction Algorithms" also included. The goal of this algorithm is to take a game theory approach (Thinking of something similar to the matching problem), attempting to find the optimal payoff matching for the agents, considering their values of objects, and the costs of the objects the agents are bidding on.
"""
# random.seed(1)

def create_matrices(number_of_agents):
    """
    Create matrices of agents and their values for varying objects
    """
    matrix = []
    agent = 0
    while agent != number_of_agents:
        temp_matrix = []
        for prices in range(0,number_of_agents):
            temp_matrix.append(round(random.uniform(1, 15),0))
        matrix.append(temp_matrix)
        agent += 1
    prices_list = []
    for prices in range(0,number_of_agents):
        prices_list.append(round(random.uniform(1, 10),0))
    return matrix, prices_list

def assign_values(number_of_agents):
    """
    Assign the values to each of the agents
    """
    temp_list = []
    agents = []
    for i in range(0,number_of_agents):
        temp_list.append(i)
    shuffle(temp_list)
    for p in range(0,number_of_agents):
        agents.append([temp_list[p],0])
    return agents

def get_best_option(agent_number, matrix, costs):
    """
    Find the best option for the given agent
    """
    diff_list = []
    for i in range(0,len(costs)):
        diff_list.append(matrix[agent_number][i] - costs[i])
    highest_val = max(diff_list)
    ind_highest_val = diff_list.index(highest_val)
    diff_list[ind_highest_val] = -1000000000
    second_highest_val = max(diff_list)
    ind_second_highest_val = diff_list.index(max(diff_list))
    return highest_val, ind_highest_val, second_highest_val, ind_second_highest_val

def find_index_of_val(mylist, value):
    """
    Find index of given value in list
    """
    l = [i[0] for i in mylist]
    if value in l:
        return l.index(value) 
    else:
        return []
    # 如果有值，则进行交换，如果没有值，则进行满意

def check_happiness(agent_matrix,payoff,cost):
    """
    Check if agents are happy with their current situation
    """
    for i in range(0,len(agent_matrix)):
        if agent_matrix[i][1] == 0:
            high,ind_high,sec_high,sec_ind_high = get_best_option(i,payoff_matrix,cost_list)
            if ind_high == agent_matrix[i][0]:
                agent_matrix[i][1] = 1

# def naive_auction(agents,payoff_matrix,cost_list):
#     """
#     Running the naive version of the auction algorithm where we do not force agents to increase bids by epsilon each time they bid
#     """
#     while sum(n for _, n in agents) != len(agents):
#         for i in range(0,len(agents)):
#             check_happiness(agents,payoff_matrix,cost_list)
#             if agents[i][1] == 0:
#                 high,ind_high,sec_high,sec_ind_high = get_best_option(i, payoff_matrix, cost_list)
#                 switch_index = find_index_of_val(agents, int(ind_high))
#                 agents[switch_index][1] = 0
#                 agents[i][1] = 1
#                 agents[switch_index][0] = agents[i][0]
#                 agents[i][0] = ind_high
#                 cost_list[ind_high] = cost_list[ind_high] + abs(((payoff_matrix[i][ind_high] - cost_list[ind_high]) - (payoff_matrix[i][sec_ind_high] - cost_list[sec_ind_high])))
#     for agent in range(0,len(cost_list)):
#         print("Agent {} will pay ${} for object {}. Their original value of this object was ${}".format(agent,cost_list[agents[agent][0]], agents[agent][0], payoff_matrix[agent][agents[agent][0]]))

# def sophisticated_auction(epsilon, agents, payoff_matrix, cost_list):
#     """
#     Running the sophisticated version of the auction algorithm where we force agents to increase bids by epsilon each time they bid
#     """
#     while sum(n for _, n in agents) != len(agents):
#         for i in range(0,len(agents)):
#             check_happiness(agents, payoff_matrix, cost_list)
#             if agents[i][1] == 0:
#                 high,ind_high,sec_high,sec_ind_high = get_best_option(i,payoff_matrix,cost_list)
#                 switch_index = find_index_of_val( agents, int(ind_high) )
#                 agents[switch_index][1] = 0
#                 agents[i][1] = 1 
#                 agents[switch_index][0] = agents[i][0]
#                 agents[i][0] = ind_high
#                 cost_list[ind_high] = cost_list[ind_high] + abs(((payoff_matrix[i][ind_high] - cost_list[ind_high]) - (payoff_matrix[i][sec_ind_high] - cost_list[sec_ind_high]))) + epsilon
#     for agent in range(0,len(cost_list)):
#         print("Agent {} will pay ${} for object {}. Their original value of this object was ${}".format(agent,round(cost_list[agents[agent][0]],2), agents[agent][0], payoff_matrix[agent][agents[agent][0]]))


def sophisticated_auction(epsilon, agents, payoff_matrix, cost_list):
    """
    Running the sophisticated version of the auction algorithm where we force agents to increase bids by epsilon each time they bid
    """
    item_Num = 0
    while sum(n for _, n in agents) != len(agents):
        for i in range(0,len(agents)):
            check_happiness(agents, payoff_matrix, cost_list)
            item_Num = item_Num + 1
            if agents[i][1] == 0:
                high,ind_high,sec_high,sec_ind_high = get_best_option(i, payoff_matrix, cost_list)
                switch_index = find_index_of_val( agents, int(ind_high) )
                if switch_index == []: # 如果为 switch_index 没有用户占用，则不需要交换
                    agents[i][1] = 1
                    agents[i][0] = ind_high
                else:
                    agents[switch_index][1] = 0
                    agents[i][1] = 1 
                    agents[switch_index][0] = agents[i][0]
                    agents[i][0] = ind_high
                    cost_list[ind_high] = cost_list[ind_high] + abs(((payoff_matrix[i][ind_high] - cost_list[ind_high]) - (payoff_matrix[i][sec_ind_high] - cost_list[sec_ind_high]))) + epsilon
    return item_Num
    # for agent in range(0,len(cost_list)):
    #     print("Agent {} will pay ${} for object {}. Their original value of this object was ${}".format(agent,round(cost_list[agents[agent][0]],2), agents[agent][0], payoff_matrix[agent][agents[agent][0]]))


# import numpy as np
# N = 10
# epsilon = 0.01
# agents = assign_values(N)
# payoff_matrix, cost_list = create_matrices(N)
# payoff_matrix, cost_list = create_matrices(N)
# cost_list = np.zeros((len(agents),N))


# # agents = [[0,0],[2,0],[1,0]]
# # payoff_matrix = [[7,9,12],[10,6,7],[9,13,5]]
# # cost_list = [6,3,5]

# sophisticated_auction(epsilon,agents,payoff_matrix,cost_list)





# import numpy as np

'''
# 实验1
U = 6
N = 10
epsilon = 0.005
agents = assign_values(U)
payoff_matrix = np.random.random((U,N)).tolist()
cost_list = np.zeros(N).tolist()
'''

'''
U = 3
N = 5
epsilon = 0.005
agents = assign_values(U)
payoff_matrix = [ [0,1,2,3,4], [4,1,3,2,0], [3,2,1,4,0] ]
cost_list = np.zeros(N).tolist()
'''



import numpy as np
U = 10
N = 15

agents = assign_values(U)
payoff_matrix = np.random.random((U,N))/10

round_bit = 3
epsilon = 10**(-round_bit)/U

a = [0.03047991, 0.07829985, 0.07910911, 0.06932115, 0.04767608,
        0.06911233, 0.08040998, 0.08001767, 0.05070302, 0.037563  ,
        0.00871545, 0.08084166, 0.03718937, 0.02120145, 0.0444637 ]

b = [0.03045991, 0.07849985, 0.05910911, 0.06933115, 0.04767708,
        0.06941233, 0.08046998, 0.086001767, 0.075070302, 0.0375563  ,
        0.008715345, 0.080841266, 0.037185937, 0.021205145, 0.04544637 ]

c = [0.0031834 , 0.09432094, 0.02504857, 0.08595762, 0.09344346,
        0.01166003, 0.09088026, 0.08001752, 0.04531868, 0.03821867,
        0.09589523, 0.07743549, 0.00148112, 0.03095393, 0.04124315]

d = [0.04476202, 0.04441169, 0.01640232, 0.05517897, 0.06234385,
        0.09901189, 0.01754748, 0.05457877, 0.08174805, 0.01477799,
        0.01495669, 0.07167825, 0.02839954, 0.03904074, 0.04413727]

e = [0.07963718, 0.09897221, 0.09844144, 0.06662802, 0.05660827,
        0.04911791, 0.09925096, 0.0537918 , 0.08529276, 0.03874968,
        0.02689463, 0.02110793, 0.02887496, 0.05908473, 0.0365867 ]

r_6 = [0.05408449, 0.07978146, 0.00663122, 0.06858369, 0.04975201,
        0.0792292 , 0.04504756, 0.02568289, 0.07304227, 0.09370058,
        0.07843574, 0.07727225, 0.02131152, 0.00124547, 0.00014914]

r_7 = [0.04808834, 0.08030137, 0.05724753, 0.01966888, 0.0922376 ,
        0.07652051, 0.03177763, 0.02651273, 0.05145662, 0.07878805,
        0.05799768, 0.03818961, 0.02592525, 0.06994451, 0.05537327]

r_8 = [0.00296588, 0.05961416, 0.09225893, 0.02052477, 0.03749729,
        0.07353831, 0.04125098, 0.07678576, 0.05988287, 0.07472688,
        0.00791454, 0.06365293, 0.04069177, 0.07539366, 0.0461151 ]

r_9 = [0.05052845, 0.00613586, 0.00758589, 0.00762842, 0.0194429 ,
        0.05300345, 0.0418968 , 0.08914731, 0.00130316, 0.07449692,
        0.05194923, 0.08734783, 0.06407258, 0.03602464, 0.07802836]

r_10 = [0.02394134, 0.08004786, 0.06764164, 0.03020507, 0.05295933,
        0.05703768, 0.06328666, 0.04412107, 0.03425445, 0.05922649,
        0.03723675, 0.00460293, 0.06539875, 0.02228677, 0.02571492]


a = [ round(a[i],round_bit) for i in range(len(a)) ]
b = [ round(b[i],round_bit) for i in range(len(b)) ]
c = [ round(c[i],round_bit) for i in range(len(c)) ]
d = [ round(d[i],round_bit) for i in range(len(d)) ]
e = [ round(e[i],round_bit) for i in range(len(e)) ]
r_6 = [ round(r_6[i],round_bit) for i in range(len(r_6)) ]
r_7 = [ round(r_7[i],round_bit) for i in range(len(r_7)) ]
r_8 = [ round(r_8[i],round_bit) for i in range(len(r_8)) ]
r_9 = [ round(r_9[i],round_bit) for i in range(len(r_9)) ]
r_10 = [ round(r_10[i],round_bit) for i in range(len(r_10)) ]

payoff_matrix = np.array(
      [a,
       b,
       c,
       d,
       e,
       r_6,
       r_7,
       r_8,
       r_9,
       r_10
       ] 
      )

# [ [0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4] ]

cost_list = np.zeros(N).tolist()



Iter_Num = sophisticated_auction(epsilon, agents, payoff_matrix, cost_list)
print(Iter_Num)

results = []
for i in range(len(agents)):
    results.append( [ i,agents[i][0] ])


hungarian = Hungarian()
hungarian.calculate(payoff_matrix, is_profit_matrix=True)
allocation_relationship = hungarian.get_results()


print(results)

print(allocation_relationship)









