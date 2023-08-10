%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description : This is used to obtain the benchmark when the statistical
% information is known by users
%               
%               Author: xxx
%               Date  : 2019/03/22
%               Modified: 2023/05/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 仿真实验数据
channel_free_prob = [  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1  ]'; % 信道空闲概率

channel_free_prob = [0.8811, 0.5390,0.3468,0.9522,0.7823,0.0471,0.7968]';   

N = length(channel_free_prob); % 信道向量长度
M = 2; % 表示感知信道数量
K = 1; % 表示信道接入数量
P_d = 0.8;  % 检测概率
P_fa = 0.3;  % 虚警概率：主用户未工作，检测判断为工作

%% Benchmark: 首先求得 M*， 选出最好的 M 条信道，找出空闲的信道，并接入K条信道中
f_theta = zeros(1,N); % 求出每条信道的观察空闲概率
f_theta_sort = zeros(1,N); % 每条信道的观察空闲概率 降序排序
f_theta_sort_sequence = zeros(1,N); % 每条信道的观察空闲概率 降序排序序号
f_theta_M_best = zeros(1,M); % 最优的 M 条信道
f_theta_M_best_sequence = zeros(1,M); % 最优的 M 条信道序号
Reward_expect_all_status = 0; % 最优的 M 条信道下接入信道的throughout
for i = 1:N
    f_theta(i) = (1-P_fa) * channel_free_prob(i) + (1-P_d) * (1-channel_free_prob(i));
end

[f_theta_sort f_theta_sort_sequence] = sort(f_theta,'descend');  % 对信道观察空闲概率进行排序
f_theta_M_best = f_theta_sort(1:M); % 找出 M 条最优信道
f_theta_M_best_sequence = f_theta_sort_sequence(1:M); % 记录 M 条最优信道序号

% f_theta_M_best(1) = f_theta_sort(1);
% f_theta_M_best(2) = f_theta_sort(6);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(1); % 记录 M 条最优信道序号
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(6); 
% 
% f_theta_M_best(1) = f_theta_sort(2);
% f_theta_M_best(2) = f_theta_sort(5);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(2); % 记录 M 条最优信道序号
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(5); 
% 
% f_theta_M_best(1) = f_theta_sort(3);
% f_theta_M_best(2) = f_theta_sort(4);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(3); % 记录 M 条最优信道序号
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(4); 


for channel_combination_status = 0:2^M-1  % channel_combination_status 表示 M 条信道不同的组合方式
    channel_combination_status_dec2bin = dec2bin(channel_combination_status); % 将channel_combination_status 转换为二进制，表征每条信道状态
    while length(channel_combination_status_dec2bin) < M
        channel_combination_status_dec2bin = strcat('0',channel_combination_status_dec2bin);
    end
%     channel_combination_status_dec2bin
    % channel_combination_status_dec2bin 此时该变量已经为 长度 为 M 的char类型字符串，表征信道状态
    channel_combination_status_dec2bin_char2double = double(uint8(channel_combination_status_dec2bin)>48);  % 将char类型转换为 double 类型
    % Sensing_result 表示每一个信道组合存在的概率
    Sensing_result = 1;
%     channel_combination_status_dec2bin_char2double
    % 求 M* 中每个信道组合状态下的传输概率——为求外期望
    for j = 1:M
        if channel_combination_status_dec2bin_char2double(j)>0
            Sensing_result = Sensing_result * f_theta_M_best(j);
        else
            Sensing_result = Sensing_result * (1-f_theta_M_best(j));
        end
    end
%     Sensing_result
    cnt_channel_access = 0; % 表示接入信道数量 计数 K 个
    Reward_single_status = 0;
    for k = 1:M
        if cnt_channel_access < K
            if channel_combination_status_dec2bin_char2double(k) > 0 
                Reward_single_status = Reward_single_status + channel_free_prob(f_theta_M_best_sequence(k))/f_theta_M_best(k);  % 信道 k 的条件reward， \hat{\theta_i} / f(\hat{\theta})
%                 Reward_single_status = Reward_single_status + channel_free_prob(k)/f_theta(k) ; 
                cnt_channel_access = cnt_channel_access + 1;
            end
        end
    end
    Reward_expect_all_status = Reward_expect_all_status + Reward_single_status * Sensing_result;
end
Reward_expect_all_status = Reward_expect_all_status * (1-P_fa)





