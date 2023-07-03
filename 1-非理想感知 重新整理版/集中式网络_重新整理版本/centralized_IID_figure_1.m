

%--------------------------------------------------------------------------------------------------------------------------------------------------%
% Description : This is used to simulate the centralized algorithm in Chap 3.2
%               
%               Author: xxx
%               Date  : 2019/03/22
%               Modified : 2023/06/01
%
% 集中式网络中存在中心节点调度各次要用户进行感知接入，仿真时可转换为单个次要用户感知接入多条信道
% 仿真说明：
    %% 1、可在参数设置中调整 U、M_u、K_u 得到不同条件下的仿真结果。为便于读者理解，将 U=2、M_u=3、K_u=2 仿真条件下的仿真程序详细给出

    %% 2、调整参数 P_d=0.7、0.8; P_f=4.1、0.2、0.3 获得不同的仿真结果

    %% 3、不同信道数量下的仿真结果
    % N = 15  
    % channel_free_prob = [ 0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 , 0.5580, 0.4067, 0.8799, 0.7994 ]';
    % N = 14 
    % channel_free_prob = [  0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904, 0.5580, 0.4067 ,0.8799  ]';
    % N = 13 
    % channel_free_prob = [ 0.0499, 0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017 , 0.1354, 0.9707,0.4904 , 0.5580, 0.4067  ]';

    %% 4、 不同时隙下对信道空闲概率的估计，可调整 sample_num = 100、500、1000、2000 分别得到
    
    %% 5、 不同时隙下对信道选取的次数，在算法仿真中变量 T 表示每条信道感知的数量， Y 表示信道接入的数量，记录后可画图得到仿真结果
    
%--------------------------------------------------------------------------------------------------------------------------------------------------%
 
clc
clear
tic

%% Parameter Setting
sample_num = 100000;  % 样本数量
channel_free_prob = [  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1  ]'; % 信道空闲概率
channel_free_prob = sort(channel_free_prob,'descend');
N = length(channel_free_prob); % 信道数量
P_d = 0.8;  % 检测概率
P_fa = 0.3;  % 虚警概率：主用户未工作，检测判断为工作
Montecalo_num = 65; % 蒙特卡洛仿真次数
theta_difference=zeros(1,sample_num); % 与设定的 theta 比较的累积量
U = 2; % 表示集中式网络中的次要用户数量
M_u = 3; % 每个次要用户能够感知的信道数量
K_u = 1; % 每个次要用户能够接入的信道数量
M = U * M_u; % 表示能感知的信道数量 
K = U * K_u; % 表示接入信道数量 
Reward = zeros(1,sample_num); 
 
%% Simulation
for n = 1:Montecalo_num
    uniformRV= rand(N,sample_num);            % 信道实际空闲概率
    uniformRV_sense = rand(N,sample_num);     % 用户观测空闲概率
    channel_state = zeros(N,sample_num);      % 信道状态： ‘1’时为空闲，‘0’时为busy
    observation = zeros(N,sample_num);        % 用户观察得到的信道状态
    %% 初始化
    for j = 1:sample_num
        channel_state(:,j) = (uniformRV(:,j)<channel_free_prob);   % 信道实际状态
        observation(:,j) = channel_state(:,j) .* ( uniformRV_sense(:,j) < 1-P_fa ) ...
                           + (1 - channel_state(:,j)) .* ( uniformRV_sense(:,j) < 1-P_d);    % 观察得到的信道状态
    end
    estimated_prob = zeros(1,N);      % 估计信道空闲概率
    indexes = zeros(1,N);             % 信道编号
    T = ones(1,N);                    % 每个信道被观测了一次
    Y = observation(:,1)';

    %% 开始   
    for t = 2:sample_num
       for i = 1:N
           estimated_prob(i) = (Y(i)/T(i) + (P_d-1))/(P_d-P_fa);
           indexes(i) = estimated_prob(i) + sqrt( 2*log(t-1)/T(i) ) /(P_d-P_fa); 
       end

       %% 感知 M 条信道 记录在 M_sensed_channels
       M_sensed_channels = zeros(1,M);
       temp = indexes;
       [value sequence] = sort(temp,'descend');  % sequence 中的前M个为最大的前M条信道
       M_sensed_channels = sequence(1:M);
       %% 感知前 M 条信道，更新 T 和 Y
       for i = 1:M
           T(M_sensed_channels(i)) = T(M_sensed_channels(i)) + 1;
           if observation(M_sensed_channels(i),t-1) > 0
               Y(M_sensed_channels(i)) = Y(M_sensed_channels(i)) + 1;
           end
       end
       
       channel_chosen_access = zeros(1,K);
 
       for i = 1:M
           temp_indexes(i) = temp(i);
       end
             
       for i = 1:M
           if observation(M_sensed_channels(i),t) < 1
               temp_indexes(i) = -1000;               % 得到 M 条信道 '\theta+sqrt'，如果用户观察为0，该信道为不可用
           end
       end
       %% 接入最多 K 条信道

        cnt_K_access = 1;
        flag = 0;
        while (cnt_K_access < K+1 && flag < 1)
            flag = 1;
            for j = 1:M
                if temp_indexes(j)>-1000  % 若无可用信道，则放弃该次接入过程
                    flag = 0;
                    continue;
                end
            end
            for j = 1:M
                if (temp_indexes(j) == max(temp_indexes) && temp_indexes(j)>-1000 && cnt_K_access < K+1)
                    channel_chosen_access(cnt_K_access) = M_sensed_channels(j);
                    cnt_K_access = cnt_K_access + 1;
                    temp_indexes(j) = -1000;
                end
            end
        end
        
        for i = 1:K
            if channel_chosen_access(i) > 0  
                Reward(t) = Reward(t) + channel_state(channel_chosen_access(i),t);  % 接入信道获得收益 
            end
        end
        
        if mod(t, 10000)==0
             fprintf('sample_num = %d,n = %d,t = %d\n',sample_num,n,t );
        end   
    end
end

%% Benchmark: 在信道空闲概率已知情况下的最优感知接入方法，即$\mathbb{E}\{ U^{\pi^{*C}}(t) \} $
% 需要注意的是，每时隙中每条信道的空闲--占用状态为随机变量，即选择 M 条信道时，每时隙中有
% 2^M 种信道状态的组合方式，每种信道组合方式在每时隙中以一定概率出现，因此需对获得的吞吐量求期望

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
 

for channel_combination_status = 0:2^M-1  % channel_combination_status 表示 M 条信道不同的组合方式
    channel_combination_status_dec2bin = dec2bin(channel_combination_status); % 将channel_combination_status 转换为二进制，表征每条信道状态
    while length(channel_combination_status_dec2bin) < M
        channel_combination_status_dec2bin = strcat('0',channel_combination_status_dec2bin);
    end

    % channel_combination_status_dec2bin 此时该变量已经为 长度 为 M 的char类型字符串，表征信道状态
    channel_combination_status_dec2bin_char2double = double(uint8(channel_combination_status_dec2bin)>48);  % 将char类型转换为 double 类型
    % Sensing_result 表示每一个信道组合存在的概率
    Sensing_result = 1;

    % 求 M* 中每个信道组合状态下的传输概率——为求外期望
    for j = 1:M
        if channel_combination_status_dec2bin_char2double(j)>0
            Sensing_result = Sensing_result * f_theta_M_best(j);
        else
            Sensing_result = Sensing_result * (1-f_theta_M_best(j));
        end
    end

    cnt_channel_access = 0; % 表示接入信道数量 计数 K 个
    Reward_single_status = 0;
    for k = 1:M
        if cnt_channel_access < K
            if channel_combination_status_dec2bin_char2double(k) > 0 
                Reward_single_status = Reward_single_status + channel_free_prob(f_theta_M_best_sequence(k))/f_theta_M_best(k);  % 信道 k 的条件reward， \hat{\theta_i} / f(\hat{\theta})
                cnt_channel_access = cnt_channel_access + 1;
            end
        end
    end
    Reward_expect_all_status = Reward_expect_all_status + Reward_single_status * Sensing_result;
end
Reward_expect_all_status = Reward_expect_all_status * (1-P_fa);  % 最终得到的reward



%% 仿真得到的数据
Reward_ALL = 0;
Regret = 0;
Regret_as_time = zeros(1,sample_num); % regret/ln t 的形式
Regret_as_time_no_normalized = zeros(1,sample_num); % 仅有 regret 的形式 两者区别主要在纵坐标的scale，吞吐量损失的趋势相同 

Reward_average = Reward/Montecalo_num;
theta_difference_average = theta_difference/Montecalo_num;
Reward_average_as_time = zeros(1,sample_num);
temp_for_Reward = 0;
for t1 = 1:sample_num
    temp_for_Reward = temp_for_Reward + Reward_average(t1);
    Reward_average_as_time(t1) =  temp_for_Reward/t1;
    Regret_as_time(t1) = (Reward_expect_all_status * t1 - temp_for_Reward)/log(t1);
    Regret_as_time_no_normalized(t1) = (Reward_expect_all_status * t1 - temp_for_Reward);
end
 
% 平均吞吐量
figure;
plot(1:sample_num,Reward_average_as_time,'b');
hold on;
plot(1:sample_num,Reward_expect_all_status*ones(1,sample_num),'r');
hold off;

% 归一化的吞吐量损失，即regret/log t 形式。该形式与未归一化的形式趋势相同，区别在于纵坐标。
figure;
plot(1:sample_num,Regret_as_time,'b');
hold off;

% 未归一化的吞吐量损失，即采用 regret 的形式表示仿真结果。
figure;
plot(1:sample_num,Regret_as_time_no_normalized,'b');
hold off;

%% 将仿真得到的数据存储，后续使用python等工具进行统一画图

% save 'U_4_M_1_K_1_regret.mat' Regret_as_time
% save 'U_4_M_1_K_1_reward.mat' Reward_average_as_time
% save 'U_4_M_1_K_1_reward_optimal.mat' Reward_expect_all_status

 
toc  
      
       
 
                   
           
           
 
               
       



