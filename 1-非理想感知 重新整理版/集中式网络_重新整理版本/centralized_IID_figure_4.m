

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description : This is used to simulate the algorithm_1_1
%               
%               Author: xxx
%               Date  : 2019/03/22
%               Modified : 2023/06/01
% 在该仿真中 设置不同的 sample_num，记录对应的 estimated_prob，得到表3.1
% 值得说明，每次仿真得到的结果不同，实际上应该进行蒙特卡洛仿真取统计平均
% 但得到结果的趋势是相同的

% 记录 T 可以得到对不同信道选择的次数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
sample_num = 1000;  % 样本数量
% channel_free_prob = [0.5296,0.4001,0.9817,0.1931,0.2495]';  % 信道空闲概率
 
channel_free_prob = [ 0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994 ]';

% N = 14 
% channel_free_prob = [  0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994 ]';


 % N = 13 
% channel_free_prob = [ 0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017 ,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994 ]';



% channel_free_prob = [0.8457,0.9411,0.6287,0.2052,0.5259,0.1450,0.4818,0.7789]';
% channel_free_prob = sort(channel_free_prob,'descend');

% channel_free_prob = [0.9817,0.5296,0.4001,0.2495,0.1931]';  
N = length(channel_free_prob);
P_d = 0.8;  % 检测概率
P_fa = 0.3;  % 虚警概率：主用户未工作，检测判断为工作
Montecalo_num = 1;
theta_difference=zeros(1,sample_num); % 与设定的 theta 比较的累积量
M = 6; % 表示能感知的信道数量 U=3,M=2,K=1
K = 3; % 表示接入信道数量
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
       theta_difference(t) = theta_difference(t) + norm(estimated_prob'- channel_free_prob);
       
       %% 寻找 M 条信道去sense 记录在 M_sensed_channels
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
           temp_indexes(i) = (Y(M_sensed_channels(i))/T(M_sensed_channels(i)) + P_d-1)/(P_d - P_fa) + sqrt(2*log(t-1)/T(M_sensed_channels(i)) )/(P_d-P_fa);
%            temp_indexes(i) = (Y(M_sensed_channels(i))/T(M_sensed_channels(i)) + P_d-1) + sqrt(2*log(t-1)/T(M_sensed_channels(i)) );
       end
 
       
       for i = 1:M
           if observation(M_sensed_channels(i),t) < 1
               temp_indexes(i) = -1000;               % 得到 M 条信道 '\theta+sqrt'，如果用户观察为0，该信道为不可用
           end
       end
       %% 将 M 条信道按照 '\theta+sqrt' 进行排序，选取前 K 条信道进行传输
%        [M_sensed_channels_value M_sensed_channels_sequence] = sort(temp_indexes,'descend');
%        busy_index = find(M_sensed_channels_value == -1000); % 找到busy的信道，前面所有(busy_index-1)个信道为可以传输信息的信道
%        % 与最大接入信道数量 K 进行比较，可接入信道数量大于 K 时，选最大的 K 个信道用来传输
%        idle_number = 0;
%        idle_number = length(M_sensed_channels_value) - length(busy_index);
%        if K <= idle_number
%            channel_chosen_access(1:K) = M_sensed_channels_sequence(1:K);
%        else
%            channel_chosen_access(1:idle_number) = M_sensed_channels_sequence(1:idle_number);
%        end
        cnt_K_access = 1;
        flag = 0;
        while (cnt_K_access < K+1 && flag < 1)
            flag = 1;
            for j = 1:M
                if temp_indexes(j)>-1000
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
                Reward(t) = Reward(t) + channel_state(channel_chosen_access(i),t);
            end
        end
        
        if mod(t, 10000)==0
             fprintf('sample_num = %d,n = %d,t = %d\n',sample_num,n,t );
        end   
    end
end

%% Benchmark: 首先求得 M*， 选出最好的 M 条信道，找出空闲的信道，并接入<=K条中

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
Reward_expect_all_status = Reward_expect_all_status * (1-P_fa);  



%% 所需数据：本节考虑的是不同时隙下对信道空闲概率的估计，因此记录 sample_num = 100、500、1000、2000 下的 estimated_prob

fprintf('the simulation time is');
sample_num

fprintf('the estimated idle probability is');
estimated_prob

fprintf('the chosen number is');
T

%% 求得各种数据
% Reward_ALL = 0;
% Regret = 0;
% Regret_as_time = zeros(1,sample_num);
% Reward_average = Reward/Montecalo_num;
% theta_difference_average = theta_difference/Montecalo_num;
% Reward_average_as_time = zeros(1,sample_num);
% temp_for_Reward = 0;
% for t1 = 1:sample_num
%     temp_for_Reward = temp_for_Reward + Reward_average(t1);
%     Reward_average_as_time(t1) =  temp_for_Reward/t1;
% end
% Reward_ALL = temp_for_Reward;
% Reward_expect_ALL = Reward_expect_all_status * sample_num;
% Regret = Reward_expect_ALL - Reward_ALL;
% Regret_log = Regret/log(sample_num)
 


%% plot record
% Num_record = [ round(0.1*sample_num), round(0.2*sample_num),round(0.3*sample_num),round(0.4*sample_num),round(0.5*sample_num),round(0.6*sample_num),round(0.7*sample_num),round(0.8*sample_num),round(0.9*sample_num),round(1*sample_num) ]
% Reward_record = [];
% Regret_record = [];
% for i = 1:length(Num_record)
%     Reward_record(i) = Reward_average_as_time(Num_record(i));
%     Regret_record(i) = ( Reward_expect_all_status - Reward_record(i) ) * Num_record(i)/log(Num_record(i) );
% end   
% 
% figure;
% plot(Num_record, Reward_expect_all_status, 'r','LineWidth',2);
% hold on;
% plot( Num_record , Reward_record, 'r');
% hold on;

% plot( Num_record , Regret_record, 'r')


% sample_num_plot = [ 40 60 80 100 200 300 400 600 800 1000 2000 3000 4000 5000 6000] ;
% Regret_log_plot = [ 2.1867 2.4553 2.7041 3.0435 3.4682 3.8959 4.2773 4.7569 5.1967 5.5564 6.861 7.584 7.63527 7.6557 7.6914];

%% 画图
% figure;
% t = [1:sample_num];
% plot(t, Reward_average_as_time,'b');
% xlabel('time');
% ylabel('reward')
% hold on;
% plot(1:sample_num, Reward_expect_all_status * ones(1,sample_num), 'r');
% toc
%% 画图 algorithm_1_1
% figure;
% xlabel('time');
% ylabel('Regret/log(t)')
% hold on;
% plot(sample_num_plot, Regret_log_plot, 'r*--','LineWidth',2);
% toc  
      
       
 
                   
           
           
 
               
       



