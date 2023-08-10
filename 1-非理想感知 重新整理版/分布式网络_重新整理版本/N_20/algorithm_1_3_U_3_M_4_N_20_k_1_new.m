%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            Algorithm 1.3
%    多用户分布接入算法 Algorithm 1.3，该方法前提假设信道数量 N 大于用户数量
%                         NOTICE: N must > U * M
%
%                   Author: xxx
%                   Date  : 2019/04/18
%                       重新写一遍的算法
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
sample_num_iter = 20000 ;  % 样本数量

channel_free_prob = [0.0965,0.1320,0.9221,0.9861,0.5352,0.0598,0.2348,0.3532,0.8612,0.0154,0.0430,0.1690,0.6891,0.7317,0.6477,0.4709,0.5870,0.2963,0.7847,0.1890]';
% (1.2202 + 1.2233 + 1.2044) * (1-P_fa)
channel_free_prob = sort(channel_free_prob,'descend');
N = length(channel_free_prob);
% P_d = 0.9;  % 检测概率

P_d = 0.8;  % 检测概率
P_fa = 0.3;  % 虚警概率：主用户未工作，检测判断为工作
Montecalo_num = 30 ;
U = 3;
M = 4; % 表示能感知的信道数量
K = 1; % 表示接入信道数量
num = 1;
 
Regret_log = zeros(1,length(sample_num_iter));
collision_Num = 0;

for num = 1 : length(sample_num_iter)
sample_num = sample_num_iter(num);
    
Reward = zeros(1,sample_num); 

Reward_1 = zeros(1,sample_num); 
Reward_2 = zeros(1,sample_num); 
Reward_3 = zeros(1,sample_num); 
%% Simulation of algorithm_1_2
for n = 1:Montecalo_num
    uniformRV = rand(N,sample_num);            % 信道实际空闲概率
    channel_state = zeros(N,sample_num);       % 信道状态： ‘1’时为空闲，‘0’时为busy
    uniformRV_sense = [];
    observation = [];
    for u = 1:U
        uniformRV_sense(:,:,u) =  rand(N,sample_num); % 用户观测空闲概率
    end 
    for u = 1:U
        observation(:,:,u) =  zeros(N,sample_num);    % 用户观察得到的信道状态  
    end
    
    for u = 1:U
        for j = 1:sample_num
            channel_state(:,j,u) = (uniformRV(:,j)<channel_free_prob);   % 信道实际状态
            observation(:,j,u) = channel_state(:,j,u) .* ( uniformRV_sense(:,j,u) < 1-P_fa )...
                                 + (1 - channel_state(:,j,u)) .* ( uniformRV_sense(:,j,u) < 1-P_d );
        end
    end
    
    % User 1 
    estimated_prob_user_1 = zeros(1,N); 
%     estimated_prob_user_1 = channel_free_prob;
    indexes_user_1 = zeros(1,N);
    T_user_1 = ones(1,N);
    Y_user_1 = observation(:,1,1)';   
    
    % User 2
    estimated_prob_user_2 = zeros(1,N); 
%     estimated_prob_user_2 = channel_free_prob;
    indexes_user_2 = zeros(1,N);
    T_user_2 = ones(1,N);
    Y_user_2 = observation(:,1,2)';      
    
    % User 3
    estimated_prob_user_3 = zeros(1,N); 
%     estimated_prob_user_3 = channel_free_prob;
    indexes_user_3 = zeros(1,N);
    T_user_3 = ones(1,N);
    Y_user_3 = observation(:,1,3)';     
    
    u_1_flag = 0;
    u_2_flag = 0;
    u_3_flag = 0;

    u_1 = 1;
    u_2 = 1;
    u_3 = 1;
    
    for t = 2:sample_num
        for i = 1:N
            estimated_prob_user_1(i) = (Y_user_1(i)/T_user_1(i) + (P_d-1) )/(P_d-P_fa);
            indexes_user_1(i) = estimated_prob_user_1(i) + sqrt( 2*log(t-1)/T_user_1(i) ) /(P_d-P_fa);
        end

        for i = 1:N
            estimated_prob_user_2(i) = (Y_user_2(i)/T_user_2(i) + (P_d-1) )/(P_d-P_fa);
            indexes_user_2(i) = estimated_prob_user_2(i) + sqrt( 2*log(t-1)/T_user_2(i) ) /(P_d-P_fa);
        end
        
        for i = 1:N
            estimated_prob_user_3(i) = (Y_user_3(i)/T_user_3(i) + (P_d-1) )/(P_d-P_fa);
            indexes_user_3(i) = estimated_prob_user_3(i) + sqrt( 2*log(t-1)/T_user_3(i) ) /(P_d-P_fa);
        end     
        
        %% 用户将信道降序排序后得到信道集合 N_star，记录在 M_sensed_channels_u 中
        M_sensed_channels_1 = zeros(1,M);
        M_sensed_channels_2 = zeros(1,M);
        M_sensed_channels_3 = zeros(1,M);
        [value_1 sequence_1] = sort(indexes_user_1,'descend'); 
        [value_2 sequence_2] = sort(indexes_user_2,'descend'); 
        [value_3 sequence_3] = sort(indexes_user_3,'descend');       
%         M_sensed_channels_1 = sequence_1(1:M);
%         M_sensed_channels_2 = sequence_2(1:M);
%         M_sensed_channels_3 = sequence_3(1:M);    
 
        
        M_sensed_channels_1(1) = sequence_1(u_1);
        for i = 2:M
            M_sensed_channels_1(i) = sequence_1(M*(U-u_1)+(u_1+i-1));
        end
       
        M_sensed_channels_2(1) = sequence_2(u_2);
        for i = 2:M
            M_sensed_channels_2(i) = sequence_2(M*(U-u_2)+(u_2+i-1));
        end
        
        M_sensed_channels_3(1) = sequence_3(u_3);
        for i = 2:M
            M_sensed_channels_3(i) = sequence_3(M*(U-u_3)+(u_3+i-1));
        end

        %% 更新 Y 和 T
        for i = 1:M
            T_user_1(M_sensed_channels_1(i)) = T_user_1(M_sensed_channels_1(i)) + 1;
            if observation(M_sensed_channels_1(i),t-1,1) > 0
                Y_user_1(M_sensed_channels_1(i)) = Y_user_1(M_sensed_channels_1(i)) + 1;
            end
        end   
        
        for i = 1:M
            T_user_2(M_sensed_channels_2(i)) = T_user_2(M_sensed_channels_2(i)) + 1;
            if observation(M_sensed_channels_2(i),t-1,1) > 0
                Y_user_2(M_sensed_channels_2(i)) = Y_user_2(M_sensed_channels_2(i)) + 1;
            end
        end 
        
        for i = 1:M
            T_user_3(M_sensed_channels_3(i)) = T_user_3(M_sensed_channels_3(i)) + 1;
            if observation(M_sensed_channels_3(i),t-1,1) > 0
                Y_user_3(M_sensed_channels_3(i)) = Y_user_3(M_sensed_channels_3(i)) + 1;
            end
        end
        
        %% 识别空闲信道
        cnt_K_access_1 = 1;
        cnt_K_access_2 = 1;
        cnt_K_access_3 = 1;
        flag_1 = 0;
        flag_2 = 0;
        flag_3 = 0;
%         channel_chosen_access_1 = zeros(1,K);
%         channel_chosen_access_2 = zeros(1,K);
%         channel_chosen_access_3 = zeros(1,K);
        channel_chosen_access_1 = [];
        channel_chosen_access_2 = [];
        channel_chosen_access_3 = [];        
        
        temp_indexes_1 = zeros(1,M);
        temp_indexes_2 = zeros(1,M);
        temp_indexes_3 = zeros(1,M);
        for i = 1:M
            temp_indexes_1(i) = (Y_user_1(M_sensed_channels_1(i))/ T_user_1(M_sensed_channels_1(i))+ P_d-1)/(P_d - P_fa) + sqrt(2*log(t-1)/T_user_1(M_sensed_channels_1(i)) )/(P_d-P_fa);
        end  
        for i = 1:M
            temp_indexes_2(i) = (Y_user_2(M_sensed_channels_2(i))/ T_user_2(M_sensed_channels_2(i))+ P_d-1)/(P_d - P_fa) + sqrt(2*log(t-1)/T_user_2(M_sensed_channels_2(i)) )/(P_d-P_fa);
        end  
        for i = 1:M
            temp_indexes_3(i) = (Y_user_3(M_sensed_channels_3(i))/ T_user_3(M_sensed_channels_3(i))+ P_d-1)/(P_d - P_fa) + sqrt(2*log(t-1)/T_user_3(M_sensed_channels_3(i)) )/(P_d-P_fa);
        end         
        
        %% for user_1
         for i = 1:M
            if observation(M_sensed_channels_1(i),t,1) < 1
                temp_indexes_1(i) = -1000;
            end
        end
        
        while (cnt_K_access_1 < K+1 && flag_1 < 1)
            flag_1 = 1;
            for j = 1:M
                if temp_indexes_1(j) > -1000
                    flag_1 = 0;
                    continue;
                end
            end
            
            for j = 1:M
                if (temp_indexes_1(j) == max(temp_indexes_1) && temp_indexes_1(j) > -1000)
                    channel_chosen_access_1(cnt_K_access_1) = M_sensed_channels_1(j);
                    cnt_K_access_1 = cnt_K_access_1 + 1;
                    temp_indexes_1(j) = -1000;
                end
            end     
        end     
        
        %% for user_2
        for i = 1:M
            if observation(M_sensed_channels_2(i),t,2) < 1
                temp_indexes_2(i) = -1000;
            end
        end
        
        while (cnt_K_access_2 < K+1 && flag_2 < 1)
            flag_2 = 1;
            for j = 1:M
                if temp_indexes_2(j) > -1000
                    flag_2 = 0;
                    continue;
                end
            end
            
            for j = 1:M
                if (temp_indexes_2(j) == max(temp_indexes_2) && temp_indexes_2(j) > -1000)
                    channel_chosen_access_2(cnt_K_access_2) = M_sensed_channels_2(j);
                    cnt_K_access_2 = cnt_K_access_2 + 1;
                    temp_indexes_2(j) = -1000;
                end
            end     
        end
        
        %% for user_3
        for i = 1:M
            if observation(M_sensed_channels_3(i),t,3) < 1
                temp_indexes_3(i) = -1000;
            end
        end
        
        while (cnt_K_access_3 < K+1 && flag_3 < 1)
            flag_3 = 1;
            for j = 1:M
                if temp_indexes_3(j) > -1000
                    flag_3 = 0;
                    continue;
                end
            end
            
            for j = 1:M
                if (temp_indexes_3(j) == max(temp_indexes_3) && temp_indexes_3(j) > -1000)
                    channel_chosen_access_3(cnt_K_access_3) = M_sensed_channels_3(j);
                    cnt_K_access_3 = cnt_K_access_3 + 1;
                    temp_indexes_3(j) = -1000;
                end
            end     
        end        
        
        %% 检测是否有冲突

        check_1_2 = [];
        check_1_3 = [];
        check_2_3 = [];
        check_1_2 = intersect(channel_chosen_access_1,channel_chosen_access_2);
        if length(check_1_2) > 0
            u_1 = randi(3);
            u_2 = randi(3);
            collision_Num = collision_Num + 1;
        end

        check_1_3 = intersect(channel_chosen_access_1,channel_chosen_access_3);
        if length(check_1_3) > 0
            u_1= randi(3);
            u_3 = randi(3);
            collision_Num = collision_Num + 1;
        end
        
        check_2_3 = intersect(channel_chosen_access_2,channel_chosen_access_3);
        if length(check_2_3) > 0
            u_2 = randi(3);
            u_3 = randi(3);
            collision_Num = collision_Num + 1;
        end
                 
%         判断是否有冲突            
        for i = 1:length(channel_chosen_access_1)
            for j = 1:length(channel_chosen_access_2)
                for k = 1:length(channel_chosen_access_3)
                    if channel_chosen_access_1(i) == channel_chosen_access_2(j)
                        if channel_chosen_access_1(i) == channel_chosen_access_3(k) 
                            channel_chosen_access_1(i) = 0;
                            channel_chosen_access_2(j) = 0;
                            channel_chosen_access_3(k) = 0;
                        else
                            channel_chosen_access_1(i) = 0;
                            channel_chosen_access_2(j) = 0;
                        end
                    end
                    if channel_chosen_access_1(i) == channel_chosen_access_3(k)
                        if channel_chosen_access_1(i) == channel_chosen_access_2(j)
                            channel_chosen_access_1(i) = 0;
                            channel_chosen_access_2(j) = 0;
                            channel_chosen_access_3(k) = 0;
                        else
                            channel_chosen_access_1(i) = 0;
                            channel_chosen_access_3(k) = 0;
                        end
                    end
                    if channel_chosen_access_2(j) == channel_chosen_access_3(k)
                        if channel_chosen_access_2(j) == channel_chosen_access_1(i)
                            channel_chosen_access_1(i) = 0;
                            channel_chosen_access_2(j) = 0;
                            channel_chosen_access_3(k) = 0;
                        else
                            channel_chosen_access_2(j) = 0;
                            channel_chosen_access_3(k) = 0;
                        end
                    end
                end
            end
        end
       channel_chosen_access_1(find(channel_chosen_access_1 == 0)) = [];
       channel_chosen_access_2(find(channel_chosen_access_2 == 0)) = [];
       channel_chosen_access_3(find(channel_chosen_access_3 == 0)) = [];
        
        %% 求 Reward
        % 求 Reward_1
        if channel_chosen_access_1 >0
            for i = 1:min(length(channel_chosen_access_1),K)
                if channel_chosen_access_1(i) > 0
                    Reward_1(t) = Reward_1(t) + channel_state(channel_chosen_access_1(i),t);
                end
            end
        end    
        % 求 Reward_2
        if channel_chosen_access_2 > 0
            for i = 1:min(length(channel_chosen_access_2),K)
                if channel_chosen_access_2(i) > 0
                    Reward_2(t) = Reward_2(t) + channel_state(channel_chosen_access_2(i),t);
                end
            end  
        end
        % 求 Reward_3
        if channel_chosen_access_3 > 0
            for i = 1:min(length(channel_chosen_access_3),K)
                if channel_chosen_access_3(i) > 0
                   Reward_3(t) = Reward_3(t) + channel_state(channel_chosen_access_3(i),t);
                end
             end  
        end 
        
    %% 显示出来
        if mod(t, 1000)==0
            fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d, check_1_2 = %d, check_1_3 = %d, check_2_3 = %d\n ',n , t , u_1, u_2, u_3, check_1_2,check_1_3,check_2_3  );  
        end 
%     fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d\n ',n,t,u_1,u_2,u_3 );  
%     fprintf('check_1_2 = %d, check_1_3 = %d, check_2_3 = %d\n ',check_1_2,check_1_3,check_2_3 ); 
%     fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d, check_1_2 = %d, check_1_3 = %d, check_2_3 = %d\n ',n , t , u_1, u_2, u_3, check_1_2,check_1_3,check_2_3  );     
%     fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d \n ',n , t , u_1, u_2, u_3 );        
    end
end

% Reward_expect_all = (1-P_fa) * sum(channel_free_prob(1:6)); 
if P_d == 0.9
    Reward_expect_all = 0.9119 + 0.8662 + 0.9342 ; % P_d = 0.9
elseif P_d == 0.8
    Reward_expect_all = 0.9144 + 0.8961 + 0.8567;
end
   
% Reward_expect_all = 0.9144 + 0.8961 + 0.8567;



Reward_average_1 = Reward_1/Montecalo_num;
Reward_average_2 = Reward_2/Montecalo_num;
Reward_average_3 = Reward_3/Montecalo_num;  
Reward_ALL = sum(Reward_average_1 + Reward_average_2 + Reward_average_3);
Regret_log(num) = (Reward_expect_all * sample_num - Reward_ALL)/log(sample_num);
end

Reward_average_1 = Reward_1/Montecalo_num;
Reward_average_2 = Reward_2/Montecalo_num;
Reward_average_3 = Reward_3/Montecalo_num;  
Reward_average_as_time_1 = zeros(1,sample_num); 
Reward_average_as_time_2 = zeros(1,sample_num); 
Reward_average_as_time_3 = zeros(1,sample_num);     
temp_for_Reward_1 = 0;
temp_for_Reward_2 = 0;
temp_for_Reward_3 = 0;
for t1 = 1:sample_num
    temp_for_Reward_1 = temp_for_Reward_1 + Reward_average_1(t1);
    Reward_average_as_time_1(t1) =  temp_for_Reward_1/t1;
end

for t1 = 1:sample_num
    temp_for_Reward_2 = temp_for_Reward_2 + Reward_average_2(t1);
    Reward_average_as_time_2(t1) =  temp_for_Reward_2/t1;
end

for t1 = 1:sample_num
    temp_for_Reward_3 = temp_for_Reward_3 + Reward_average_3(t1);
    Reward_average_as_time_3(t1) =  temp_for_Reward_3/t1;
end


 %% add new experiments

save_result1=zeros(1,sample_num);
save_result2=zeros(1,sample_num);
temp4=0;
prob_avergae_result = ( Reward_1 + Reward_2 + Reward_3 )/Montecalo_num;

for t1=2:sample_num
   temp4=temp4+prob_avergae_result(t1);
   save_result1(t1)=temp4/t1;  
   save_result2(t1)=(Reward_expect_all*t1-temp4)/log(t1);
end

figure;
plot(1:sample_num,save_result1,'b');
hold on;
plot(1:sample_num,Reward_expect_all*ones(1,sample_num),'r');
hold off;
figure;
plot(1:sample_num,save_result2,'b');
 

%% new 

% Num_record = [ round(0.1*sample_num), round(0.2*sample_num),round(0.3*sample_num),round(0.4*sample_num),round(0.5*sample_num),round(0.6*sample_num),round(0.7*sample_num),round(0.8*sample_num),round(0.9*sample_num),round(1*sample_num) ]
% Reward_record = [];
% Regret_record = [];
% 
% for i = 1:length(Num_record)
%     Reward_record(i) = save_result1(Num_record(i));
%     Regret_record(i) = save_result2(Num_record(i));
% end   
% 
% figure;
% plot(Num_record,Reward_record,'b');
% % hold on;
% % plot(Num_record,comparison_value*ones(1,n_slot),'r');
% % hold off;
% figure;
% plot(Num_record,Regret_record,'b') ;


% save 'N_20_U_3_M_4_K_1_regret.mat' save_result2
% save 'N_20_U_3_M_4_K_1_reward.mat' save_result1
% save 'N_20_U_3_M_4_K_1_reward_optimal.mat' Reward_expect_all



collision_Num = collision_Num/Montecalo_num







