%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            Algorithm 1.3
%    ���û��ֲ������㷨 Algorithm 1.3���÷���ǰ������ŵ����� N �����û�����
%                         NOTICE: N must > U * M
%
%                   Author: xxx
%                   Date  : 2019/04/18
%                       ����дһ����㷨
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sample_num = 20000;  % ��������
% channel_free_prob = [0.5296,0.4001,0.9817,0.1931,0.2495]';  % �ŵ����и���
channel_free_prob = [0.6923,0.5430,0.3544,0.8753,0.5212,0.6759,0.8783,0.9762]';   
% channel_free_prob = [0.8811, 0.5390,0.3468,0.9522,0.7823,0.0471,0.7968]';    % (1.0645 + 1.0748 + 1.1110)* (1-P_fa)
% channel_free_prob = [0.8811, 0.5390,0.3468,0.9522,0.7323,0.0471,0.7968]';
% channel_free_prob = [0.9 0.8 0.7 0.6 0.5 0.4 0.3]';  % 0.7280 + 0.700 + 0.6790 = 2.1070
channel_free_prob = sort(channel_free_prob,'descend');
N = length(channel_free_prob);
P_d = 0.8;  % ������
P_fa = 0.3;  % �龯���ʣ����û�δ����������ж�Ϊ����
Montecalo_num = 30;
U = 3;
M = 2; % ��ʾ�ܸ�֪���ŵ�����
K = 2; % ��ʾ�����ŵ�����
Reward = zeros(1,sample_num); 

Reward_1 = zeros(1,sample_num); 
Reward_2 = zeros(1,sample_num); 
Reward_3 = zeros(1,sample_num); 
%% Simulation of algorithm_1_2
for n = 1:Montecalo_num
    uniformRV = rand(N,sample_num);            % �ŵ�ʵ�ʿ��и���
    channel_state = zeros(N,sample_num);       % �ŵ�״̬�� ��1��ʱΪ���У���0��ʱΪbusy
    uniformRV_sense = [];
    observation = [];
    for u = 1:U
        uniformRV_sense(:,:,u) =  rand(N,sample_num); % �û��۲���и���
    end 
    for u = 1:U
        observation(:,:,u) =  zeros(N,sample_num);    % �û��۲�õ����ŵ�״̬  
    end
    
    for u = 1:U
        for j = 1:sample_num
            channel_state(:,j,u) = (uniformRV(:,j)<channel_free_prob);   % �ŵ�ʵ��״̬
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
        
        %% �û����ŵ����������õ��ŵ����� N_star����¼�� M_sensed_channels_u ��
        M_sensed_channels_1 = zeros(1,M);
        M_sensed_channels_2 = zeros(1,M);
        M_sensed_channels_3 = zeros(1,M);
        [value_1 sequence_1] = sort(indexes_user_1,'descend'); 
        [value_2 sequence_2] = sort(indexes_user_2,'descend'); 
        [value_3 sequence_3] = sort(indexes_user_3,'descend');       
%         M_sensed_channels_1 = sequence_1(1:M);
%         M_sensed_channels_2 = sequence_2(1:M);
%         M_sensed_channels_3 = sequence_3(1:M);    
        
        %% �����֪�Ƿ���У�
%         u = randi(3);
%         u_1 = mod(u-1,3) + 1;
%         M_sensed_channels_1(1) = sequence_1(u_1);
%         for i = 2:M
%             M_sensed_channels_1(i) = sequence_1(M*(U-u_1)+(u_1+i-1));
%         end
%         
%         u_2 = mod(u,3) + 1;
%         M_sensed_channels_2(1) = sequence_2(u_2);
%         for i = 2:M
%             M_sensed_channels_2(i) = sequence_2(M*(U-u_2)+(u_2+i-1));
%         end
% 
%         u_3 = mod(u+1,3) + 1;
%         M_sensed_channels_3(1) = sequence_3(u_3);
%         for i = 2:M
%             M_sensed_channels_3(i) = sequence_3(M*(U-u_3)+(u_3+i-1));
%         end
        
        %% algorithm 1.4
%         u_1 = 1;
%         u_2 = 1;
%         u_3 = 1;
        
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

        %% ���� Y �� T
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
        
        %% ʶ������ŵ�
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
        
        %% ����Ƿ��г�ͻ

        check_1_2 = [];
        check_1_3 = [];
        check_2_3 = [];
        check_1_2 = intersect(channel_chosen_access_1,channel_chosen_access_2);
        if length(check_1_2) > 0
            u_1 = randi(3);
            u_2 = randi(3);
        end

        check_1_3 = intersect(channel_chosen_access_1,channel_chosen_access_3);
        if length(check_1_3) > 0
            u_1= randi(3);
            u_3 = randi(3);
        end
        
        check_2_3 = intersect(channel_chosen_access_2,channel_chosen_access_3);
        if length(check_2_3) > 0
            u_2 = randi(3);
            u_3 = randi(3);
        end
                 
%         �ж��Ƿ��г�ͻ            
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
        
        %% �� Reward
        % �� Reward_1
        if channel_chosen_access_1 >0
            for i = 1:length(channel_chosen_access_1)
               if channel_chosen_access_1(i) > 0
                   Reward_1(t) = Reward_1(t) + channel_state(channel_chosen_access_1(i),t);
               end
            end
        end    
        % �� Reward_2
        if channel_chosen_access_2 > 0
            for i = 1:length(channel_chosen_access_2)
                if channel_chosen_access_2(i) > 0
                    Reward_2(t) = Reward_2(t) + channel_state(channel_chosen_access_2(i),t);
                end
            end  
        end
        % �� Reward_3
        if channel_chosen_access_3 > 0
            for i = 1:length(channel_chosen_access_3)
                if channel_chosen_access_3(i) > 0
                   Reward_3(t) = Reward_3(t) + channel_state(channel_chosen_access_3(i),t);
                end
             end  
        end 
        
    %% ��ʾ����
        if mod(t, 1000)==0
            fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d, check_1_2 = %d, check_1_3 = %d, check_2_3 = %d\n ',n , t , u_1, u_2, u_3, check_1_2,check_1_3,check_2_3  );  
        end 
%     fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d\n ',n,t,u_1,u_2,u_3 );  
%     fprintf('check_1_2 = %d, check_1_3 = %d, check_2_3 = %d\n ',check_1_2,check_1_3,check_2_3 ); 
%     fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d, check_1_2 = %d, check_1_3 = %d, check_2_3 = %d\n ',n , t , u_1, u_2, u_3, check_1_2,check_1_3,check_2_3  );     
%     fprintf('n = %d, t = %d , u_1 = %d, u_2 = %d, u_3 = %d \n ',n , t , u_1, u_2, u_3 );        
    end
end

% Reward_expect_all = 0.8019 + 0.7855 + 0.7883;
% Reward_expect_all = 0.7452 + 0.7524 + 0.7777;
% Reward_expect_all = 0.7451 + 0.7390 + 0.7636; 
% Reward_expect_all = sum(channel_free_prob(1:3)) * (1-P_fa);
% Reward_expect_all = (1.0645 + 1.0748 + 1.1110)* (1-P_fa);
% Reward_expect_all = (1.1456 + 1.1202 + 1.0627 )*(1-P_fa);

Reward_expect_all = (1-P_fa) * sum(channel_free_prob(1:6));
% Reward_expect_all = 0.7280 + 0.700 + 0.6790;
Reward_ALL = 0;
Regret = 0;
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

Reward_ALL = sum(Reward_average_1 + Reward_average_2 + Reward_average_3)            
Reward_expect_all * sample_num

Reward_expect_all * sample_num - Reward_ALL

(Reward_expect_all * sample_num - Reward_ALL)/log(sample_num);

% 529 882 1352 1459 1632 1928 2476 2773 

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

Num_record = [ round(0.1*sample_num), round(0.2*sample_num),round(0.3*sample_num),round(0.4*sample_num),round(0.5*sample_num),round(0.6*sample_num),round(0.7*sample_num),round(0.8*sample_num),round(0.9*sample_num),round(1*sample_num) ]
Reward_record = [];
Regret_record = [];

for i = 1:length(Num_record)
    Reward_record(i) = save_result1(Num_record(i));
    Regret_record(i) = save_result2(Num_record(i));
end   

figure;
plot(Num_record,Reward_record,'b');
% hold on;
% plot(Num_record,comparison_value*ones(1,n_slot),'r');
% hold off;
figure;
plot(Num_record,Regret_record,'b') ;





save 'N_8_U_3_M_2_K_2_regret.mat' save_result2
save 'N_8_U_3_M_2_K_2_reward.mat' save_result1
save 'N_8_U_3_M_2_K_2_reward_optimal.mat' Reward_expect_all









% fprintf('sample_num = %d \nReward_average_as_time_1= %f\nReward_average_as_time_2=%f\nReward_average_as_time_3=%.6f\n',...
%              sample_num,Reward_average_as_time_1(end),Reward_average_as_time_2(end),Reward_average_as_time_3(end) );
% fprintf('Reward_Experiment = %f\nReward_expect_all = %f \n',Reward_average_as_time_1(end)+Reward_average_as_time_2(end)+Reward_average_as_time_3(end), Reward_expect_all  );
% 
% fprintf('Regret/log = %f \n',(Reward_expect_all * sample_num - Reward_ALL)/log(sample_num) );
% 
% % sample_num_Zhou_Zhang = [ 80 100 200 400 600 800 1000 2000 4000 6000 8000 10000 12000 15000 18000 22000 26000 29000 33000 36000 40000 45000 50000 55000];
% % Regret_log_Zhou_Zhang = [ 11.508222  13.281087  22.546152  36.020130  49.159521 57.788447 67.0935 106.32 156.252581  183.362718 207.69 226.9 238.692289  256.555427  260.443569 270.272494 274.959743 279.867221 289.092596 289.942837 298.52 303.559 305.995446 306.741379 ];
% 
% % sample_num_Zhou_Zhang = [ 20 40 60 80 100 200 400 600 800 1000 2000 5000 10000 20000 30000 40000  45000 50000 60000 70000 80000 ] ;
% % Regret_log_Zhou_Zhang = [ 8.419978 12.103404 16.369409  20.069709  22.164219 37.745946  62.65246 83.816773 105.923878 124.25 198.792184  333.372736 464.249944 580.411695 618.108377 659.704725   699.963373    748.313 804.3 819.352343 820.520721 ];
% sample_num_Zhou_Zhang = [ 20 40 60 80 100 200 400 600 800 1000 2000 5000 10000 20000 30000 40000 ] ;
% Regret_log_Zhou_Zhang = [ 8.419978 12.103404 16.369409  20.069709  22.164219 37.745946  62.65246 83.816773 105.923878 124.25 198.792184  333.372736 464.249944 580.411695 618.108377 659.704725];
% 
% figure
% plot(sample_num_Zhou_Zhang,Regret_log_Zhou_Zhang,'bo-','LineWidth',2)
% title('Zhou Zhang Algorithm with P_d=0.8 and P_fa=0.3')
% legend('N=8,U=3,M=2,K=1')
% xlabel('time')
% ylabel('Regret/log(t)')

% sample_num_Zhou_Zhang = [ 400 600 800 1000 2000 5000 10000 20000 30000 40000 45000 48000 ];
% Regret_log_Zhou_Zhang = [ 62.65246 83.816773 105.923878 124.25 198.792184  333.372736 464.249944 580.411695 618.108377 659.704725  684.488897 715.6 ];





