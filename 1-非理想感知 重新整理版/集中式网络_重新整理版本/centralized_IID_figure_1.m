

%--------------------------------------------------------------------------------------------------------------------------------------------------%
% Description : This is used to simulate the centralized algorithm in Chap 3.2
%               
%               Author: xxx
%               Date  : 2019/03/22
%               Modified : 2023/06/01
%
% ����ʽ�����д������Ľڵ���ȸ���Ҫ�û����и�֪���룬����ʱ��ת��Ϊ������Ҫ�û���֪��������ŵ�
% ����˵����
    %% 1�����ڲ��������е��� U��M_u��K_u �õ���ͬ�����µķ�������Ϊ���ڶ�����⣬�� U=2��M_u=3��K_u=2 ���������µķ��������ϸ����

    %% 2���������� P_d=0.7��0.8; P_f=4.1��0.2��0.3 ��ò�ͬ�ķ�����

    %% 3����ͬ�ŵ������µķ�����
    % N = 15  
    % channel_free_prob = [ 0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 , 0.5580, 0.4067, 0.8799, 0.7994 ]';
    % N = 14 
    % channel_free_prob = [  0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904, 0.5580, 0.4067 ,0.8799  ]';
    % N = 13 
    % channel_free_prob = [ 0.0499, 0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017 , 0.1354, 0.9707,0.4904 , 0.5580, 0.4067  ]';

    %% 4�� ��ͬʱ϶�¶��ŵ����и��ʵĹ��ƣ��ɵ��� sample_num = 100��500��1000��2000 �ֱ�õ�
    
    %% 5�� ��ͬʱ϶�¶��ŵ�ѡȡ�Ĵ��������㷨�����б��� T ��ʾÿ���ŵ���֪�������� Y ��ʾ�ŵ��������������¼��ɻ�ͼ�õ�������
    
%--------------------------------------------------------------------------------------------------------------------------------------------------%
 
clc
clear
tic

%% Parameter Setting
sample_num = 100000;  % ��������
channel_free_prob = [  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1  ]'; % �ŵ����и���
channel_free_prob = sort(channel_free_prob,'descend');
N = length(channel_free_prob); % �ŵ�����
P_d = 0.8;  % ������
P_fa = 0.3;  % �龯���ʣ����û�δ����������ж�Ϊ����
Montecalo_num = 65; % ���ؿ���������
theta_difference=zeros(1,sample_num); % ���趨�� theta �Ƚϵ��ۻ���
U = 2; % ��ʾ����ʽ�����еĴ�Ҫ�û�����
M_u = 3; % ÿ����Ҫ�û��ܹ���֪���ŵ�����
K_u = 1; % ÿ����Ҫ�û��ܹ�������ŵ�����
M = U * M_u; % ��ʾ�ܸ�֪���ŵ����� 
K = U * K_u; % ��ʾ�����ŵ����� 
Reward = zeros(1,sample_num); 
 
%% Simulation
for n = 1:Montecalo_num
    uniformRV= rand(N,sample_num);            % �ŵ�ʵ�ʿ��и���
    uniformRV_sense = rand(N,sample_num);     % �û��۲���и���
    channel_state = zeros(N,sample_num);      % �ŵ�״̬�� ��1��ʱΪ���У���0��ʱΪbusy
    observation = zeros(N,sample_num);        % �û��۲�õ����ŵ�״̬
    %% ��ʼ��
    for j = 1:sample_num
        channel_state(:,j) = (uniformRV(:,j)<channel_free_prob);   % �ŵ�ʵ��״̬
        observation(:,j) = channel_state(:,j) .* ( uniformRV_sense(:,j) < 1-P_fa ) ...
                           + (1 - channel_state(:,j)) .* ( uniformRV_sense(:,j) < 1-P_d);    % �۲�õ����ŵ�״̬
    end
    estimated_prob = zeros(1,N);      % �����ŵ����и���
    indexes = zeros(1,N);             % �ŵ����
    T = ones(1,N);                    % ÿ���ŵ����۲���һ��
    Y = observation(:,1)';

    %% ��ʼ   
    for t = 2:sample_num
       for i = 1:N
           estimated_prob(i) = (Y(i)/T(i) + (P_d-1))/(P_d-P_fa);
           indexes(i) = estimated_prob(i) + sqrt( 2*log(t-1)/T(i) ) /(P_d-P_fa); 
       end

       %% ��֪ M ���ŵ� ��¼�� M_sensed_channels
       M_sensed_channels = zeros(1,M);
       temp = indexes;
       [value sequence] = sort(temp,'descend');  % sequence �е�ǰM��Ϊ����ǰM���ŵ�
       M_sensed_channels = sequence(1:M);
       %% ��֪ǰ M ���ŵ������� T �� Y
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
               temp_indexes(i) = -1000;               % �õ� M ���ŵ� '\theta+sqrt'������û��۲�Ϊ0�����ŵ�Ϊ������
           end
       end
       %% ������� K ���ŵ�

        cnt_K_access = 1;
        flag = 0;
        while (cnt_K_access < K+1 && flag < 1)
            flag = 1;
            for j = 1:M
                if temp_indexes(j)>-1000  % ���޿����ŵ���������ôν������
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
                Reward(t) = Reward(t) + channel_state(channel_chosen_access(i),t);  % �����ŵ�������� 
            end
        end
        
        if mod(t, 10000)==0
             fprintf('sample_num = %d,n = %d,t = %d\n',sample_num,n,t );
        end   
    end
end

%% Benchmark: ���ŵ����и�����֪����µ����Ÿ�֪���뷽������$\mathbb{E}\{ U^{\pi^{*C}}(t) \} $
% ��Ҫע����ǣ�ÿʱ϶��ÿ���ŵ��Ŀ���--ռ��״̬Ϊ�����������ѡ�� M ���ŵ�ʱ��ÿʱ϶����
% 2^M ���ŵ�״̬����Ϸ�ʽ��ÿ���ŵ���Ϸ�ʽ��ÿʱ϶����һ�����ʳ��֣������Ի�õ�������������

f_theta = zeros(1,N); % ���ÿ���ŵ��Ĺ۲���и���
f_theta_sort = zeros(1,N); % ÿ���ŵ��Ĺ۲���и��� ��������
f_theta_sort_sequence = zeros(1,N); % ÿ���ŵ��Ĺ۲���и��� �����������
f_theta_M_best = zeros(1,M); % ���ŵ� M ���ŵ�
f_theta_M_best_sequence = zeros(1,M); % ���ŵ� M ���ŵ����
Reward_expect_all_status = 0; % ���ŵ� M ���ŵ��½����ŵ���throughout
for i = 1:N
    f_theta(i) = (1-P_fa) * channel_free_prob(i) + (1-P_d) * (1-channel_free_prob(i));
end

[f_theta_sort f_theta_sort_sequence] = sort(f_theta,'descend');  % ���ŵ��۲���и��ʽ�������
f_theta_M_best = f_theta_sort(1:M); % �ҳ� M �������ŵ�
f_theta_M_best_sequence = f_theta_sort_sequence(1:M); % ��¼ M �������ŵ����
 

for channel_combination_status = 0:2^M-1  % channel_combination_status ��ʾ M ���ŵ���ͬ����Ϸ�ʽ
    channel_combination_status_dec2bin = dec2bin(channel_combination_status); % ��channel_combination_status ת��Ϊ�����ƣ�����ÿ���ŵ�״̬
    while length(channel_combination_status_dec2bin) < M
        channel_combination_status_dec2bin = strcat('0',channel_combination_status_dec2bin);
    end

    % channel_combination_status_dec2bin ��ʱ�ñ����Ѿ�Ϊ ���� Ϊ M ��char�����ַ����������ŵ�״̬
    channel_combination_status_dec2bin_char2double = double(uint8(channel_combination_status_dec2bin)>48);  % ��char����ת��Ϊ double ����
    % Sensing_result ��ʾÿһ���ŵ���ϴ��ڵĸ���
    Sensing_result = 1;

    % �� M* ��ÿ���ŵ����״̬�µĴ�����ʡ���Ϊ��������
    for j = 1:M
        if channel_combination_status_dec2bin_char2double(j)>0
            Sensing_result = Sensing_result * f_theta_M_best(j);
        else
            Sensing_result = Sensing_result * (1-f_theta_M_best(j));
        end
    end

    cnt_channel_access = 0; % ��ʾ�����ŵ����� ���� K ��
    Reward_single_status = 0;
    for k = 1:M
        if cnt_channel_access < K
            if channel_combination_status_dec2bin_char2double(k) > 0 
                Reward_single_status = Reward_single_status + channel_free_prob(f_theta_M_best_sequence(k))/f_theta_M_best(k);  % �ŵ� k ������reward�� \hat{\theta_i} / f(\hat{\theta})
                cnt_channel_access = cnt_channel_access + 1;
            end
        end
    end
    Reward_expect_all_status = Reward_expect_all_status + Reward_single_status * Sensing_result;
end
Reward_expect_all_status = Reward_expect_all_status * (1-P_fa);  % ���յõ���reward



%% ����õ�������
Reward_ALL = 0;
Regret = 0;
Regret_as_time = zeros(1,sample_num); % regret/ln t ����ʽ
Regret_as_time_no_normalized = zeros(1,sample_num); % ���� regret ����ʽ ����������Ҫ���������scale����������ʧ��������ͬ 

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
 
% ƽ��������
figure;
plot(1:sample_num,Reward_average_as_time,'b');
hold on;
plot(1:sample_num,Reward_expect_all_status*ones(1,sample_num),'r');
hold off;

% ��һ������������ʧ����regret/log t ��ʽ������ʽ��δ��һ������ʽ������ͬ���������������ꡣ
figure;
plot(1:sample_num,Regret_as_time,'b');
hold off;

% δ��һ������������ʧ�������� regret ����ʽ��ʾ��������
figure;
plot(1:sample_num,Regret_as_time_no_normalized,'b');
hold off;

%% ������õ������ݴ洢������ʹ��python�ȹ��߽���ͳһ��ͼ

% save 'U_4_M_1_K_1_regret.mat' Regret_as_time
% save 'U_4_M_1_K_1_reward.mat' Reward_average_as_time
% save 'U_4_M_1_K_1_reward_optimal.mat' Reward_expect_all_status

 
toc  
      
       
 
                   
           
           
 
               
       



