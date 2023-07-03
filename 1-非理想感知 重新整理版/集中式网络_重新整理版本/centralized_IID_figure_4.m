

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description : This is used to simulate the algorithm_1_1
%               
%               Author: xxx
%               Date  : 2019/03/22
%               Modified : 2023/06/01
% �ڸ÷����� ���ò�ͬ�� sample_num����¼��Ӧ�� estimated_prob���õ���3.1
% ֵ��˵����ÿ�η���õ��Ľ����ͬ��ʵ����Ӧ�ý������ؿ������ȡͳ��ƽ��
% ���õ��������������ͬ��

% ��¼ T ���Եõ��Բ�ͬ�ŵ�ѡ��Ĵ���
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
sample_num = 1000;  % ��������
% channel_free_prob = [0.5296,0.4001,0.9817,0.1931,0.2495]';  % �ŵ����и���
 
channel_free_prob = [ 0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994 ]';

% N = 14 
% channel_free_prob = [  0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994 ]';


 % N = 13 
% channel_free_prob = [ 0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017 ,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994 ]';



% channel_free_prob = [0.8457,0.9411,0.6287,0.2052,0.5259,0.1450,0.4818,0.7789]';
% channel_free_prob = sort(channel_free_prob,'descend');

% channel_free_prob = [0.9817,0.5296,0.4001,0.2495,0.1931]';  
N = length(channel_free_prob);
P_d = 0.8;  % ������
P_fa = 0.3;  % �龯���ʣ����û�δ����������ж�Ϊ����
Montecalo_num = 1;
theta_difference=zeros(1,sample_num); % ���趨�� theta �Ƚϵ��ۻ���
M = 6; % ��ʾ�ܸ�֪���ŵ����� U=3,M=2,K=1
K = 3; % ��ʾ�����ŵ�����
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
       theta_difference(t) = theta_difference(t) + norm(estimated_prob'- channel_free_prob);
       
       %% Ѱ�� M ���ŵ�ȥsense ��¼�� M_sensed_channels
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
           temp_indexes(i) = (Y(M_sensed_channels(i))/T(M_sensed_channels(i)) + P_d-1)/(P_d - P_fa) + sqrt(2*log(t-1)/T(M_sensed_channels(i)) )/(P_d-P_fa);
%            temp_indexes(i) = (Y(M_sensed_channels(i))/T(M_sensed_channels(i)) + P_d-1) + sqrt(2*log(t-1)/T(M_sensed_channels(i)) );
       end
 
       
       for i = 1:M
           if observation(M_sensed_channels(i),t) < 1
               temp_indexes(i) = -1000;               % �õ� M ���ŵ� '\theta+sqrt'������û��۲�Ϊ0�����ŵ�Ϊ������
           end
       end
       %% �� M ���ŵ����� '\theta+sqrt' ��������ѡȡǰ K ���ŵ����д���
%        [M_sensed_channels_value M_sensed_channels_sequence] = sort(temp_indexes,'descend');
%        busy_index = find(M_sensed_channels_value == -1000); % �ҵ�busy���ŵ���ǰ������(busy_index-1)���ŵ�Ϊ���Դ�����Ϣ���ŵ�
%        % ���������ŵ����� K ���бȽϣ��ɽ����ŵ��������� K ʱ��ѡ���� K ���ŵ���������
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

%% Benchmark: ������� M*�� ѡ����õ� M ���ŵ����ҳ����е��ŵ���������<=K����

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

% f_theta_M_best(1) = f_theta_sort(1);
% f_theta_M_best(2) = f_theta_sort(6);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(1); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(6); 
% 
% f_theta_M_best(1) = f_theta_sort(2);
% f_theta_M_best(2) = f_theta_sort(5);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(2); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(5); 
% 
% f_theta_M_best(1) = f_theta_sort(3);
% f_theta_M_best(2) = f_theta_sort(4);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(3); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(4); 


for channel_combination_status = 0:2^M-1  % channel_combination_status ��ʾ M ���ŵ���ͬ����Ϸ�ʽ
    channel_combination_status_dec2bin = dec2bin(channel_combination_status); % ��channel_combination_status ת��Ϊ�����ƣ�����ÿ���ŵ�״̬
    while length(channel_combination_status_dec2bin) < M
        channel_combination_status_dec2bin = strcat('0',channel_combination_status_dec2bin);
    end
%     channel_combination_status_dec2bin
    % channel_combination_status_dec2bin ��ʱ�ñ����Ѿ�Ϊ ���� Ϊ M ��char�����ַ����������ŵ�״̬
    channel_combination_status_dec2bin_char2double = double(uint8(channel_combination_status_dec2bin)>48);  % ��char����ת��Ϊ double ����
    % Sensing_result ��ʾÿһ���ŵ���ϴ��ڵĸ���
    Sensing_result = 1;
%     channel_combination_status_dec2bin_char2double
    % �� M* ��ÿ���ŵ����״̬�µĴ�����ʡ���Ϊ��������
    for j = 1:M
        if channel_combination_status_dec2bin_char2double(j)>0
            Sensing_result = Sensing_result * f_theta_M_best(j);
        else
            Sensing_result = Sensing_result * (1-f_theta_M_best(j));
        end
    end
%     Sensing_result
    cnt_channel_access = 0; % ��ʾ�����ŵ����� ���� K ��
    Reward_single_status = 0;
    for k = 1:M
        if cnt_channel_access < K
            if channel_combination_status_dec2bin_char2double(k) > 0 
                Reward_single_status = Reward_single_status + channel_free_prob(f_theta_M_best_sequence(k))/f_theta_M_best(k);  % �ŵ� k ������reward�� \hat{\theta_i} / f(\hat{\theta})
%                 Reward_single_status = Reward_single_status + channel_free_prob(k)/f_theta(k) ; 
                cnt_channel_access = cnt_channel_access + 1;
            end
        end
    end
    Reward_expect_all_status = Reward_expect_all_status + Reward_single_status * Sensing_result;
end
Reward_expect_all_status = Reward_expect_all_status * (1-P_fa);  



%% �������ݣ����ڿ��ǵ��ǲ�ͬʱ϶�¶��ŵ����и��ʵĹ��ƣ���˼�¼ sample_num = 100��500��1000��2000 �µ� estimated_prob

fprintf('the simulation time is');
sample_num

fprintf('the estimated idle probability is');
estimated_prob

fprintf('the chosen number is');
T

%% ��ø�������
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

%% ��ͼ
% figure;
% t = [1:sample_num];
% plot(t, Reward_average_as_time,'b');
% xlabel('time');
% ylabel('reward')
% hold on;
% plot(1:sample_num, Reward_expect_all_status * ones(1,sample_num), 'r');
% toc
%% ��ͼ algorithm_1_1
% figure;
% xlabel('time');
% ylabel('Regret/log(t)')
% hold on;
% plot(sample_num_plot, Regret_log_plot, 'r*--','LineWidth',2);
% toc  
      
       
 
                   
           
           
 
               
       



