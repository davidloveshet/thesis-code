
%--------------------------------------------------------------------------------------------------------------------------------------------------%
% Description : This is used to obtain the benchmark of the algo
%               
%               Author: xxx
%               Date  : 2019/03/22
%               Modified : 2023/05/05
% ����˵����
% ��Ҫ���ŵ����и��ʼ�¼�� channel_free_prob �У��ٰ��� ����ʽ(3.34)
% �Բ�ͬ��Ҫ�û��������������������õ�ֵ��¼�ڷ����ļ��е� Reward_expect_all
   
%--------------------------------------------------------------------------------------------------------------------------------------------------%

M = 4;
K = 1;
U = 3;
P_d = 0.8;  % ������
P_fa = 0.3;  % �龯���ʣ����û�δ����������ж�Ϊ����

% channel_free_prob = [0.3395 0.5221 0.0346 0.0220 0.4045 0.3279 0.9503 0.6190 0.8405 0.9522 0.0459 0.8273 0.9013 0.8677 0.1579 0.2270 0.5197 0.1143 0.4079 0.7249]';
% channel_free_prob = [0.0965,0.1320,0.9221,0.9861,0.5352,0.0598,0.2348,0.3532,0.8612,0.0154,0.0430,0.1690,0.6891,0.7317,0.6477,0.4709,0.5870,0.2963,0.7847,0.1890]';
% channel_free_prob = [0.6923,0.5430,0.3544,0.8753,0.5212,0.6759,0.8783,0.9762]'; 
% channel_free_prob = [0.9822    0.9503    0.9013    0.8677    0.8405    0.8273    0.7249    0.6190    0.5221    0.5197    0.4079    0.4045    0.3395    0.3279    0.2270    0.1579    0.1143    0.0459    0.0346    0.0220 ]'; 

% channel_free_prob = [ 0.9494    0.9091    0.8000    0.7826    0.6809    0.6170    0.5390    0.3854    0.3433  0.0421 ];


channel_free_prob = [0.0965,0.1320,0.9221,0.9861,0.5352,0.0598,0.2348,0.3532,0.8612,0.0154,0.0430,0.1690,0.6891,0.7317,0.6477,0.4709,0.5870,0.2963,0.7847,0.1890]';


% channel_free_prob = [ 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.1 ];





channel_free_prob = sort(channel_free_prob,'descend');
N = length(channel_free_prob);

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

% f_theta_M_best = f_theta_sort(1:M); % �ҳ� M �������ŵ�
% f_theta_M_best_sequence = f_theta_sort_sequence(1:M); % ��¼ M �������ŵ����

%% for user 1, M = 2
% f_theta_M_best(1) = f_theta_sort(2);
% f_theta_M_best(2) = f_theta_sort(6);
% f_theta_M_best(3) = f_theta_sort(7);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(2); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(6);
% f_theta_M_best_sequence(3) = f_theta_sort_sequence(7);

%% for user 1, M = 2
% f_theta_M_best(1) = f_theta_sort(1);
% f_theta_M_best(2) = f_theta_sort(6);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(1); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(6); 


%% for user 2, M = 2
% f_theta_M_best(1) = f_theta_sort(2);
% f_theta_M_best(2) = f_theta_sort(5);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(2); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(5); 

%% for user 3, M = 2
% f_theta_M_best(1) = f_theta_sort(3);
% f_theta_M_best(2) = f_theta_sort(4);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(3); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(4); 

%% for user 1, M = 4
f_theta_M_best(1) = f_theta_sort(1);
f_theta_M_best(2) = f_theta_sort(10);
f_theta_M_best(3) = f_theta_sort(11);
f_theta_M_best(4) = f_theta_sort(12);
f_theta_M_best_sequence(1) = f_theta_sort_sequence(1); % ��¼ M �������ŵ����
f_theta_M_best_sequence(2) = f_theta_sort_sequence(10);
f_theta_M_best_sequence(3) = f_theta_sort_sequence(11);
f_theta_M_best_sequence(4) = f_theta_sort_sequence(12);

%% for user 2, M = 4
% f_theta_M_best(1) = f_theta_sort(2);
% f_theta_M_best(2) = f_theta_sort(7);
% f_theta_M_best(3) = f_theta_sort(8);
% f_theta_M_best(4) = f_theta_sort(9);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(2); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(7);
% f_theta_M_best_sequence(3) = f_theta_sort_sequence(8);
% f_theta_M_best_sequence(4) = f_theta_sort_sequence(9);

%% for user 3, M = 4
% f_theta_M_best(1) = f_theta_sort(3);
% f_theta_M_best(2) = f_theta_sort(4);
% f_theta_M_best(3) = f_theta_sort(5);
% f_theta_M_best(4) = f_theta_sort(6);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(3); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(4);
% f_theta_M_best_sequence(3) = f_theta_sort_sequence(5);
% f_theta_M_best_sequence(4) = f_theta_sort_sequence(6);

%% for user 1, M = 5
% f_theta_M_best(1) = f_theta_sort(1);
% f_theta_M_best(2) = f_theta_sort(12);
% f_theta_M_best(3) = f_theta_sort(13);
% f_theta_M_best(4) = f_theta_sort(14);
% f_theta_M_best(5) = f_theta_sort(15);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(1); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(12);
% f_theta_M_best_sequence(3) = f_theta_sort_sequence(13);
% f_theta_M_best_sequence(4) = f_theta_sort_sequence(14);
% f_theta_M_best_sequence(5) = f_theta_sort_sequence(15);

%% for user 2, M = 5
% f_theta_M_best(1) = f_theta_sort(2);
% f_theta_M_best(2) = f_theta_sort(8);
% f_theta_M_best(3) = f_theta_sort(9);
% f_theta_M_best(4) = f_theta_sort(10);
% f_theta_M_best(5) = f_theta_sort(11);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(2); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(8);
% f_theta_M_best_sequence(3) = f_theta_sort_sequence(9);
% f_theta_M_best_sequence(4) = f_theta_sort_sequence(10);
% f_theta_M_best_sequence(5) = f_theta_sort_sequence(11);

%% for user 3, M = 5
% f_theta_M_best(1) = f_theta_sort(3);
% f_theta_M_best(2) = f_theta_sort(4);
% f_theta_M_best(3) = f_theta_sort(5);
% f_theta_M_best(4) = f_theta_sort(6);
% f_theta_M_best(5) = f_theta_sort(7);
% f_theta_M_best_sequence(1) = f_theta_sort_sequence(3); % ��¼ M �������ŵ����
% f_theta_M_best_sequence(2) = f_theta_sort_sequence(4);
% f_theta_M_best_sequence(3) = f_theta_sort_sequence(5);
% f_theta_M_best_sequence(4) = f_theta_sort_sequence(6);
% f_theta_M_best_sequence(5) = f_theta_sort_sequence(7);



Reward_expect_all_status = 0; % ���ŵ� M ���ŵ��½����ŵ���throughout
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
Reward_expect_all_status = Reward_expect_all_status * (1-P_fa) 

