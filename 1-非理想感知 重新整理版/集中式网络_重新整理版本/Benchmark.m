%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description : This is used to obtain the benchmark when the statistical
% information is known by users
%               
%               Author: xxx
%               Date  : 2019/03/22
%               Modified: 2023/05/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ����ʵ������
channel_free_prob = [  0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1  ]'; % �ŵ����и���

channel_free_prob = [0.8811, 0.5390,0.3468,0.9522,0.7823,0.0471,0.7968]';   

N = length(channel_free_prob); % �ŵ���������
M = 2; % ��ʾ��֪�ŵ�����
K = 1; % ��ʾ�ŵ���������
P_d = 0.8;  % ������
P_fa = 0.3;  % �龯���ʣ����û�δ����������ж�Ϊ����

%% Benchmark: ������� M*�� ѡ����õ� M ���ŵ����ҳ����е��ŵ���������K���ŵ���
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
Reward_expect_all_status = Reward_expect_all_status * (1-P_fa)





