 


% 估计的概率
 

% estimated =   [   0.0499,0.6435,0.2852 ,0.7326, 0.4541,0.6913,0.3624,0.6017, 0.1354,0.9707,0.4904 ,0.8799, 0.5580, 0.4067,0.7994;
%        0.0667 ,   0.6667   , 0.2667  ,  0.6667   , 0.6476  ,  0.5444 ,   0.4276,    0.0000 ,   0.5630 ,   0.8676,    0.2667  ,  0.9763, 0.7282,    0.4667 ,   0.9514;
%        0.1000  ,  0.7103,    0.0783 ,   0.6853 ,   0.5104  ,  0.7387  ,  0.3250 ,   0.6152  ,  0.0316  ,  0.9830 ,   0.5825  ,  0.8347 ,  0.4411  ,  0.4269  ,  0.7643;
%         0.0296 ,   0.5312 ,   0.1135 ,   0.7554  ,  0.4249  ,  0.6824 ,   0.3571  ,  0.5391  ,  0.0615,    0.9273  , 0.4600,    0.8615,    0.5646 ,   0.4187 ,   0.7893   ;
%         0.0774  ,  0.6550  ,  0.2129  ,  0.7189  ,  0.5423 ,   0.6802,    0.3814,    0.6078 ,   0.2066 ,   0.9691,    0.5111,  0.8401 ,   0.5398,    0.4777  ,  0.8265]
%     
% 
% bar3(estimated,0.6)
% hXLabel = xlabel('channel');
% hYLabel = ylabel('number');
% hZLabel = zlabel('idle probability');
 

% 拉动的次数

% 100, 500, 1000

% 横向 为  时间 t=100,300,500,800,1000,1300, 1500
% 纵向 为 M 不同的数量
t = [       1e03,  2000, 3000,   4000,   5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000 ];
cc = [      497.15000,        868.3000,     1.1933e+03,     1.4326e+03 ,     1.7269e+03,1.8417e+03, 1.8601e+03, 2.1597e+03, 2.2151e+03, 2.3548e+03, 2.4392e+03,  2.4462e+03   ] ;
bar(t,cc)
figure_FontSize = 14
xlabel('time slot','FontSize',figure_FontSize)
ylabel('collision number','FontSize',figure_FontSize)
% play_num = [
% 53.7400,     252.4500 ,  497.15000,     683.5500  ,   868.3000,  1.0038e+03,  1.1933e+03,  1.3132e+03,  1.4326e+03 ,  1.5354e+03 ,  1.7269e+03  ,  76;
% 33  , 114 ,   60 ,  203,    66  , 142 ,   64 ,  129 ,   42 ,  289  ,  74  , 279 ,   98,    68 ,  148;
% 49 ,  243   , 64  , 404 ,   93  , 236 ,   82  ,  99 ,   55,   489 ,  145 ,  491,   109 ,   49,   401;
% 92 ,  249 ,   71 ,  572,   167 ,  495  , 105  , 246 ,   71,   797 ,  144,   737 ,  259 ,  167,   637;
% 62 ,  570  , 125 ,  515  , 221,   562 ,  175 ,  359 ,   64  , 955 ,  182 ,  877 ,  246 ,  194 ,  902
% ]

% % bar3(play_num,0.6)
% hXLabel = xlabel('channel');
% hYLabel = ylabel('number');
% hZLabel = zlabel('chosen number');






























