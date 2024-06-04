% 定义样本数量和相应的RMSE值
N = [300, 500, 700, 900, 1000, 1200, 1400, 1600, 1800, 2000];
KIFNN_RMSE_L32 = [2.86, 3.04, 3.76, 2.16, 1.71, 1.73, 2.51, 1.62, 1.47, 1.63];
KIFNN_RMSE_L128 = [1.69, 1.89, 1.50, 1.69, 1.75, 1.60, 1.51, 1.72, 1.60, 1.69];
FNN_RMSE_L32 = [29.78, 10.75, 6.74, 6.92, 8.97, 6.58, 4.65, 5.48, 3.92, 0.94];

% 将百分比转化为小数
KIFNN_RMSE_L32 = KIFNN_RMSE_L32 / 100;
KIFNN_RMSE_L128 = KIFNN_RMSE_L128 / 100;
FNN_RMSE_L32 = FNN_RMSE_L32 / 100;

% 绘制图表
figure;
hold on;
plot(N, KIFNN_RMSE_L32, '-o', 'DisplayName', 'KIFNN\_RMSE\_L=32');
plot(N, KIFNN_RMSE_L128, '-s', 'DisplayName', 'KIFNN\_RMSE\_L=128');
plot(N, FNN_RMSE_L32, '-d', 'DisplayName', 'FNN\_RMSE\_L=32');

% 设置图表属性
xlabel('Data Volume');
ylabel('RMSE');
title('RMSE vs Data Volume');
xlim([200 2000]);
ylim([0 0.4]);
legend;
grid on;

% 显示图表
hold off;