% 加载.mat文件中的数据
% 测试数据
data_u_t_minus_1_test_2005 = load('u_t_minus_1_test_2005_data.mat');
u_t_minus_1_test_2005 = data_u_t_minus_1_test_2005.u_t_minus_1_test_2005;

data_u_t_minus_2_test_2005 = load('u_t_minus_2_test_2005_data.mat');
u_t_minus_2_test_2005 = data_u_t_minus_2_test_2005.u_t_minus_2_test_2005;

data_u_t_minus_3_test_2005 = load('u_t_minus_3_test_2005_data.mat');
u_t_minus_3_test_2005 = data_u_t_minus_3_test_2005.u_t_minus_3_test_2005;

data_y_noisy_test_2005 = load('y_noisy_test_2005_data.mat');
y_noisy_test_2005 = data_y_noisy_test_2005.y_noisy_test_2005;

data_y_noisy_t_minus_1_test_2005 = load('y_noisy_t_minus_1_test_2005_data.mat');
y_noisy_t_minus_1_test_2005 = data_y_noisy_t_minus_1_test_2005.y_noisy_t_minus_1_test_2005;

data_y_noisy_t_plus_1_test_2005 = load('y_noisy_t_plus_1_test_2005_data.mat');
y_noisy_t_plus_1_test_2005 = data_y_noisy_t_plus_1_test_2005.y_noisy_t_plus_1_test_2005;


% 前向填充处理输入数据
u_t_minus_1_test_2005(1) = u_t_minus_1_test_2005(2);  % 如果u(t-1)的第一个值是NaN
u_t_minus_2_test_2005(1:2) = u_t_minus_2_test_2005(3);  % 如果u(t-2)的前两个值是NaN
u_t_minus_3_test_2005(1:3) = u_t_minus_3_test_2005(4);  % 如果u(t-3)的前三个值是NaN
y_noisy_t_minus_1_test_2005(1) = y_noisy_t_minus_1_test_2005(2);  % 如果y_noisy(t-1)的第一个值是NaN

% 后向填充处理输出数据
y_noisy_t_plus_1_test_2005(end) = y_noisy_t_plus_1_test_2005(end-1);  % 如果y_noisy(t+1)的最后一个值是NaN

% 组合输入数据
inputs_test_2005 = [u_t_minus_1_test_2005, u_t_minus_2_test_2005, u_t_minus_3_test_2005, y_noisy_test_2005, y_noisy_t_minus_1_test_2005]; % 确保是水平堆叠
inputs_test_2005_transpose = inputs_test_2005';
disp(size(inputs_test_2005_transpose));
disp('---------');

% 确保目标向量是正确的列向量形式
y_noisy_t_plus_1_test_2005 = y_noisy_t_plus_1_test_2005(:);
y_noisy_t_plus_1_test_2005_transpose = y_noisy_t_plus_1_test_2005';
disp(size(y_noisy_t_plus_1_test_2005_transpose));
disp('---------');

%{
% 验证填充结果
disp('Filled values at start of u_t_minus_1: ');
disp(u_t_minus_1(1:3));
disp('Filled values at start of u_t_minus_2: ');
disp(u_t_minus_2(1:3));
disp('Filled values at start of u_t_minus_3: ');
disp(u_t_minus_3(1:3));
disp('Filled value at start of y_noisy_t_minus_1: ');
disp(y_noisy_t_minus_1(1));
disp('Filled value at end of y_noisy_t_plus_1: ');
disp(y_noisy_t_plus_1(end));
%}

% 检查数据的维度
disp(['Size of inputs: ', mat2str(size(inputs_test_2005))]);
disp(['Size of inputs_transpose: ', mat2str(size(inputs_test_2005_transpose))]);
disp(['Size of target outputs: ', mat2str(size(y_noisy_t_plus_1_test_2005))]);
disp(['Size of target outputs_transpose: ', mat2str(size(y_noisy_t_plus_1_test_2005_transpose))]);

% 创建一个前馈神经网络
net = fitnet(32, 'trainlm'); % 使用Levenberg-Marquardt算法训练

% 设置训练算法
net.trainFcn = 'trainlm';

% 设置性能函数为均方误差(MSE)
net.performFcn = 'mse';
%net.performFcn = 'rmseFcn';    %发现用不了  MATLAB不自带这个函数   所以在后期单独计算RMSE再导出

% 设置正则化参数
%net.performParam.regularization = 0.1;

% 数据划分
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 10/100;

% 训练神经网络
[net, tr] = train(net, inputs_test_2005_transpose, y_noisy_t_plus_1_test_2005_transpose);

save('trained_KIFNN.mat', 'net');

% 使用训练好的网络进行预测
predictions_test = net(inputs_test_2005_transpose);

% 计算RMSE
%errors_test = gsubtract(y_noisy_t_plus_1_test_2005_transpose, predictions_test);
%rmse_test = sqrt(mean(errors_test.^2));
%disp(['RMSE: ', num2str(rmse_test)]);

% 评估网络性能
performance = perform(net, y_noisy_t_plus_1_test_2005_transpose, predictions_test);
performance_RMSE = sqrt(performance);

disp(['Epochs completed: ', num2str(tr.num_epochs)]);
disp(['Training performance (MSE): ', num2str(performance)]);
disp(['Training performance (RMSE): ', num2str(performance_RMSE)]);
