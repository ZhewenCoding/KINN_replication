% 加载训练好的KIFNN神经网络
load('trained_KIFNN.mat', 'net_kinn'); % 加载神经网络

% 初始化参数
P = 0.05;   % 比例增益
I = 0.4;    % 积分增益
D = 1.1;    % 微分增益
T = 50;     % 模拟时间，测试50和200
dt = 1;     % 采样时间

% 设定参考轨迹
setpoint = ones(T, 1);
setpoint(11:20) = 1.5;
setpoint(21:30) = 2;
setpoint(41:50) = 1.5;

% 初始化变量
y_pid = zeros(T, 1); % PID系统输出
u_pid = zeros(T, 1); % PID控制输入
y_mpc = zeros(T, 1); % KINNMPC系统输出
u_mpc = zeros(T, 1); % KINNMPC控制输入
time = (0:T-1)';     % 时间向量
e = zeros(T, 1);     % PID误差
e_int = 0;           % PID积分误差
e_prev = 0;          % PID前一时刻误差

% 初始状态
y_pid(1) = 0;        % PID系统初始输出
y_mpc(1) = 0;        % KINNMPC系统初始输出

% 初始化输入历史记录
u_hist = zeros(3, 1);
y_hist = zeros(2, 1);

% PID控制器
for t = 2:T
    e(t) = setpoint(t) - y_pid(t-1); 
    e_int = e_int + e(t) * dt;
    e_der = (e(t) - e_prev) / dt;
    u_pid(t) = P * e(t) + I * e_int + D * e_der;
    y_pid(t) = systemDynamics(u_pid(t), y_pid(t-1)); % 使用动态方程
    e_prev = e(t);
    
    % 更新历史记录
    u_hist = [u_pid(t); u_hist(1:2)];
    y_hist = [y_pid(t); y_hist(1)];
end

% KINNMPC控制器
G = 20; H = 6; rho = 0.2; mu = 0.5; sigma = 1;
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter', 'MaxIterations', 5, 'UseParallel', false);

for t = 2:T
    y0 = y_mpc(t-1);
    if t+G-1 > T
        y_ref = setpoint(t:end);
        y_ref = [y_ref; setpoint(end)*ones(G-length(y_ref),1)];
    else
        y_ref = setpoint(t:t+G-1);
    end
    u0 = zeros(H, 1);
    lb = -inf(H, 1); % 设置下边界
    ub = inf(H, 1);  % 设置上边界
    objective = @(u) calculateObjective(net_kinn, u, G, H, rho, mu, sigma, y_ref, y0, u_hist, y_hist);
    u_opt = fmincon(objective, u0, [], [], [], [], lb, ub, [], options);
    u_mpc(t) = u_opt(1);
    % 使用神经网络预测
    input_data = [u_mpc(t); u_hist; y_mpc(t-1); y_hist];
    input_data = input_data(1:5); % 只取前5个元素，确保输入维度为5
    input_data = reshape(input_data, 5, 1); % 转换为5行1列的列向量
    y_mpc(t) = net_kinn(input_data); % 使用神经网络进行预测

    % 更新历史记录
    u_hist = [u_mpc(t); u_hist(1:2)];
    y_hist = [y_mpc(t); y_hist(1)];
    disp(['Time step: ', num2str(t), ', u_mpc: ', num2str(u_mpc(t)), ', y_mpc: ', num2str(y_mpc(t))]);
end

% 绘图
figure;
plot(time, y_pid, 'b', 'LineWidth', 1.5); hold on;
plot(time, y_mpc, 'g', 'LineWidth', 1.5); 
plot(time, setpoint, 'r--', 'LineWidth', 1.5); hold off;
xlabel('Time');
ylabel('Output');
title('PID and KINNMPC Control Response');
legend('PID Output', 'KINNMPC Output', 'Setpoint');
grid on;

% 系统动态方程
function y_next = systemDynamics(u, y)
    y_next = y + 0.5 * (u - y); 
end

% KINNMPC目标函数
function cost = calculateObjective(net_kinn, u, G, H, rho, mu, sigma, y_ref, y0, u_hist, y_hist)
    y_pred = zeros(G, 1);
    u_pred = u(1:H);
    y_pred(1) = y0;

    for k = 2:G
        if k <= H
            u_current = u_pred(k-1);
        else
            u_current = u_pred(H);
        end
        % 构建输入数据
        input_data = [u_current; u_hist; y_pred(k-1); y_hist];
        input_data = input_data(1:5); % 只取前5个元素，确保输入维度为5
        input_data = reshape(input_data, 5, 1); % 转换为5行1列的列向量
        %disp(['输入数据大小: ', num2str(size(input_data))]); % 调试输出输入数据大小
        y_pred(k) = net_kinn(input_data); % 使用神经网络进行预测
    end

    error = y_ref(1:G) - y_pred;
    cost = sum((error.^2) .* (1:G)') + rho * sum(diff(u_pred).^2) + mu * sum((u_pred - u_pred(1)).^2);
end

