% 定义系统参数
nu = 3; % 系统输入的阶数
ny = 2; % 系统输出的阶数
d = 1; % 时间延迟

theta = [1, 1.72]; % 非线性参数
alpha = [1.1, -0.7]; % 输出权重系数
beta = [1.2, 0.6, -0.7]; % 输入权重系数

% 生成系统输入数据
N = 300; 
P = 5;
K = 5; %假定的K值
N_u = N + P; % 训练数据的总数量 （取305而不是300，因为构建D_u字典时，i = K-P, ... ,K-1是递减的，有“根据时间向前滚动”效果）
u = 2 * rand(N_u, 1); % 假设系统输入为[0, 2]范围内的均匀分布随机数

% 生成系统输出数据 模拟系统输出
[y, g] = hammerstein_model_replication(u, theta, alpha, beta, nu, ny, d);

% 添加噪声
SNR = 60;
y_noisy = awgn(y, SNR, 'measured');

t = 1:N_u; % 时间向量
% 画出输入信号u
figure;
plot(t, u, 'b', 'LineWidth', 2);
title('System Input');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 画出无噪声输出
figure;
plot(t, y, 'b', 'LineWidth', 2);
title('System Output without Noise');
xlabel('Time');
ylabel('Amplitude');
grid on;

% 画出含噪声输出
figure;
plot(t, y_noisy, 'r', 'LineWidth', 2);
title(['System Output with Gaussian White Noise (SNR = ' num2str(SNR) ' dB)']);
xlabel('Time');
ylabel('Amplitude');
grid on;

% 生成所有可能的theta_s组合 （模拟sampling然后组合三个theta值的过程）
theta_s1_vals = 0:0.1:1;
theta_s2_vals = 0:0.1:1;
theta_s3_vals = 0:0.1:1;
[theta_s1_grid, theta_s2_grid, theta_s3_grid] = ndgrid(theta_s1_vals, theta_s2_vals, theta_s3_vals);
theta_s_combinations = [theta_s1_grid(:), theta_s2_grid(:), theta_s3_grid(:)];

disp(theta_s_combinations);
%测试theta_s输出
first_theta = theta_s_combinations(1, :);   % 获取第一组 theta_s
middle_theta = theta_s_combinations(5, :);  % ...
last_theta = theta_s_combinations(end, :);  % 获取最后一组 theta_s
%disp(first_theta);
%disp(last_theta);

% 初始化D_u字典
D_u = zeros(N, numel(theta_s1_vals)^3 * P);

% 遍历Du_0到Du_4
for offset = 0:4    %依照Du_i构建D_u
    % 初始化当前Du_x矩阵
    Du_x = zeros(N, numel(theta_s1_vals)^3);
    
    % 遍历每一个theta_s组合
    for i = 1:size(theta_s_combinations, 1)
        % 提取当前的 theta_s1, theta_s2, theta_s3
        theta_s1 = theta_s_combinations(i, 1);
        theta_s2 = theta_s_combinations(i, 2);
        theta_s3 = theta_s_combinations(i, 3);
        
        % 计算对应的g(k)列向量，考虑偏移
        g_k = theta_s1 * sin(pi/2 * u(1+offset:N+offset)) + ...
              theta_s2 * sin(pi * u(1+offset:N+offset)) + ...
              theta_s3 * sin(pi * 1.5 * u(1+offset:N+offset));
        
        % 将计算结果g(k)存储为Du_x的第i列
        Du_x(:, i) = g_k;
    end
    
    % 将当前Du_x矩阵拼接到D_u矩阵
    D_u(:, offset*numel(theta_s1_vals)^3 + (1:numel(theta_s1_vals)^3)) = Du_x;
end

% 输出结果
% D_u现在是一个shape为(N,5*11^3)的矩阵，包含Du_0到Du_4的所有列

%{
disp(D_u);  %测试
du_size1 = size(D_u);   
disp('D_u 的尺寸是:'); %(300, 6655) 
disp(du_size1); %测试
%}
D_y = zeros(N, P);   % 初始化 D_y  D_y的shape为(300,5)

% 填充D_y字典
for j = 1:P
    % 对于每一列，将 y_noisy 向量赋值给 D_y 的对应列    
    D_y(:, j) = y_noisy(j:j+N-1);
end

y_K = y_noisy(K+1:K+N);

D_yu = [D_y D_u];

xyu = OMP(D_yu, y_K, 20);

%================================以下，画xyu点图================================

n = numel(xyu); % 向量中的元素数量
x = 1:n; % 横坐标是从1到n的整数

figure; % 创建一个新图形窗口
hStem = stem(x, xyu, 'b', 'filled'); % 创建散点图，使用蓝色填充
hStem.BaseLine.Color = 'none'; % 隐藏基线

% 设置标题、坐标轴标签
title('x_y_u Amplitude');   %title里下划线得到下标显示
xlabel('Element subscript');
ylabel('Amplitude');

% 调整横坐标的范围和纵坐标范围，以更好地显示数据
xlim([0, n+1]);
ylim([min(xyu)-1, max(xyu)+1]);

% 修改散点图对象的颜色属性为蓝色
set(hStem, 'Color', 'b', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');

% 保存图形到文件   不如直接截图
%saveas(gcf, 'xyu amplitude_vs_element_subscript.png');

%================================以上，画xyu点图================================

% Zero-Operation    把xy和xu分开处理（分开找最大值）
tau = -12;  %不同于文中数值仿真部分取-25，这里尝试一下小一点的tau值（实测发现在本theta采样背景下也较合适）

% 提取向量xy和xu
xy = xyu(1:P);
xu = xyu(P+1:end);

% 找到xy中绝对值最大的元素及其索引
[~, m] = max(abs(xy));
xym = xy(m);

% 对xy中的每个元素进行操作 （严格依照paper中的判定方法）
for i = 1:length(xy)
    A = abs(xy(i)) / abs(xym); % 计算A的值
    if 20*log10(A) < tau
        xy(i) = 0; % 如果条件成立，设置对应的元素为0
    end
end

% 输出修改后的xy向量

% 找到xu中绝对值最大的元素及其索引
[~, m] = max(abs(xu));
xum = xu(m);

% 对xu中的每个元素进行操作
for i = 1:length(xu)
    A = abs(xu(i)) / abs(xum); % 计算A的值
    if 20*log10(A) < tau
        xu(i) = 0; % 如果条件成立，设置对应的元素为0
    end
end

xyu_z = [xy; xu];   %拼接得到了xyu_z
%{
% 输出修改后的xu向量
disp('xu:');    %测试
disp(xu);
disp('---');

xyu_z = [xy; xu];
disp('xyu_z:');
disp(xyu_z);    %测试
%}

%================================以下，画xyu_z图================================

n = numel(xyu_z); % 向量中的元素数量
x = 1:n; % 横坐标是从 1 到 n 的整数

figure; % 创建一个新图形窗口
hStem = stem(x, xyu_z, 'b', 'filled'); % 创建散点图，使用蓝色填充
hStem.BaseLine.Color = 'none'; % 隐藏基线

% 设置标题、坐标轴标签
title('x_y_u^z Amplitude'); %title里 ^ 得到上标显示
xlabel('Element subscript');
ylabel('Amplitude');

% 调整横坐标的范围和纵坐标范围，以更好地显示数据
xlim([0, n+1]);
ylim([min(xyu_z)-1, max(xyu_z)+1]);

% 修改散点图对象的颜色属性为蓝色
set(hStem, 'Color', 'b', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');

% 保存图形到文件 不如直接截图
%saveas(gcf, 'xyu_z amplitude_vs_element_subscript.png');

%================================以上，画xyu_z图================================

% Conversion-Operation
% 计算字典D_yu中每列的L2-范数
l_yu = vecnorm(D_yu);

% “防止除零操作，可以将所有的零范数设置为一个非常小的数”
l_yu(l_yu == 0) = eps; % eps是MATLAB中非常小的正数
l_yu_transposed = l_yu.';

%%G = P + m * P; % 计算G的大小  发现G没有用上，因为向量的shape已经默认使用了G的值
xyu_c = xyu_z ./ l_yu_transposed; % 元素对应相除


%================================以下，画xyu_c图================================

n = numel(xyu_c); % 向量中的元素数量
x = 1:n; % 横坐标是从 1 到 n 的整数

figure; % 创建一个新图形窗口
hStem = stem(x, xyu_c, 'b', 'filled'); % 创建散点图，使用蓝色填充
hStem.BaseLine.Color = 'none'; % 隐藏基线

% 设置标题、坐标轴标签
title('x_y_u^c Amplitude');
xlabel('Element subscript');
ylabel('Amplitude');

% 调整横坐标范围、纵坐标范围，以更好地显示数据
xlim([0, n+1]);
ylim([min(xyu_c)-1, max(xyu_c)+1]);

% 修改散点图对象的颜色属性为蓝色
set(hStem, 'Color', 'b', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');

% 保存图形到文件   不如直接截图
%saveas(gcf, 'xyu_c amplitude_vs_element_subscript.png');

%================================以上，画xyu_c图================================

% Integration-Operation
% 定义分割点
%split_points = [6, 1337, 2668, 3999, 5330];    % 对应具体参数时
split_points = [P+11^3*0+1, P+11^3*1+1, P+11^3*2+1, P+11^3*3+1, P+11^3*4+1];     % general时 泛化时（还不是最终的泛化形式） 11^3是3个theta采样并组合的总可能个数
theta_s_combinations_size = 11^3; % theta_s_combinations 的大小  同上一行的注释

% 存储所有组合的单元数组
all_combinations = {};

% 对于每个小矩阵的范围
for i = 1:length(split_points)
    % 计算当前小矩阵的起始和结束下标
    start_idx = split_points(i);
    if i == length(split_points)
        end_idx = P+11^3*5; % 6660 最后一组的subscript
    else
        end_idx = split_points(i+1) - 1;
    end
    
    % 获取对应的xyu_c中的元素
    segment = xyu_c(start_idx:end_idx);
    
    % 检测非零元素并获取下标
    non_zero_elements = segment(segment ~= 0);  % ~= “不等于”
    non_zero_indices = find(segment ~= 0) + (start_idx - 1); % 全局下标
    
    % 如果非零元素多于一个，则进行Integration运算：
    if numel(non_zero_elements) > 1
        % 计算临时的新subscript值
        temp_new_subscript = non_zero_indices - (5 + theta_s_combinations_size * (i - 1));
        
        % 打印结果
        fprintf('在第 %d 个小矩阵范围内，进行 A 运算的非零元素和对应的theta_s组合为：\n', i);  % 测试，用fprintf来控制输出的格式%d
        
        % 遍历每个临时的新subscript
        for j = 1:length(temp_new_subscript)
            % 提取相应的theta_s组合
            theta_s = theta_s_combinations(temp_new_subscript(j), :);
            fprintf('非零元素：%.2f, Subscript值：[%d], theta_s组合：[%f, %f, %f]\n', ... % 测试
                    non_zero_elements(j), non_zero_indices(j), theta_s(1), theta_s(2), theta_s(3));
            all_combinations{end+1} = [theta_s, non_zero_elements(j)];
        end
        
    end
end    
% 打印所有组合 （测试）
disp('{');
for i = 1:length(all_combinations)
    fprintf('[(%f, %f, %f), %f], ', all_combinations{i});   % 测试
end
disp('}');

disp(all_combinations{1});  % 测试

% 初始化U和X
U = []; 
X = []; 

% 遍历all_combinations，提取theta_s组合和非零元素值
for i = 1:length(all_combinations)
    % 提取出theta_s组合，假设theta_s组合是前3个元素
    U = [U; all_combinations{i}(1:3)]; % 添加到U矩阵中
    % 提取出非零元素值，假设非零元素值是第4个元素
    X = [X; all_combinations{i}(4)]; % 添加到X矩阵中
end

% 打印U和X （测试）
disp('U = ');
disp(U);
disp('X = ');
disp(X);

U_transpose = U.';
% 计算Phi paper中定义为Phi
Phi = U_transpose * X;

% 提取Phi中的第一个元素，命名为phi_0
phi_0 = Phi(1);

% 计算Phi / phi_0
theta_e_predi = Phi / phi_0;    %theta_e_predi就是theta_e的预测值（paper中为theta_e_帽）

% 输出theta_e_predi
disp('theta_e_predi = ');
disp(theta_e_predi);
