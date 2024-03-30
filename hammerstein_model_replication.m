function [y, g] = hammerstein_model_replication(u, theta, alpha, beta, nu, ny, d)
    % 初始化输出
    N = length(u);
    y = zeros(N, 1);
    g = zeros(N, 1);
    
    % 计算非线性块输出
    for k = 1:N
        g(k) = theta(1) * sin(pi/2 * u(k)) + theta(2) * sin(pi * u(k));
        %g(k) = theta(1) + theta(2) * u(k) + theta(3) * (u(k)^2);
    end
    % 应用动态线性块       计算y(t)，需要从t=2开始迭代，因为y(t-1)的计算依赖于前一个时间点的y值

    for t = (ny+d+1):N  %ny+d+1为4，由于原方程涉及到g(t-3)所以t要从4开始递增
        
        if t-d-1 > 0
            g1 = g(t-d-1); % 原方程g(t-1) 对应 g(t-d-1)
        else
            g1 = 0; % 如果索引超出范围，则设为0
        end
        
        if t-d-2 > 0
            g2 = g(t-d-2); % 原方程g(t-2) 对应 g(t-d-2)
        else
            g2 = 0; % 如果索引超出范围，则设为0
        end

        if t-d-3 > 0
            g3 = g(t-d-3); % 原方程g(t-3) 对应 g(t-d-3)
        else
            g3 = 0; % 如果索引超出范围，则设为0
        end

        % 更新y(t+1)的值
        y(t) = alpha(1)*y(t-1) + alpha(2)*y(t-2) + beta(1)*g1 + beta(2)*g2 + beta(3)*g3;
        %y(t) = alpha * y(t-1) + beta * g1;
        %y(t) = alpha(1) * y(t-1) + alpha(2) * y(t-2) + beta(1) * g(t-d-1) + beta(2) * g(t-d-2) + beta(3) * g(t-d-3); % 输出输出的贡献
        %y(t) = alpha(1) * y(t-1) + alpha(2) * y(t-2) + beta(1) * g(t-d) + beta(2) * g(t-d-1) + beta(3) * g(t-d-2); % 输出输出的贡献
    end
 

end