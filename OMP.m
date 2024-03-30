function xyu = OMP(D_yu, y_K, S)
    % 初始化变量
    r = y_K;                          % 残差Residual
    xyu = zeros(size(D_yu, 2), 1);   % 初始化稀疏向量
    supportSet = [];                % 建立支持集
    epsilon = 1e-6;   
    
    for iter = 1:S
        
        correlations = D_yu' * r;   % 将残差与Dyu的列相关联
        
        [~, idx] = max(abs(correlations));  % 选择具有最大相关性的索引
        
        supportSet = unique([supportSet, idx]); % 更新支持集
        
        subDyu = D_yu(:, supportSet);   % 解最小二乘问题
        sparseCoeffs = pinv(subDyu) * y_K;
        
        xyu(supportSet) = sparseCoeffs; % 更新
        
        r = y_K - D_yu * xyu;   % 更新残差
        
        if norm(r) < epsilon    % 判断残差是否足够小
            break;
        end
    end
end


