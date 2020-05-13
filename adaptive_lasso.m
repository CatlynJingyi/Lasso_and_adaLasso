function [beta] = adaptive_lasso(X,y,lambda,gamma,weight)
% Adaptive Lasso
%----------------------------------------
% Inputs:
% X: n*p data matrix
% y: respond vector
% lambda, gamma, weight: see (1.12) in paper
%-----------------------------------------
% Outputs:
% beta: adaptive lasso's estimator
%% 
if nargin<5
    weight=[];
end
if nargin<4
    gamma=1; % default=1
end
if nargin<3
    lambda=[];
end
%% 
p=size(X,2);
if isempty(weight)
    if rank(X'*X)<p
        disp('better find a ridge solution first');
        if isempty(lambda)
            disp('need lambda!');
            return;
        else
            beta_or=(X'*X+lambda*eye(p))\X'*y;
        end
    else
        beta_or=regress(y,X); % use OLS's beta estimator
    end
    weight=1./(abs(beta_or).^gamma);
end

X_new=X./weight';
if ~isempty(lambda)
    beta_lasso=lasso(X_new,y,lambda);
else
    beta_lasso=lasso(X_new,y);
end
beta=beta_lasso./weight;
end

