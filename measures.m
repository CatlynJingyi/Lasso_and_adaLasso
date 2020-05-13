function [measure_all] = measures(Xtest,ytest,beta,intercept,sigma)
%%
% measurements of 'MSE', 'R2', 'adjR2'(adjust R2), 'AIC'
%%
if nargin<5
    sigma=[];
end
if nargin<4
    intercept=0;
end
if nargin<3
    disp('missing input values');
    return;
end
%%
[n,p]=size(Xtest);
residual=ytest-Xtest*beta-intercept;
% df: degrees of freedom
if intercept==0
    df=sum(beta~=0);
else
    df=sum(beta~=0)+1;
end
% sigma
if isempty(sigma)
    sigma2=residual'*residual/(n-p-1);
    sigma=sqrt(sigma2);
end
% MSE
measure_all.MSE=sum(residual.^2)/n;
% R2
RSS=sum(residual.^2);
TSS=sum((ytest-mean(ytest)).^2);
measure_all.R2=1-RSS/TSS;
% adjusted R2
measure_all.adjR2=1-(RSS/TSS)*(n-1)/(n-df-1);
% AIC
measure_all.AIC=2*df+n*log(RSS/n);
end

