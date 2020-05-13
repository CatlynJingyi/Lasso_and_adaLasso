function [beta,solution_path] = lasso(X,y,lambda,t,normalize,intercept)
%%----------------------------------------------------------------
% compute Lasso by LARS algorithm
%-----------------------------------------------------------------
% Inputs:
% X,y: data matrix
% lambda: parameter from (1.10) in paper
% t: parameter from (1.9) in paper
% nomalize: default is true (requiring normalization)
% intercept: default is true (beta0)
%--------------------------------------------------------------------
% Outputs:
% beta: Lasso's estimator
% solution_path: record (beta, T, lambda, activeset, RSS....)'s change through the solution path
%%
if nargin<6
    intercept=true;
end
if nargin<5
    normalize=true;
end
if nargin<4
    t=[];
end
if nargin<3
    disp('no limitation')
    [beta,solution_path]=lar(X,y);
    return;
end
if nargin<2
    disp('Parameter invalid, please check it.');
    return;
end
%% 
eps=2.220446e-16;
[n,p]=size(X);
maxstep=min(n,p);
% intercept
if intercept==true
    beta0=mean(y);
    X=X-mean(X);
    y=y-beta0;
else
    beta0=0;
end
%标准化
if normalize==true
    stdX=sqrt(ones(1,n)*(X.^2));
    X=X./stdX;
else
    stdX=ones(1,p);
end
    
% Initialization
mu=zeros(n,1);
A=[]; % record activeset
index=1:maxstep; % recort activeset's complement
R=[];
ignore=[]; % records the index of highly-correlated variable
%Gram=X'*X;
beta=zeros(p,1);
solution_path.activeset={};
solution_path.ignoreset={};
solution_path.beta=zeros(p,1);
solution_path.mu=mu;
solution_path.wA=zeros(p,1);
solution_path.AA=[];
solution_path.Told=[];
solution_path.T=[];
solution_path.gamma=[];
%% Iteration
while ~isempty(index)
    cov=X'*(y-mu);
    C=max(abs(cov(index)));
    if C<eps*100
        disp('max|corr|=0, end');
        break;
    end
    newI=find(abs(cov(index))>=C-eps);
    newI=index(newI);
    for I=newI
        %Ik=index(I);
        R=updateR(X(:,I),X(:,A),R,eps);
        if rank(R)==length(A) % whether the new variable is highly-corrleated
            ignore=[ignore,I];
            i=1:length(A);
            R=R(i,i);
        else
            A=[A,I];
            s=sign(cov(A));
        end
        index(index==I)=[];
    end
    pi=length(A);
    GA=R'*R;
    AA=1/sqrt(s'*(GA\s));
    wA=AA*(GA\s);
    uA=X(:,A)*wA;
    solution_path.wA(A,end+1)=wA;
    % compute lars_gamma
    a=X'*uA;
    lars_gamma_temp=[(C-cov(index))./(AA-a(index));(C+cov(index))./(AA+a(index))];
    lars_gamma=min([lars_gamma_temp(lars_gamma_temp>eps);C/AA]);
    % compute lasso_gamma
    lasso_gamma_temp=-beta(A)./wA;
    lasso_gamma_temp(lasso_gamma_temp<=eps)=Inf;
    lasso_gamma=min(lasso_gamma_temp(lasso_gamma_temp>eps));
    % whether fits lasso's requirement
    lassoworks=0;
    gamma=min(lars_gamma,lasso_gamma);
    if lasso_gamma<lars_gamma
        lassoworks=1;
        gamma=lasso_gamma;
        deleteindex=find(lasso_gamma_temp==gamma);
    end
    beta(A)=beta(A)+gamma*wA;
    mu=mu+gamma*uA;
    if lassoworks==1
        R=downdateR(R,deleteindex);
        deleteindex=A(deleteindex);
        A(A==deleteindex)=[];
        beta(deleteindex)=0;
        mu=X*beta;
    end
    index=setdiff(1:maxstep,[A,ignore]);
    Told=norm(beta,1);
    T=norm(beta./stdX',1);
    solution_path.activeset{end+1}=A;
    solution_path.ignoreset{end+1}=ignore;
    solution_path.beta=[solution_path.beta,beta];
    solution_path.mu(:,end+1)=mu;
    solution_path.AA=[solution_path.AA,AA];
    solution_path.Told=[solution_path.Told,Told]; 
    solution_path.T=[solution_path.T,T]; 
    solution_path.gamma=[solution_path.gamma,gamma];
end
%%
solution_path.intercept=beta0;
%%
solution_path.residual=y-X*solution_path.beta;
%% 评价指标
%RSS
solution_path.RSS=sum((solution_path.residual).^2);
%R^2
solution_path.R2=1-solution_path.RSS/solution_path.RSS(1);
%df
len=length(solution_path.activeset);
for i=1:len
    solution_path.df(i)=length(solution_path.activeset{i});
end
if intercept==true
    solution_path.df=[1,solution_path.df+1];
else
    solution_path.df=[0,solution_path.df];
end
%sigma2
RSS=solution_path.RSS(end);
df=solution_path.df(end);
if RSS<eps || df<eps
    sigma2=NaN;
else
    sigma2=RSS/(n-df);
end
solution_path.sigma2=sigma2;
%Cp
Cp=solution_path.RSS/sigma2-n+2*solution_path.df;
solution_path.Cp=Cp;
%%
if ~isempty(lambda)
    Lambda=zeros(1,len);
    for j=(len-1):(-1):1
        Lambda(j)=Lambda(j+1)+(solution_path.Told(j+1)-solution_path.Told(j))*(solution_path.AA(j)^2);
    end
    solution_path.Lambda=Lambda;
    finalindex=1;
    for k=1:len
        if Lambda(k)>lambda
            finalindex=finalindex+1;
        else
            break;
        end
    end
    %disp(finalindex);
    if finalindex==(len+1)
        finalbeta=solution_path.beta(:,finalindex);
    else
        t=solution_path.Told(finalindex)-(lambda-Lambda(finalindex))*(solution_path.AA(finalindex)^(-2));
    end
end
if ~isempty(t)
    %len=length(solution_path.T);
    finalindex=1;
    for j=1:len
        if solution_path.Told(j)<t
            finalindex=finalindex+1;
        else
            break;
        end
    end
    %disp(finalindex);
    Ttrace=[0,solution_path.Told];
    if finalindex==(len+1)
        finalbeta=solution_path.beta(:,finalindex);
    else
        finalbeta=solution_path.beta(:,finalindex)+(t-Ttrace(finalindex))*solution_path.AA(finalindex)*...
            solution_path.wA(:,finalindex+1);
    end
end

solution_path.beta=(solution_path.beta)./stdX';
beta=finalbeta./stdX';
end
