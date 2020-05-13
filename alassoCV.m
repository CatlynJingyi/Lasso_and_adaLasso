function [best_lambda,info] = alassoCV(X,y,lambda,gamma,K,measure)
%% select optimal lambda by cross validation 
%-------------------------------------------------------------
% Inputs:
% X, y: data matrix
% lambda: giving an array for selection. eg. [0.01, 0.1, 1, 2, ...]
% gamma: a given number, default is 1
% K: folds, default is 5
% measure: 'MSE', 'R2', 'adjR2'(adjust R2), 'AIC'
%-----------------------------------------------------------------
% Outputs:
% best_lambda: the optimal lambda
% info: record measurement
%%
if nargin<6
    measure='MSE'; % default using MSE measure
end
if nargin<5
    K=5; % default K=5
end
if nargin<4
    gamma=1;
end
if nargin<3 || isempty(lambda)
    disp('need lambda!');
    return;
end
%%
n=size(X,1);
if n<K
    disp('data is not enough!');
    return
end
%%
len=length(lambda);
info.lambda=lambda;
info.measure=zeros(1,len);
info.measure_neg=zeros(1,len);
info.measure_pos=zeros(1,len);
for i=1:len
    indices=crossvalind('Kfold',n,K);%indices相同的归为一类
    m=[];
    for j=1:K
        % split data into training data and testing data
        valindex=(indices==j);
        trainindex=~valindex;
        validationX=X(valindex,:);
        validationy=y(valindex,:);
        trainX=X(trainindex,:);
        trainy=y(trainindex,:);
        %
        beta=adaptive_lasso(trainX,trainy,lambda(i),gamma);
        m=[m,measures(validationX,validationy,beta)];
    end
    switch measure
        case 'MSE'
            mmean=mean([m.MSE]);
            mmin=min([m.MSE]);
            mmax=max([m.MSE]);
        case 'R2'
            mmean=mean([m.R2]);
            mmin=min([m.R2]);
            mmax=max([m.R2]);
        case 'adjR2'
            mmean=mean([m.adjR2]);
            mmin=min([m.adjR2]);
            mmax=max([m.adjR2]);
        case 'AIC'
            mmean=mean([m.AIC]);
            mmin=min([m.AIC]);
            mmax=max([m.AIC]);
    end
    info.measure(i)=mmean;
    info.measure_neg(i)=mmean-mmin;
    info.measure_pos(i)=mmax-mmean;
end
%disp(info.measure);
switch measure
    case {'MSE','AIC'}
        [~,index]=min(info.measure);
    case {'R2','adjR2'}
        [~,index]=max(info.measure);
end
best_lambda=lambda(index);
end

