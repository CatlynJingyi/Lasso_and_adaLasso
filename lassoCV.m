function [best_lambda,info] = lassoCV(X,y,lambda,K,measure)
%% select optimal lambda by cross validation 
%-------------------------------------------------------------
% Inputs:
% X, y: data matrix
% lambda: giving an array for selection. eg. [0.01, 0.1, 1, 2, ...]
% K: folds, default is 5
% measure: 'MSE', 'R2', 'adjR2'(adjust R2), 'AIC'
%-----------------------------------------------------------------
% Outputs:
% best_lambda: the optimal lambda
% info: record measurement
%%
if nargin<5
    measure='MSE';
end
if nargin<4
    K=5;
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
    indices=crossvalind('Kfold',n,K);
    m=[];
    for j=1:K
        %分割数据集为训练集和验证集
        valindex=(indices==j);
        trainindex=~valindex;
        validationX=X(valindex,:);
        validationy=y(valindex,:);
        trainX=X(trainindex,:);
        trainy=y(trainindex,:);
        disp(i);
        disp(j);
        beta=lasso(trainX,trainy,lambda(i));
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

