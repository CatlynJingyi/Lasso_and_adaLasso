function [R] = downdateR(R,k)
%%------------------------------------------
% downdate matrix R
% k=deleteindex
%%
if nargin<2
    disp('missing values')
    return;
end

R(:,k)=[];
dimR=size(R,2);
for i=k:dimR
    [G,R(i:(i+1),i)]=planerot(R(i:(i+1),i));
    if i<dimR
        R(i:(i+1),(i+1):dimR)=G*R(i:(i+1),(i+1):dimR);
    end
end
R(end,:)=[];
end

