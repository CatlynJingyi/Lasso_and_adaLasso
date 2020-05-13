function [R] = updateR(xnew,xold,R,eps)
%%------------------------------------
% cholesky factorize [xold,xnew]'*[xold,xnew]
%   [xold,xnew]'*[xold,xnew]=R'*R
%%
if nargin<4
    eps=2.220446e-16;
end
if nargin<3
    disp('missing values');
    return;
end

diag=xnew'*xnew;
diag_new=sqrt(diag);
if isempty(R)
    R=diag_new;
else
   col=xnew'*xold;
   r=R'\col'; 
   rnew=diag_new-r'*r;
   if rnew<eps
       rnew=0;
   else
       rnew=sqrt(rnew);
   end
   R=[R,r;zeros(1,size(R,2)),rnew];

end

