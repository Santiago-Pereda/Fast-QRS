function [S,s]=checkfn(x,tau)

%Check function
%
%Input:
%
%x = variable (usually the residual of the dependent variable minus the
%fitted value at quantile tau)
%
%tau = quantile
%
%Output:
%
%s=vector of values of the check function for each observation
%
%S=sum across observations of the check function

if length(tau)==1
    tau=tau*ones(size(x,1),1);
end

s=tau.*x-(x<=0).*x;

S=sum(s);