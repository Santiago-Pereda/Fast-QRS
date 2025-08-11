function rho=checks_rqr(y,x,prop,w,nq,family,t,b)

%Check function for RQR estimates
%
%Input:
%
%y=dependent variable
%
%x=regressors
%
%prop=propensity score
%
%w=sample weights
%
%nq=number of quantiles in grid
%
%family=copula family
%
%t=copula parameter
%
%b=RQR coefficients
%
%Output:
%
%rho=value of the rotated check function

[N,~]=size(x);

gridq=linspace(1/(nq+1),nq/(nq+1),nq);

eps=0.00001;

%Compute the copula conditional on participation
C=reshape(copulacdf(family,[kron(ones(N,1),gridq'),kron(prop,ones(nq,1))],t),nq,N)';
G=C./(prop*ones(1,nq));

%Ensure that the conditional copula is strictly between 0 and 1
G=max(min(G,1-eps),eps);

rho=zeros(nq,1);
for i1=1:1:nq
    [rho(i1),~]=checkfn(w.*(y-x*b(:,i1)),G(:,i1));
end