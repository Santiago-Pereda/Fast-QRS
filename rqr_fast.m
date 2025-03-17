function b = rqr_fast(x,y,w,G,zeta,m,initq)

%Algorithm 2
%Algorithm with preprocessing for Rotated Quantile Regression (RQR) for a 
%grid of quantiles and the rotation obtained with a copula.
%
%
%Input:
%
%x = regressors
%
%y = dependent variable
%
%w = sample weights (by default, it is equal to 1 for every observation)
%
%G = copula conditional on participation
%
%zeta = conservative estimate of the standard error of the residuals
%
%m = parameter to select interval of observations in top and bottom groups
%(by default, it is equal to 1)
%
%initq = initial quantile to estimate regularly and obtain initial values 
%for preprocessing afterwards
%
%
%Output:
%
%b = estimated beta parameters

[N,K]=size(x);

%Sample weights
if isempty(w)
    w=ones(N,1);
end

%Parameter to select interval of observations in top and bottom groups
if isempty(m)
    m=1;
end

%Number of quantiles
Q=size(G,2);

%Pregenerate matrix with beta parameters
b=zeros(K,Q);

%Compute beta parameters for initial quantile (initial step)
b(:,initq)=rq(x,y,w,G(:,initq));

%Compute remaining beta parameters with preprocessing (steps 1-4)
for i1=initq+1:1:Q
    b(:,i1)=rqrtau_fast(x,y,w,G(:,i1),zeta,m,b(:,i1-1)');
end
for i1=1:1:initq-1
    b(:,initq-i1)=rqrtau_fast(x,y,w,G(:,initq-i1),zeta,m,b(:,initq-i1+1)');
end