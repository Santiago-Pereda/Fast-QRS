function b = rqrb0_fast(x,y,w,G,zeta,m,b0)

%Algorithm with preprocessing for Rotated Quantile Regression (RQR) for a
%grid of quantiles, the rotation obtained with a copula and  initial values
%of the beta coefficients (used by Algorithms 3-4).
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
%m = parameter to select interval of observations in top and bottom groups
%(by default, it is equal to 1)
%
%zeta = conservative estimate of the standard error of the residuals
%
%b0 = initial value of beta parameters
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

%Compute beta parameters with preprocessing (steps 1-4)
for i1=1:1:Q
    b(:,i1)=rqrtau_fast(x,y,w,G(:,i1),zeta,m,b0(:,i1)');
end