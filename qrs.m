function [beta,theta,m_min,m,b]=qrs(y,x,prop,w,Q,family,gridtheta)

%Estimation of QRS
%
%
%Input:
%
%y = dependent variable
%
%x = regressors
%
%prop = propensity score
%
%w = sample weights (by default, it is equal to 1 for every observation)
%
%Q = number of quantiles in grid
%
%family = copula family
%
%gridtheta = grid of values for copula parameter
%
%
%Output:
%
%beta = estimated beta parameters
%
%theta = estimated copula parameter
%
%m_min = value of objective function at the optimum
%
%m = value of objective function for grid of values for copula parameter
%
%b = estimated beta parameters for grid of values for copula parameter

[N,K]=size(x);

%Sample weights
if isempty(w)==1
    w=ones(N,1);
end

%Number of quantiles
gridq=linspace(1/(Q+1),Q/(Q+1),Q);

%Instrument
phi=prop.*w;

%Prevent the conditional copula from being too close to 0 or 1
eps=.00001;

%Pregenerate matrices to store values of beta coefficients and criterion
%function
m=zeros(length(gridtheta),1);
b=zeros(K,Q,length(gridtheta));
for i1=1:1:length(gridtheta)
    theta=gridtheta(i1);
    
    %Copula conditional on participation
    C=reshape(copulacdf(family,[kron(ones(N,1),gridq'),kron(prop,ones(Q,1))],theta),Q,N)';
    G=C./(prop*ones(1,Q));
    G=max(min(G,1-eps),eps);
    
    %Slope parameters given copula
    for i2=1:1:Q
        b(:,i2,i1)=rq(x,y,w,G(:,i2));
    end
    
    %Objective function for copula parameter
    m(i1)=((phi'*(sum(double(y*ones(1,Q)<=x*b(:,:,i1))-G,2)))/N).^2;
end

%Find minimum of objective function
[m_min,argminf]=min(m);

%Optimum copula and beta parameters
theta=gridtheta(argminf);
beta=reshape(b(:,:,argminf),K,Q);