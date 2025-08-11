clear
clc

%Parameters to simulate data
N=1000;
K=2;
theta0=.5;
family='Gaussian';

%Quantile grids
Q1=9;
Q2=99;
gridq=linspace(1/(Q2+1),Q2/(Q2+1),Q2);

%Copula grid
gridtheta=linspace(0,.9,91);

%Parameter to select interval of observations in top and bottom groups
m=1;

%Number of evaluated values of parameter with large quantile grid
P=10;

%Parameter for bootstrap weights
gampar=10;

%Number of bootstrap repetitions
reps=200;

%Generate sample
betar=rand(K-1,1);
beta0=[norminv(gridq);betar*gridq];

x=[ones(N,1),2+rand(N,K-1)];
z=[x,rand(N,1)];
copu=copularnd('Gaussian',theta0,N);
v=copu(:,1);
u=copu(:,2);
gamma=[-1.5;.1*rand(K-1,1);2];
beta=[norminv(u),u.^(ones(N,1))*betar'];
prop=exp(z*gamma)./(1+exp(z*gamma));
d=double(v<=prop);
y=d.*sum(x.*beta,2);
w=ones(N,1);

%Compute estimates with Algorithm 3
[betahat1,thetahat1,~,bhat1]=qrs_fast(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q1,Q2,P,family,gridtheta,m);

%Compute bootstrap estimates with Algorithm 4
[betahat1_b,thetahat1_b,~]=qrs_fast_bt(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q1,Q2,P,family,gridtheta,m,bhat1,reps,gampar);