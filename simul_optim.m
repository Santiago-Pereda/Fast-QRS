clear
clc

%Number of repetitions
reps=1000;

%Parameters that vary across experiments
params_exp=[10000,2;
    10000,10;
    10000,20;
    20000,2;
    20000,10;
    20000,20];
nexp=size(params_exp,1);
K=max(params_exp(:,2));

%Quantile grids
Q1=9;
Q2=99;
gridq=linspace(1/(Q2+1),Q2/(Q2+1),Q2);

%Copula grid
gridtheta=linspace(0,.9,91);

%Copula parameter
theta0=.5;
family='Gaussian';

%Parameter to select interval of observations in top and bottom groups
m=1;

%number of evaluated values of parameter with large quantile grid
P=10;

times=zeros(nexp,4,reps);
thetad=zeros(nexp,4,reps);
betad=zeros(nexp,4,K,Q2,reps);
checks=zeros(nexp,4,Q2,reps);
ms=zeros(nexp,4,reps);
for i1=1:1:nexp
    %Select parameters that vary across experiments
    N=params_exp(i1,1);
    k=params_exp(i1,2);
    
    betar=rand(k-1,1);
    beta0=[norminv(gridq);betar*gridq];
    
    parfor i2=1:reps
        rng(1000+i2)
        
        times_b=zeros(4,1);
        thetad_b=zeros(4,1);
        betad_b=zeros(4,K,Q2);
        checks_b=zeros(4,Q2);
        ms_b=zeros(4,1);

        %Generate sample for each repetition
        x=[ones(N,1),2+rand(N,k-1)];
        z=[x,rand(N,1)];
        copu=copularnd('Gaussian',theta0,N);
        v=copu(:,1);
        u=copu(:,2);
        gamma=[-1.5;.1*rand(k-1,1);2];
        beta=[norminv(u),u.^(ones(N,1))*betar'];
        prop=exp(z*gamma)./(1+exp(z*gamma));
        d=double(v<=prop);
        y=d.*sum(x.*beta,2);
        w=ones(N,1);
        
        %First estimation: standard algorithm
        start1=tic;
        [b_1,theta_1,m_1]=qrs(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,family,gridtheta);
        times_b(1,1)=toc(start1);
        thetad_b(1,1)=theta_1-theta0;
        betad_b(1,1:k,:)=b_1-beta0;
        checks_b(1,:)=checks_rqr(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,family,theta_1,b_1);
        ms_b(1,1)=m_1;
        
        %Second estimation: algorithm with preprocessing (Algorithm 2
        %repeatedly)
        start2=tic;
        [b_2,theta_2,m_2]=qrs_fast(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,Q2,1,family,gridtheta,m);
        times_b(2,1)=toc(start2);
        thetad_b(2,1)=theta_2-theta0;
        betad_b(2,1:k,:)=b_2-beta0;
        checks_b(2,:)=checks_rqr(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,family,theta_2,b_2);
        ms_b(2,1)=m_2;
        
        %Third estimation: algorithm with preprocessing and grid reduction,
        %only one candidate selected with smaller quantile grid (Algorithm
        %3)
        start3=tic;
        [b_3,theta_3,m_3]=qrs_fast(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q1,Q2,1,family,gridtheta,m);
        times_b(3,1)=toc(start3);
        thetad_b(3,1)=theta_3-theta0;
        betad_b(3,1:k,:)=b_3-beta0;
        checks_b(3,:)=checks_rqr(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,family,theta_3,b_3);
        ms_b(3,1)=m_3;
        
        %Fourth estimation: algorithm with preprocessing and grid
        %reduction, several candidate selected with smaller quantile grid 
        %(Algorithm 4)
        start4=tic;
        [b_4,theta_4,m_4]=qrs_fast(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q1,Q2,P,family,gridtheta,m);
        times_b(4,1)=toc(start4);
        thetad_b(4,1)=theta_4-theta0;
        betad_b(4,1:k,:)=b_4-beta0;
        checks_b(4,:)=checks_rqr(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,family,theta_4,b_4);
        ms_b(4,1)=m_4;
        
        times(i1,:,i2)=times_b;
        thetad(i1,:,i2)=thetad_b;
        betad(i1,:,:,:,i2)=betad_b;
        checks(i1,:,:,i2)=checks_b;
        ms(i1,:,i2)=ms_b;
        disp(['experiment,repetition: ' num2str(i1) ', ' num2str(i2)]);
    end
end

%Collect all results and average across repetitions
msetheta=mean(thetad.^2,3);
msebeta=mean(betad.^2,5);
meanchecks=mean(checks,4);
meanms=mean(ms,3);

tabletimes=mean(times,3);

quantiles=[10,25,50,75,90];
Q=length(quantiles);

tablemse=zeros((1+2*Q),nexp*4);
for i1=1:1:nexp
    tablemse(1,(i1-1)*4+1:i1*4)=msetheta(i1,:);
    for i2=1:1:Q
        tablemse(1+i2,(i1-1)*4+1:i1*4)=msebeta(i1,:,1,quantiles(i2));
        tablemse(1+Q+i2,(i1-1)*4+1:i1*4)=msebeta(i1,:,2,quantiles(i2));
    end
end

tablechecks=zeros(Q,nexp*4);
for i1=1:1:nexp
    for i2=1:1:Q
        tablechecks(i2,(i1-1)*4+1:i1*4)=checks(i1,:,quantiles(i2));
    end
end

save simul_optim.mat times msetheta msebeta meanchecks meanms quantiles...
    tablemse tabletimes tablechecks