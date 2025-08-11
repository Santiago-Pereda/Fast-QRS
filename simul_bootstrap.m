clear
clc

%Number of bootstrap repetitions
reps1=200;
reps2=200;

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

%Number of evaluated values of parameter with large quantile grid
P=10;

gampar=10;

times=zeros(nexp,3,reps1);
thetad=zeros(nexp,3,reps1);
thetarej=zeros(nexp,3,3,reps1);
thetalength=zeros(nexp,3,3,reps1);
for i1=1:1:nexp
    %Select parameters that vary across experiments
    N=params_exp(i1,1);
    k=params_exp(i1,2);
    
    parfor i2=1:reps1
        rng(i2)
        
        times_b=zeros(3,1);
        thetad_b=zeros(3,1);
        thetarej_b=zeros(3,3);
        thetalength_b=zeros(3,3);

        %Generate sample
        betar=rand(k-1,1);
        beta0=[norminv(gridq);betar*gridq];
        
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
        
        %Compute estimates with Algorithm 3
        [betahat1,thetahat1,~,bhat1]=qrs_fast(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q1,Q2,1,family,gridtheta,m);
        [betahat2,thetahat2,~,bhat2]=qrs_fast(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,Q2,1,family,gridtheta,m);
        
        %First bootstrap estimates: Algorithm 4 with preprocessing and no
        %grid reduction, no smaller grid
        start1=tic;
        [b_1,theta_1,m_1]=qrs_fast_bt(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q2,Q2,1,family,gridtheta,m,bhat2,reps2,gampar);
        times_b(1,1)=toc(start1);
        thetad_b(1,1)=mean(theta_1)-theta0;
        theta_q=[prctile(theta_1,5),prctile(theta_1,95);
            prctile(theta_1,2.5),prctile(theta_1,97.5);
            prctile(theta_1,0.5),prctile(theta_1,99.5)];
        for i3=1:1:3
            if theta_q(i3,1)>theta0 || theta0>theta_q(i3,2)
                thetarej_b(1,i3)=1;
            end
        end
        thetalength_b(1,:)=theta_q(:,2)-theta_q(:,1);
        disp(['repetition: ' num2str(i2) , 'first bootstrap']);
        
        %Second boostrap estimates: Algorithm 4 with preprocessing and grid
        %reduction, only one candidate selected with smaller quantile grid
        start2=tic;
        [b_2,theta_2,m_2]=qrs_fast_bt(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q1,Q2,1,family,gridtheta,m,bhat1,reps2,gampar);
        times_b(2,1)=toc(start2);
        thetad_b(2,1)=mean(theta_2)-theta0;
        theta_q=[prctile(theta_2,5),prctile(theta_2,95);
            prctile(theta_2,2.5),prctile(theta_2,97.5);
            prctile(theta_2,0.5),prctile(theta_2,99.5)];
        for i3=1:1:3
            if theta_q(i3,1)>theta0 || theta0>theta_q(i3,2)
                thetarej_b(2,i3)=1;
            end
        end
        thetalength_b(2,:)=theta_q(:,2)-theta_q(:,1);
        disp(['repetition: ' num2str(i2) , 'second bootstrap']);
        
        %Third boostrap estimates: Algorithm 4 with preprocessing and grid
        %reduction, several candidate selected with smaller quantile grid
        start3=tic;
        [b_3,theta_3,m_3]=qrs_fast_bt(y(d==1,:),x(d==1,:),prop(d==1,:),w(d==1,:),Q1,Q2,P,family,gridtheta,m,bhat1,reps2,gampar);
        times_b(3,1)=toc(start3);
        thetad_b(3,1)=mean(theta_3)-theta0;
        theta_q=[prctile(theta_3,5),prctile(theta_3,95);
            prctile(theta_3,2.5),prctile(theta_3,97.5);
            prctile(theta_3,0.5),prctile(theta_3,99.5)];
        for i3=1:1:3
            if theta_q(i3,1)>theta0 || theta0>theta_q(i3,2)
                thetarej_b(3,i3)=1;
            end
        end
        thetalength_b(3,:)=theta_q(:,2)-theta_q(:,1);
        
        times(i1,:,i2)=times_b;
        thetad(i1,:,i2)=thetad_b;
        thetarej(i1,:,:,i2)=thetarej_b;
        thetalength(i1,:,:,i2)=thetalength_b;
        disp(['repetition: ' num2str(i2) ]);
    end
    disp(['experiment: ' num2str(i1) ]);
    save simul_bt_partial.mat times thetad thetarej thetalength
end

%Collect estimates
msetheta=mean(thetad.^2,3);

thetarr=mean(thetarej,4);

thetal=mean(thetalength,4);

mtimes=mean(times,3);

save simul_bt.mat times thetad thetarej thetalength msetheta thetarr...
    thetal mtimes