function [beta,theta,objf_min,b1,objf1,gridtheta2,b2,objf2] = qrs_fast(y,x,prop,w,Q1,Q2,P,family,gridtheta,m)

%Algorithm 3:
%Algorithm with preprocessing and quantile grid reduction for Quantile 
%Regression with Selection (QRS); propensity score estimated previously.
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
%Q1 = number of quantiles in reduced grid
%
%Q2 = number of quantiles in large grid (IMPORTANT: it is implicitly 
%assumed that the quantiles used in the reduced grid are also present in 
%the large grid; for example, Q1 = 9 and Q2 = 99, with resulting grids 
%{0.1,...,0.9} and {0.01,...,0.99})
%
%P = number of evaluated values of parameter with large quantile grid (by
%default, it is set to 10)
%
%family = copula family (Gaussian, Clayton, Gumbel, Frank)
%
%gridtheta = grid of values for copula parameter
%
%m = parameter to select interval of observations in top and bottom groups
%(by default, it is equal to 1)
%
%
%Output:
%
%beta = estimated beta parameters
%
%theta = estimated copula parameter
%
%objf_min = value of objective function at the optimum
%
%b1 = estimated beta parameters for grid of values for copula parameter 
%with reduced quantile grid
%
%objf1 = value of objective function for grid of values for copula parameter 
%with reduced quantile grid
%
%gridtheta2 = grid of values for copula parameter selected during the first
%part of the algorithm
%
%b2 = estimated beta parameters for grid of values for copula parameter 
%with large quantile grid
%
%objf2 = value of objective function for grid of values for copula parameter 
%with large quantile grid

[N,K]=size(x);

%Sample weights
if isempty(w)
    w=ones(N,1);
end

%Number of evaluated values of parameter with large quantile grid
if isempty(P)
    P=10;
end

%Weighted regressors
xw=x.*(w*ones(1,K));

%Conservative estimate of standard error
zeta=(xw*sqrt(diag(inv(xw'*xw/N))));
small=10^-6;
zeta=max(zeta,small);

%Quantile grids
gridq1=linspace(1/(Q1+1),Q1/(Q1+1),Q1);
gridq2=linspace(1/(Q2+1),Q2/(Q2+1),Q2);

%Instrument
phi=prop.*w;

%Prevent the conditional copula from being too close to 0 or 1
eps=.00001;

%Begin with central values of the theta parameter and quantile grids
%(comment the following two lines and uncomment the next two if user wants
%to begin from the first values of the theta parameter and the quantile
%grid)
initq=floor((Q1+1)/2);
initt=floor((length(gridtheta)+1)/2);

%Pregenerate matrices to store values of beta coefficients and criterion
%function
objf1=zeros(length(gridtheta),1);
b1=zeros(K,Q1,length(gridtheta));

%Estimation with reduced quantile grid
for i1=initt:1:length(gridtheta)
    t=gridtheta(i1);
    
    %Copula conditional on participation
    C=reshape(copulacdf(family,[kron(ones(N,1),gridq1'),kron(prop,ones(Q1,1))],t),Q1,N)';
    G=C./(prop*ones(1,Q1));
    G=max(min(G,1-eps),eps);
    
    %Slope parameters given copula
    if i1==initt %for the first value of the copula parameter, no preliminary estimate
        b1(:,:,i1)=rqr_fast(x,y,w,G,zeta,m,initq);
    elseif i1>1
        b1(:,:,i1)=rqrb0_fast(x,y,w,G,zeta,m,b1(:,:,i1-1));
    end
    
    %Objective function for copula parameter
    objf1(i1)=((phi'*(sum(double(y*ones(1,Q1)<=x*b1(:,:,i1))-G,2)))/N).^2;
end
for i1=1:1:initt-1
    t=gridtheta(initt-i1);
    
    %Copula conditional on participation
    C=reshape(copulacdf(family,[kron(ones(N,1),gridq1'),kron(prop,ones(Q1,1))],t),Q1,N)';
    G=C./(prop*ones(1,Q1));
    G=max(min(G,1-eps),eps);
    
    %Slope parameters given copula
    b1(:,:,initt-i1)=rqrb0_fast(x,y,w,G,zeta,m,b1(:,:,initt+1-i1));
    
    %Objective function for copula parameter
    objf1(initt-i1)=((phi'*(sum(double(y*ones(1,Q1)<=x*b1(:,:,initt-i1))-G,2)))/N).^2;
end

%Sort parameter values by objective function; select P candidate values for
%estimation with large quantile grid
[~,index]=sort(objf1,'ascend');
gridtheta2=gridtheta(index(1:P));

%Estimation with large quantile grid
if Q1<Q2
    objf2=zeros(P,1);
    b2=zeros(K,Q2,P);
    for i1=1:1:P
        t=gridtheta2(i1);
        
        %Copula conditional on participation
        C=reshape(copulacdf(family,[kron(ones(N,1),gridq2'),kron(prop,ones(Q2,1))],t),Q2,N)';
        G=C./(prop*ones(1,Q2));
        G=max(min(G,1-eps),eps);
        
        %Slope parameters given copula
        %Assign values already estimated
        for i2=1:1:Q1
            [~,minim]=min(abs(gridq1(i2)-gridq2));
            if i2==1
                initq2=minim;
            end
            b2(:,minim,i1)=b1(:,i2,index(i1));
        end
        
        %Estimate the rest using as preliminary estimate those with the
        %same copula parameter and a close quantile
        for i2=initq2:1:Q2
            if b2(:,i2,i1)==zeros(K,1)
                b2(:,i2,i1)=rqrtau_fast(x,y,w,G(:,i2),zeta,m,b2(:,i2-1,i1)');
            end
        end
        for i2=1:1:initq2-1
            if b2(:,initq2-i2,i1)==zeros(K,1)
                b2(:,initq2-i2,i1)=rqrtau_fast(x,y,w,G(:,initq2-i2),zeta,m,b2(:,initq2+1-i2,i1)');
            end
        end
        
        %Objective function for copula parameter
        objf2(i1)=((phi'*(sum(double(y*ones(1,Q2)<=x*b2(:,:,i1))-G,2)))/N).^2;
    end
else
    objf2=objf1(index(1:P));
    b2=b1(:,:,index(1:P));
end

%Find minimum of objective function
if P>1
    [objf_min,argminf]=min(objf2);
else
    objf_min=objf2;
    argminf=1;
end

%Optimum copula and beta parameters
theta=gridtheta2(argminf);
beta=reshape(b2(:,:,argminf),K,Q2);