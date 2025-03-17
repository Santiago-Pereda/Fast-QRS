function b = rqrtau_fast(x,y,w,tau,zeta,m,b0)

%Algorithm 1:
%Algorithm with preprocessing for Rotated Quantile Regression (RQR) with 
%initial values of the beta coefficients for a single quantile tau
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
%tau = copula conditional on participation at a single quantile
%
%zeta = conservative estimate of the standard error of the residuals
%
%m = parameter to select interval of observations in top and bottom groups
%(by default, it is equal to 1)
%
%b0 = initial value of beta parameters to obtain residuals
%
%
%Output:
%
%b = estimated beta parameters

warning('off')

[N,K]=size(x);

%Set some parameters
M=m*sqrt(N*K);
maxit=100;

%Weigh observations
xw=x.*(w*ones(1,K));
yw=y.*w;

%Calculate residuals
r=yw-xw*b0';

taumean=mean(tau);
for i1=1:1:maxit
    %Classify observations into three groups according to their residuals
    ub=prctile(r./zeta,100*min(taumean+M/2/N,(N-1)/N));
    lb=prctile(r./zeta,100*max(taumean-M/2/N,1/N));
    J=(r>lb*zeta & r<ub*zeta);
    JL=(r<lb*zeta);
    JH=(r>ub*zeta);
    
    %Preliminary regression
    b=rq([xw(J,:);sum((tau(JL)*ones(1,K)-1).*xw(JL,:)/taumean,1);sum((tau(JH)*ones(1,K)).*xw(JH,:)/taumean,1)],[yw(J);sum((tau(JL)-1).*yw(JL)/taumean);sum(tau(JH).*yw(JH)/taumean)],ones(sum(J)+2,1),[tau(J);taumean*ones(2,1)]);
    
    %Preliminary residuals
    rr=yw-xw*b;
    
    %Find residuals with mispredicted sign
    sum_mispred=sum(sign(r(JL))~=sign(rr(JL)))+sum(sign(r(JH))~=sign(rr(JH)));
    if sum_mispred==0%if all signs are well predicted, solution found
        break;
    elseif sum_mispred>=M/10%if many signs mispredicted, double M and repeat
        M=2*M;
        r=rr;
    else%if few signs mispredicted, take observations out from the L and G groups
        while sum_mispred>0
            %Update classification of observations
            J(JL)=(sign(r(JL))~=sign(rr(JL)));
            J(JH)=(sign(r(JH))~=sign(rr(JH)));
            JL=(JL-J>0);
            JH=(JH-J>0);
            
            %New regression (if either of the two groups has no 
            %observations, the input is appropriately modified to account 
            %for it)
            if sum(JH)==0
                b=rq([xw(J,:);sum((tau(JL)*ones(1,K)-1).*xw(JL,:)/taumean,1)],[yw(J);sum((tau(JL)-1).*y(JL)/taumean)],ones(sum(J)+1,1),[tau(J);taumean]);
            elseif sum(JL)==0
                b=rq([xw(J,:);sum((tau(JH)*ones(1,K)).*xw(JH,:)/taumean,1)],[yw(J);sum(tau(JH).*y(JH)/taumean)],ones(sum(J)+1,1),[tau(J);taumean]);
            elseif sum(JH)>0 && sum(JL)>0
                b=rq([xw(J,:);sum((tau(JL)*ones(1,K)-1).*xw(JL,:)/taumean,1);sum((tau(JH)*ones(1,K)).*xw(JH,:)/taumean,1)],[yw(J);sum((tau(JL)-1).*y(JL)/taumean);sum(tau(JH).*y(JH)/taumean)],ones(sum(J)+2,1),[tau(J);taumean*ones(2,1)]);
            end
            
            
            %New residuals
            rr=yw-xw*b;
            
            %Find residuals with mispredicted sign
            sum_mispred=sum(sign(r(JL))~=sign(rr(JL)))+sum(sign(r(JH))~=sign(rr(JH)));
            r=rr;
        end
        break;
    end
end