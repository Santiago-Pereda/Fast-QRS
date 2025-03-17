clear
clc

%Parameters for DGP
n=10000;
k=2;
rho0=.5;
family='Gaussian';
rng(151)

%Quantile grid and true slope parameters
Q=99;
gridq=linspace(1/(Q+1),Q/(Q+1),Q);
beta0=[norminv(gridq'),(gridq'*ones(1,k-1)).^(ones(Q,1)*linspace(1,k-1,k-1))]';

%Copula parameter grid
vecrhoa=linspace(-.90,.90,91);

%Generate sample
x=[ones(n,1),2+rand(n,k-1)];
z=[x,rand(n,1)];
uv=copularnd(family,rho0,n);
u=uv(:,1);
zeta=uv(:,2);

betar=1;
betau=[norminv(u),u.^(ones(n,1))*betar'];
gamma=[-1.5;.1*rand(k-1,1);2];

prop=exp(z*gamma)./(1+exp(z*gamma));
d=double(zeta<=prop);
y=d.*sum(x.*betau,2);

C=reshape(copulacdf(family,[kron(ones(n,1),gridq'),kron(prop,ones(Q,1))],rho0),Q,n)';
G=C./(prop*ones(1,Q));
G=max(min(G,.99999),.00001);

w=ones(n,1);

%Parameters and variables for fast QRS
m=1;
xw=x.*(w*ones(1,k));
zeta=(xw*sqrt(diag(inv(xw'*xw/N))));
small=10^-6;
zeta=max(zeta,small);

n1=sum(d);
bb1=zeros(k,Q,length(vecrhoa));
bb2=zeros(k,Q,length(vecrhoa));
bb3=zeros(k,Q,length(vecrhoa));
m1=zeros(length(vecrhoa),1);
m2=zeros(length(vecrhoa),1);
m3=zeros(length(vecrhoa),1);
m1_0=zeros(length(vecrhoa),Q);
m2_0=zeros(length(vecrhoa),Q);
m3_0=zeros(length(vecrhoa),Q);
for i1=1:1:length(vecrhoa)
    rho=vecrhoa(i1);
    C=reshape(copulacdf(family,[kron(ones(n1,1),gridq'),kron(prop(d==1),ones(Q,1))],rho),Q,n1)';
    G=C./(prop(d==1)*ones(1,Q));
    G=max(min(G,.99999),.00001);
    
    %First estimation: fast RQRb = rqr_fast(x,y,w,G,zeta,m,initq)
    bb1(:,:,i1)=rqr_fast(x(d==1,:),y(d==1),w(d==1),G,zeta(d==1),m,50);
    for i2=1:1:Q
        %Second estimation: standard
        bb2(:,i2,i1)=rq(x(d==1,:),y(d==1),w(d==1),G(:,i2));
        %Third estimation: standard with modified optimization parameters
        bb3(:,i2,i1)=rq_pen(x(d==1,:),y(d==1),G(:,i2));
    end
    
    %Objective functions
    m1(i1)=((prop(d==1)'*(sum(double(y(d==1)*ones(1,Q)<=x(d==1,:)*bb1(:,:,i1))-G,2)))/n1).^2;
    m2(i1)=((prop(d==1)'*(sum(double(y(d==1)*ones(1,Q)<=x(d==1,:)*bb2(:,:,i1))-G,2)))/n1).^2;
    m3(i1)=((prop(d==1)'*(sum(double(y(d==1)*ones(1,Q)<=x(d==1,:)*bb3(:,:,i1))-G,2)))/n1).^2;
    for i2=1:1:Q
        m1_0(i1,i2)=checkfn(y(d==1)-x(d==1,:)*bb1(:,i2,i1),G(:,i2));
        m2_0(i1,i2)=checkfn(y(d==1)-x(d==1,:)*bb2(:,i2,i1),G(:,i2));
        m3_0(i1,i2)=checkfn(y(d==1)-x(d==1,:)*bb3(:,i2,i1),G(:,i2));
    end
end

[~,min1]=min(m1);
[~,min2]=min(m2);
[~,min3]=min(m3);

theta1=vecrhoa(min1);
theta2=vecrhoa(min2);
theta3=vecrhoa(min3);

beta1(:,:,1)=bb1(:,:,min1);
beta2(:,:,1)=bb2(:,:,min2);
beta3(:,:,1)=bb3(:,:,min3);


%Collect results
eps=10^-3;
npercdifq=[sum(m2_0./m1_0>1+eps,1);sum(m3_0./m1_0>1+eps,1)];
npercdift=[sum(m2_0./m1_0>1+eps,2)';sum(m3_0./m1_0>1+eps,2)'];

save simul_precision.mat beta0 vecrhoa Q gridq m1 m2 m3 m1_0 m2_0 m3_0...
    bb1 bb2 bb3 npercdifq npercdift theta1 theta2 theta3 beta1 beta2...
    beta3

figure(1)
fig1=figure(1);
subplot(1,2,1)
plot(vecrhoa,npercdift(1,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(vecrhoa,npercdift(2,:),'LineWidth',1,'LineStyle','-');
hold on;
xlabel('$\theta$','interpreter','latex')
subplot(1,2,2)
plot(gridq,npercdifq(1,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,npercdifq(2,:),'LineWidth',1,'LineStyle','-');
hold on;
xlabel('$\tau$','interpreter','latex')
set(fig1, 'PaperPosition', [1 1 18 7])
print -depsc2 discrep.eps

figure(2)
fig2=figure(2);
subplot(3,3,1)
plot(gridq,m2_0(1,:)./m1_0(1,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(1,:)./m1_0(1,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=-0.9$', 'interpreter', 'latex')
subplot(3,3,2)
plot(gridq,m2_0(6,:)./m1_0(6,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(6,:)./m1_0(6,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=-0.8$', 'interpreter', 'latex')
subplot(3,3,3)
plot(gridq,m2_0(11,:)./m1_0(11,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(11,:)./m1_0(11,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=-0.7$', 'interpreter', 'latex')
subplot(3,3,4)
plot(gridq,m2_0(26,:)./m1_0(26,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(26,:)./m1_0(26,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=-0.4$', 'interpreter', 'latex')
subplot(3,3,5)
plot(gridq,m2_0(46,:)./m1_0(46,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(46,:)./m1_0(46,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=0$', 'interpreter', 'latex')
subplot(3,3,6)
plot(gridq,m2_0(76,:)./m1_0(76,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(76,:)./m1_0(76,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=0.4$', 'interpreter', 'latex')
subplot(3,3,7)
plot(gridq,m2_0(81,:)./m1_0(81,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(81,:)./m1_0(81,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=0.7$', 'interpreter', 'latex')
subplot(3,3,8)
plot(gridq,m2_0(86,:)./m1_0(86,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(86,:)./m1_0(86,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=0.8$', 'interpreter', 'latex')
subplot(3,3,9)
plot(gridq,m2_0(91,:)./m1_0(91,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,m3_0(91,:)./m1_0(91,:),'LineWidth',1,'LineStyle','-');
hold on;
ylim([.95,1.2])
title('$\theta=0.9$', 'interpreter', 'latex')
set(fig2, 'PaperPosition', [1 1 18 17])
print -depsc2 difcheckfn_3.eps

figure(3)
fig3=figure(3);
plot(vecrhoa,m2,'LineWidth',1,'LineStyle',':');
hold on;
plot(vecrhoa,m3,'LineWidth',1,'LineStyle','-');
hold on;
plot(vecrhoa,m1,'LineWidth',1,'Color',[0 0.5 0],'LineStyle','--');
hold on;
xlabel('$\theta$','interpreter','latex')
set(fig3, 'PaperPosition', [1 1 18 7])
print -depsc2 objfn.eps

figure(4)
fig4=figure(4);
subplot(1,2,1)
plot(gridq,beta2(1,:)-beta0(1,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,beta3(1,:)-beta0(1,:),'LineWidth',1,'LineStyle','-');
hold on;
plot(gridq,beta1(1,:)-beta0(1,:),'LineWidth',1,'Color',[0 0.5 0],'LineStyle','--');
hold on;
xlabel('$\tau$','interpreter','latex')
title('Intercept')
subplot(1,2,2)
plot(gridq,beta2(2,:)-beta0(2,:),'LineWidth',1,'LineStyle',':');
hold on;
plot(gridq,beta3(2,:)-beta0(2,:),'LineWidth',1,'LineStyle','-');
hold on;
plot(gridq,beta1(2,:)-beta0(2,:),'LineWidth',1,'Color',[0 0.5 0],'LineStyle','--');
hold on;
xlabel('$\tau$','interpreter','latex')
title('Slope')
set(fig4, 'PaperPosition', [1 1 18 7])
print -depsc2 betabias.eps