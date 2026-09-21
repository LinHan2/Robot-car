%% 2-D bearing-only target tracking: direct closed-loop tracking
% Paper correspondence: (2)-(24), (46), (50)-(51), and (63).
% Both robots are fully actuated: pT_dot=uT, pO_dot=uO in R^2.

clear; clc; close all;

%% Parameters
p.n=2; p.I=eye(2); p.J=[0 -1;1 0];
p.dt=0.01; p.T=200; t=0:p.dt:p.T; Nt=numel(t);
p.field=[-1.5 1.5 -2 2];

% Desired relative trajectory r_d(t) and tracking gain in (63)
p.Rd=0.7; p.wd=0.3; p.phi0=2*pi/3;
p.Kr=0.25*p.I;

% RBF model in (4)
p.q=30; p.nq=p.n*p.q; p.sigma=0.52; p.Inq=eye(p.nq);
a=2*pi*(0:p.q-1)/p.q;
p.centers=[cos(a);sin(a)];

% Current/history-window identifier in (11)-(24)
p.DeltaT=1; p.Nw=round(p.DeltaT/p.dt); p.Nh=20;
assert(abs(p.Nw*p.dt-p.DeltaT)<10*eps,'DeltaT/dt must be an integer.');
p.gamma=1e-7; p.ac=1e-5; p.validTol=100*eps(p.DeltaT);
p.kc=2.0; p.kh=1.2; p.sigmaTheta=0.003;

% Filter (46), Riccati equation (50), and observer (51)
p.af=5; p.Ao=[zeros(2) p.I;zeros(2) zeros(2)];
p.lambdaS=0.30; p.RoInv=inv(0.02*p.I); p.Pfloor=1e-9;

% Unknown target law: used only by the simulated target plant
p.d=[cos(155*pi/180);sin(155*pi/180)];
p.vd=0; p.ve=0.1; p.vj=0.1;

%% Initial conditions: all estimates start from zero
pT=[0;0]; pO=[0;-1.5];
thetaHat=zeros(p.nq,1); uTHat=zeros(2,1); xiHat=zeros(4,1);
Po=0.30*eye(4);
x=packState(pT,pO,thetaHat,uTHat,xiHat,Po);

% Retained history and current partial window
S.W=zeros(p.nq,p.nq,p.Nh); S.Z=zeros(p.nq,p.Nh); S.count=0;
S.Pi=zeros(p.nq); S.thetaGamma=zeros(p.nq,1);
s=signals(0,x,p); win=resetWindow(s.b,p);

% Logs
pTLog=zeros(2,Nt); pOLog=zeros(2,Nt);
uTLog=zeros(2,Nt); uOLog=zeros(2,Nt);
eU=zeros(1,Nt); ePos=zeros(1,Nt); eTrack=zeros(1,Nt);

%% Simulation
for k=1:Nt
    tk=t(k);
    [Wc,Zc,chi]=windowLRE(win,p.ac,p);              % (18)-(20)
    data=struct('W',Wc,'Z',Zc,'chi',chi, ...
        'Pi',S.Pi,'thetaGamma',S.thetaGamma);

    [~,s]=closedLoopRHS(tk,x,p,data);
    [pT,pO,~,uTHat,xiHat,~]=unpackState(x,p);
    uT=targetInput(s.b,p);                          % plant/error log only

    pTLog(:,k)=pT; pOLog(:,k)=pO;
    uTLog(:,k)=uT; uOLog(:,k)=s.uO;
    eU(k)=norm(uT-uTHat);
    ePos(k)=norm(s.r-xiHat(1:2));
    eTrack(k)=norm(s.r-s.rd);

    if k==Nt, break; end

    % Left-rectangle integrals for M_c,k, W_c,k, and Z_c,k.
    F=s.Pb*win.G; H=s.Pb*win.U; v=s.Pb*win.b0;
    win.a=win.a+p.dt*(v'*v);
    win.CF=win.CF+p.dt*(v'*F); win.CH=win.CH+p.dt*(v'*H);
    win.SFF=win.SFF+p.dt*(F'*F); win.SFH=win.SFH+p.dt*(F'*H);
    win.G=win.G+p.dt*s.Phi; win.U=win.U+p.dt*s.uO;
    win.count=win.count+1;

    % RK4 integration of (2), (24), (46), and (50)-(51).
    k1=closedLoopRHS(tk,x,p,data);
    k2=closedLoopRHS(tk+p.dt/2,x+p.dt*k1/2,p,data);
    k3=closedLoopRHS(tk+p.dt/2,x+p.dt*k2/2,p,data);
    k4=closedLoopRHS(tk+p.dt,x+p.dt*k3,p,data);
    x=x+p.dt*(k1+2*k2+2*k3+k4)/6;

    [pT,pO,thetaHat,uTHat,xiHat,Po]=unpackState(x,p);
    x=packState(pT,pO,thetaHat,uTHat,xiHat,projectSPD(Po,p.Pfloor));

    % At the window endpoint, finalize (W_k,Z_k) and update S_h.
    if win.count==p.Nw
        if win.a>p.validTol
            [Wk,Zk]=windowLRE(win,p.validTol,p);
            S=updateHistory(S,Wk,Zk,p);
        end
        sn=signals(t(k+1),x,p); win=resetWindow(sn.b,p);
    end
end

fprintf('Stored windows: %d/%d; active directions: %d/%d.\n', ...
    S.count,p.Nh,rank(S.Pi,1e-8),p.nq);
fprintf('Final input/localization/tracking errors: %.4f, %.4f, %.4f.\n', ...
    eU(end),ePos(end),eTrack(end));
fprintf('Maximum target/pursuer speeds: %.4f, %.4f m/s.\n', ...
    max(vecnorm(uTLog)),max(vecnorm(uOLog)));

%% Figures
figure;
plot(pTLog(1,:),pTLog(2,:),'LineWidth',1.5); hold on;
plot(pOLog(1,:),pOLog(2,:),'LineWidth',1.5);
axis equal; xlim(p.field(1:2)); ylim(p.field(3:4)); grid on;
xlabel('x (m)'); ylabel('y (m)');
legend('Target','Pursuer','Location','best');

figure; plot(t,eU,'LineWidth',1.5); grid on;
xlabel('t (s)'); ylabel('||u_T-u_T_hat||_2 (m/s)'); linearTicks(eU);

figure; plot(t,ePos,'LineWidth',1.5); grid on;
xlabel('t (s)'); ylabel('||r-r_hat||_2 (m)'); linearTicks(ePos);

figure; plot(t,eTrack,'LineWidth',1.5); grid on;
xlabel('t (s)'); ylabel('||r-r_d||_2 (m)'); linearTicks(eTrack);

speedT=vecnorm(uTLog); speedO=vecnorm(uOLog);
figure; plot(t,speedT,'LineWidth',1.5); hold on;
plot(t,speedO,'LineWidth',1.5); grid on;
xlabel('t (s)'); ylabel('Speed (m/s)');
legend('||u_T||_2','||u_O||_2','Location','best');
linearTicks([speedT speedO]);

%% Local functions
function [dx,s]=closedLoopRHS(t,x,p,data)
[~,~,thetaHat,uTHat,xiHat,Po]=unpackState(x,p);
Po=(Po+Po')/2; s=signals(t,x,p);

% Parameter estimator (24) and target-input filter (46)
thetaDot=p.kc*data.chi*(data.Z-data.W*thetaHat) ...
    +p.kh*(data.thetaGamma-data.Pi*thetaHat) ...
    -p.sigmaTheta*(p.Inq-data.Pi)*thetaHat;
uTHatDot=p.af*(s.Phi*thetaHat-uTHat);

% Riccati gain (50) and relative-localization estimator (51)
Co=[s.Pb zeros(2)]; Lo=Po*Co'*p.RoInv;
go=[uTHat-s.uO;zeros(2,1)];                       % (49)
PoDot=p.Ao*Po+Po*p.Ao'+p.lambdaS*Po-Po*Co'*p.RoInv*Co*Po;
xiHatDot=p.Ao*xiHat+go+Lo*(-Co*xiHat);            % y_o=0

dx=packState(targetInput(s.b,p),s.uO,thetaDot,uTHatDot,xiHatDot,PoDot);
end

function s=signals(t,x,p)
[pT,pO,~,uTHat,xiHat,~]=unpackState(x,p);
s.r=pT-pO; rho=norm(s.r);
assert(rho>1e-12,'Bearing is undefined because ||r||=0.');
s.b=s.r/rho; s.Pb=p.I-s.b*s.b';
s.Phi=rbfRegressor(s.b,p);
[s.rd,rdDot]=desiredTrajectory(t,p);

% Direct tracking from t=0: equation (63)
s.uO=uTHat+xiHat(3:4)-rdDot+p.Kr*(xiHat(1:2)-s.rd);
end

function Phi=rbfRegressor(b,p)
e=p.centers-b;
phi=exp(-sum(e.^2,1)'/(2*p.sigma^2));
Phi=kron(p.I,phi');                               % (4)
end

function uT=targetInput(b,p)
uT=p.vd*p.d+p.ve*b+p.vj*p.J*b;
end

function [rd,rdDot]=desiredTrajectory(t,p)
a=p.phi0+p.wd*t;
rd=p.Rd*[cos(a);sin(a)];
rdDot=p.Rd*p.wd*[-sin(a);cos(a)];
end

function win=resetWindow(b0,p)
win.b0=b0; win.G=zeros(2,p.nq); win.U=zeros(2,1);
win.a=0; win.CF=zeros(1,p.nq); win.CH=0;
win.SFF=zeros(p.nq); win.SFH=zeros(p.nq,1); win.count=0;
end

function [W,Z,chi]=windowLRE(win,threshold,p)
if win.a>=threshold
    W=win.SFF-win.CF'*win.CF/win.a; W=(W+W')/2;
    Z=win.SFH-win.CF'*win.CH/win.a; chi=1;
else
    W=zeros(p.nq); Z=zeros(p.nq,1); chi=0;
end
end

function S=updateHistory(S,Wnew,Znew,p)
% History-stack replacement rule preceding (21)
if S.count<p.Nh
    S.count=S.count+1;
    S.W(:,:,S.count)=Wnew; S.Z(:,S.count)=Znew;
else
    Wh=sum(S.W,3); best=spectrum(Wh); replace=0;
    for j=1:p.Nh
        candidate=spectrum(Wh-S.W(:,:,j)+Wnew);
        if lexGreater(candidate,best)
            best=candidate; replace=j;
        end
    end
    if replace>0
        S.W(:,:,replace)=Wnew; S.Z(:,replace)=Znew;
    end
end

% Spectral projection and truncated inverse in (21)-(23)
Wh=sum(S.W(:,:,1:S.count),3); Wh=(Wh+Wh')/2;
Zh=sum(S.Z(:,1:S.count),2);
[Q,D]=eig(Wh); lambda=real(diag(D)); active=lambda>=p.gamma;
if any(active)
    Qa=Q(:,active); la=lambda(active);
    S.Pi=Qa*Qa'; S.Pi=(S.Pi+S.Pi')/2;
    S.thetaGamma=Qa*((Qa'*Zh)./la);
else
    S.Pi=zeros(p.nq); S.thetaGamma=zeros(p.nq,1);
end
end

function lambda=spectrum(W)
lambda=sort(real(eig((W+W')/2)),'ascend');
tol=100*eps(max(1,norm(W,2))); lambda(abs(lambda)<tol)=0;
end

function tf=lexGreater(a,b)
tf=false; tol=1e-12;
for i=1:numel(a)
    scale=max([1,abs(a(i)),abs(b(i))]);
    if a(i)>b(i)+tol*scale, tf=true; return; end
    if a(i)<b(i)-tol*scale, return; end
end
end

function x=packState(pT,pO,thetaHat,uTHat,xiHat,Po)
x=[pT(:);pO(:);thetaHat(:);uTHat(:);xiHat(:);Po(:)];
end

function [pT,pO,thetaHat,uTHat,xiHat,Po]=unpackState(x,p)
i=0; pT=x(i+(1:2)); i=i+2; pO=x(i+(1:2)); i=i+2;
thetaHat=x(i+(1:p.nq)); i=i+p.nq;
uTHat=x(i+(1:2)); i=i+2; xiHat=x(i+(1:4)); i=i+4;
Po=reshape(x(i+(1:16)),4,4);
end

function A=projectSPD(A,floorValue)
A=(A+A')/2; [Q,D]=eig(A);
A=Q*diag(max(real(diag(D)),floorValue))*Q'; A=(A+A')/2;
end

function linearTicks(y)
ymax=max(y(:));
if ~isfinite(ymax)||ymax<=0, ymax=1; end
digits=max(0,-floor(log10(ymax/5))+1);
step=ceil((ymax/5)*10^digits)/10^digits;
ylim([0 5*step]); yticks(0:step:5*step);
ax=gca; ax.YAxis.Exponent=0; ytickformat(sprintf('%%.%df',digits));
end
