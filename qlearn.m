q=.1;ro=12;c0=5000;
maxit=5;%%no of episodes
ko=.03;epsln=.006; %nominal k and kstep epsln
v0=(ro/q);   %velocity
alpha=1; gama=.8;      % alpha & gamma
tstep=0.001;
Lstep=.75;                 %%learning step
lampda=10;
N_t=Lstep/tstep;
%N_l=time/Lstep;         % number of learning step
%N=time/tstep;           % total number of time step
%rng('shuffle');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rw=1/(c0*lampda);                     %%reward factor
epsilon =.1;              %%epsilon greedy policy
d=1;imat=zeros(1,1);      %%used in epsisode learning
r0 = [30;-30].*ro;                 %%starting postion 
T0 = [1;0];                       
n0 = [0;1];
%ic = [r0;T0;n0];                  %%initial conditions
ep=1;
cdismat=zeros(1);
rox=r0(1,:);                     %starting point coordinates
roy=r0(2,:);
omq=cell(1,1);
%%%%%%%%%%%%%%%%%%%%%%
kmin=-1;kmax=1;             %%boundry limits for k
kstp=epsln;                      
krange=[kmin:kstp:kmax];      %%expand the range of k
Nstat=ceil((kmax-kmin)/kstp); %%no of states (k states only)
qst=2*(Nstat+1);              %% no of stetes(k states)*no of stestes(sign of concentration states)
qmat= rand(qst,3);            %%create q matrix contains qs no of states and 3 actions.
%%%%%%%%%%%%%%%%%%%%%%%%
arandmat=zeros(1);        %%create storage matrix for actions.    
skmat=zeros(1);           %%create storage matrix for states.
kmat= zeros(10);          %% stored values of k materix (actions.
sdeltamat=zeros(10,1);    %% stored values of difference in stimulus between two states.
rx1mat=zeros(10000,1);    %% stored values of x values of r (trajectory) ////
ry1mat=zeros(10000,1);    %% stored values of x values of r (trajectory) ////
Rewardmat=zeros(10,1);    %% stored values of reward values  ////
%%%%%%%%%%%%%%%%%%%%%%%%         
epsilon_k=0.005;
h=tstep;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tstart=0;                    %% starting time of first step(state).
i=1;                         %% counter i
tend=tstart+Lstep;       %% ending time of first step.
odis=norm(r0);               %% current distance between starting point and final target (0,0).
osignal=lampda.*(c0./norm(r0))+sqrt(lampda.*(c0./norm(r0)))*randn ;    %% current signal. 
oss=osignal;
vg=1;                         %% counter h
rmat(1,:)=r0;
Tmat(1,:)=T0;
nmat(1,:)=n0;
kmat(1,:)=ko;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kn=ko;                       %%initial ko for first state.
tspan=[tstart tend];         %%time interval for first state

r = zeros(N_t,2);
T = zeros(N_t,2);
n = zeros(N_t,2);
% kn= zeros(N+1,1);

r(1,:)=r0;
T(1,:)=T0;
n(1,:)=n0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmat(1,:)=r0;
Tmat(1,:)=T0;
nmat(1,:)=n0;

for j=1:1*N_t       
        count=j;       
        r_1=r(count,:);
        T_1=T(count,:);
        n_1=n(count,:);
        noise=epsilon_k*randn;
        kappa=kn+noise;
        
        K1_r = v0*T_1;
        K1_T = v0*(kappa).*n_1;
        K1_n =(-v0*(kappa).*T_1);
        r_2=r(j,:)+h/2*K1_r;
        T_2=T(j,:)+h/2*K1_T;
        n_2=n(j,:)+h/2*K1_n;
        
        K2_r = v0*T_2;
        K2_T = v0*(kappa).*n_2;
        K2_n =(-v0*(kappa).*T_2);
        r_3=r(j,:)+h/2*K2_r;
        T_3=T(j,:)+h/2*K2_T;
        n_3=n(j,:)+h/2*K2_n;
        
        K3_r = v0*T_3;
        K3_T = v0*(kappa).*n_3;
        K3_n =(-v0*(kappa).*T_3);
        r_4=r(j,:)+h*K3_r;
        T_4=T(j,:)+h*K3_T;
        n_4=n(j,:)+h*K3_n;
        
        K4_r = v0*T_4;
        K4_T = v0*(kappa).*n_4;
        K4_n =(-v0*(kappa).*T_4);
        r(j+1,:)=r(j,:)+h/6*(K1_r+2*K2_r+2*K3_r+K4_r);
        T(j+1,:)=T(j,:)+h/6*(K1_T+2*K2_T+2*K3_T+K4_T);
        n(j+1,:)=n(j,:)+h/6*(K1_n+2*K2_n+2*K3_n+K4_n);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tstart=tend;                            
tend=tstart+Lstep;  
x1 = r(:,[1]);y1 = r(:,[2]);        %% x&y values of trajectory r at single step
x2 = T(:,[1]);y2 = T(:,[2]);         %% x&y values of directional velocity T at single step
x3 = n(:,[1]);y3 = n(:,[2]);
rx=x1(end);ry=y1(end);dis=[rx  ry];cdis=norm(dis);  %% x&y values of trajectory r 
%at the end of single step and calculating distance to target after single
%step
Tx=x2(end);Ty=y2(end);   %% x&y values of trajectory T,n at the end of single step.
nx=x3(end);ny=y3(end);
%%assigning x&y values of the end of final step as initial condtions
%to the start of the next step (r0,t,n).
r0 = [rx;ry];           %%assigning x&y values of the end of final step as initial condtions to nxt step. 
T0 = [Tx;Ty];                       
n0 = [nx;ny];
fsignal=lampda.*(c0./norm(dis))+sqrt(lampda.*(c0./norm(dis)))*randn;    
Reward=rw*(1/osignal-1/fsignal) ;          %%calculate reward as a difference between two points
Rewardmat(i,ep)=Reward;       %% storing reward values in reward matrix
odis=cdis;                       %% assigning old distance as current distance.      % sitmulus at current postion.
sdelta=fsignal-osignal;           % difference in sitmulus between current postion and old postion.
sdeltamat(i,ep)=sign(sdelta);     %% storing sign of s delta difference.
osignal=fsignal;                  %% assigning current signal as old signal.
kmat(i,ep)=kn;                    %% storing kn values

for u=1:N_t
rx1mat(vg,1)=x1(u,1);                   
ry1mat(vg,1)=y1(u,1);
vg=vg+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if sign(sdelta)==1       %%dterminig sign of sdelta(stimulus difference between 2 states)
    cdel=1;
elseif sign(sdelta)==-1
    cdel=2;
else
    disp error2
end
[~,sk]=min(abs(krange-kn));  %%locating which state no belongs kn.
skmat(i,:)=sk;
cstate=2*(sk-1)+cdel;        %%dterming current states postion in q matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qcs=qmat(cstate,:); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tstart;                             %% starting time.
i=2;                                  %% counter i
tend;                                 %% ending time 
%odis=norm(r0);                        %% current distance between starting point and final target (0,0).
%osignal=(c0./norm(r0));               %% current signal between starting point and final target (0,0).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=tstep;
tic
i=2;
r = zeros(N_t,2);
T = zeros(N_t,2);
n = zeros(N_t,2);
r(1,:)=r0;
T(1,:)=T0;
n(1,:)=n0;

%rmat = zeros(1);
%Tmat = zeros(1);
%nmat = zeros(1;
%kmat= zeros(1)
while(1)
%     kn=kmat(m,:);


tspan=[tstart tend];                  %% span time
qcs=qmat(cstate,:);                 %% recall q value from q table at given state
pp=rand; % get 1 uniform random number
% choose either explore or exploit
if pp < epsilon   % explore
umin=datasample(qcs,1); % choose 1 action randomly (uniform random distribution)
[~,arand]=min(abs(qcs-umin));
else        % exploit
[~,umax]=max(qcs);
arand=umax;
end
   arandmat(i,:)=arand;
if arand==1
    kn=ko+epsln;                    %% first column action.
elseif arand==2
    kn=ko;                          %% second column action.
elseif arand==3
    kn=ko-epsln;                    %% third coulmn action.
else
    disp error1                      %% show error1.
end

    for j=1:N_t        
        count=j;
        
        r_1=r(count,:);
        T_1=T(count,:);
        n_1=n(count,:);
        noise=epsilon_k*randn;
        kappa=kn+noise;
        
        K1_r = v0*T_1;
        K1_T = v0*(kappa).*n_1;
        K1_n =(-v0*(kappa).*T_1);
        r_2=r(j,:)+h/2*K1_r;
        T_2=T(j,:)+h/2*K1_T;
        n_2=n(j,:)+h/2*K1_n;
        
        K2_r = v0*T_2;
        K2_T = v0*(kappa).*n_2;
        K2_n =(-v0*(kappa).*T_2);
        r_3=r(j,:)+h/2*K2_r;
        T_3=T(j,:)+h/2*K2_T;
        n_3=n(j,:)+h/2*K2_n;
        
        K3_r = v0*T_3;
        K3_T = v0*(kappa).*n_3;
        K3_n =(-v0*(kappa).*T_3);
        r_4=r(j,:)+h*K3_r;
        T_4=T(j,:)+h*K3_T;
        n_4=n(j,:)+h*K3_n;
        
        K4_r = v0*T_4;
        K4_T = v0*(kappa).*n_4;
        K4_n =(-v0*(kappa).*T_4);
        r(j+1,:)=r(j,:)+h/6*(K1_r+2*K2_r+2*K3_r+K4_r);
        T(j+1,:)=T(j,:)+h/6*(K1_T+2*K2_T+2*K3_T+K4_T);
        n(j+1,:)=n(j,:)+h/6*(K1_n+2*K2_n+2*K3_n+K4_n);
    end
    
    tstart=tend;                            
    tend=tstart+Lstep;  
    x1 = r(:,[1]);y1 = r(:,[2]);%% x&y values of trajectory r at single step
    x2 = T(:,[1]);y2 = T(:,[2]);         %% x&y values of directional velocity T at single step
    x3 = n(:,[1]);y3 = n(:,[2]);
    rx=x1(end);ry=y1(end);
    Tx=x2(end);Ty=y2(end);
    nx=x3(end);ny=y3(end);
    r0 = [rx;ry];           
    T0 = [Tx;Ty];                       
    n0 = [nx;ny];
    r(1,:)=r0;
    T(1,:)=T0;
    n(1,:)=n0;
    dis=[rx  ry];
    cdis=norm(dis);
    fsignal=lampda.*(c0./norm(dis))+sqrt(lampda.*(c0./norm(dis)))*randn;         % sitmulus at current postion.
    sdelta=fsignal-osignal;           % difference in sitmulus between current postion and old postion.
    Reward=rw*(1/osignal-1/fsignal);  %%calculate reward as a difference between two points
    sdeltamat(i)=sign(sdelta);        %% storing sign of s delta difference.
    osignal=fsignal;                  %% assigning current signal as old signal.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   kmat(i,ep)=kn;                     %% storing kn value
u=1;
for u=1:N_t
rx1mat(vg,1)=x1(u,1);                   
ry1mat(vg,1)=y1(u,1);
vg=vg+1;
end
%%%%%%%
if sign(sdelta)==1
    cdel=1;
elseif sign(sdelta)==-1
    cdel=2;
else
    disp error2
end
[~,sk]=min(abs(krange-kn));
skmat(i,:)=sk;
fstate=2*(sk-1)+cdel;
qmat(cstate,arand)=(qmat(cstate,arand))+alpha.*(Reward+gama.*(max(qmat(fstate,:))-qmat(cstate,arand))); %%updating current q values
cstate=fstate; %assigning future states as current state

if abs(kn)>kmax
    disp ('wrong interval K')
    break 
end
cdismat(i,:)=cdis;
i=i+1;         %counter updated
ko=kn;         %assigning k new as k old
if cdis<50
 break         % break loop if distance to target is less than 1
end
if i>15000       % break loop if iterations more than given value.
    break
end
end

for ep=1:1
plot(rx1mat(:,ep),ry1mat(:,ep))
hold on;plot(0,0,'r*');plot(rox,roy,'g*');hold off
axis equal
title('radial concentration field trajectory r(t)')
xlabel('x(t)') 
ylabel('y(t)')
end
timespent=toc
i
imat(d,ep)=i;
omq{1,1}=qmat;
%figure
%plot(kmat(:,1))
%title('K-curveture Plot')
%xlabel('step no') 
%ylabel('K-curveture')
