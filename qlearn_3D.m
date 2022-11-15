clc;
q=.1;ro=12;c0=10;
maxit=5;%%no of episodes
ko=.03;epsln=.001;two=.006; epslntw=.004; %nominal k and kstep epslncv
v0=(ro/q);   %velocity
alpha=1; gama=.8;      % alpha & gamma
tstep=0.005;
Lstep=.5;                 %%learning step
N_t=Lstep/tstep;
rw=c0;                     %%reward factor
epsilon =.1;              %%epsilon greedy policy
d=1;imat=zeros(1,1);      %%used in epsisode learning
r0 = [30;-30;0].*ro;                 %%starting postion 
T0 = [1;0;0];                       
n0 = [0;1;0];
b0 = [0;0;1];
rox=r0(1,:);                     %starting point coordinates
roy=r0(2,:);
roz=r0(3,:);
ic = [r0;T0;n0;b0];                  %%initial conditions
ep=1;
omq=cell(50,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kmin=-1;kmax=1;             %%boundry limits for k
twmin=-1;twmax=1;
kstp=epsln;
twstp=epslntw;
krange=[kmin:kstp:kmax];      %%expand the range of k
twrange=[twmin:twstp:twmax];
Nstatk=length(krange) %%no of states (k states only)
Nstattw=length(twrange)
qst=2*(Nstatk)*(Nstattw)              %% no of stetes(k states)*no of stestes(sign of concentration states)
qmat= rand(qst,3);            %%create q matrix contains qs no of states and 3 actions.
%%%%%%%%%%%%%%%%%%%%%%
arandmat=zeros(1);        %%create storage matrix for actions.    
skmat=zeros(1);           %%create storage matrix for states.
stwmat=zeros(1);
cdismat=zeros(1);
kmat= zeros(10);          %% stored values of k materix (actions.
twmat=zeros(10);
sdeltamat=zeros(10,1);    %% stored values of difference in stimulus between two states.
rx1mat=zeros(10,1);    %% stored values of x values of r (trajectory) ////
ry1mat=zeros(10,1);    %% stored values of x values of r (trajectory) ////
rz1mat=zeros(10,1);
fstatemat=zeros(1);
normat=zeros(1);
Rewardmat=zeros(10,1);    %% stored values of reward values  ////
leo=0;
%%%%%%%%%%%%%%%%%%%%%%%%       
h=tstep;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tstart=0;                    %% starting time of first step(state).
i=1;                         %% counter i
tend=tstart+Lstep;           %% ending time of first step.
odis=norm(r0);               %% current distance between starting point and final target (0,0).
osignal=(c0./norm(r0));      %% current signal. 
oss=osignal;
vg=1;                         %% counter h
rmat(1,:)=r0;
Tmat(1,:)=T0;
nmat(1,:)=n0;
bmat(1,:)=b0;
kmat(1,:)=ko;
twmat(1,:)=two;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kn=ko;                       %%initial ko for first state.
twn=two;
tspan=[tstart tend];         %%time interval for first state

r = zeros(N_t,3);
T = zeros(N_t,3);
n = zeros(N_t,3);
b = zeros(N_t,3);
% kn= zeros(N+1,1);

r(1,:)=r0;
T(1,:)=T0;
n(1,:)=n0;
b(1,:)=b0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmat(1,:)=r0;
Tmat(1,:)=T0;
nmat(1,:)=n0;
bmat(1,:)=b0;

for j=1:1*N_t       
        count=j;       
        r_1=r(count,:);
        T_1=T(count,:);
        n_1=n(count,:);
        b_1=b(count,:);
        kappa=kn;
        twa=twn;

        K1_r = v0*T_1;
        K1_T = v0*(kappa).*n_1;
        K1_n =((-v0*kappa).*T_1) +(v0*b_1.*(twa));
        K1_b =-v0.*(twa).*n_1;

        r_2=r(j,:)+h/2*K1_r;
        T_2=T(j,:)+h/2*K1_T;
        n_2=n(j,:)+h/2*K1_n;
        b_2=b(j,:)+h/2*K1_b;
        
        K2_r = v0*T_2;
        K2_T = v0*(kappa).*n_2;
        K2_n =((-v0*kappa).*T_2) +(v0*b_2.*(twa));
        K2_b =-v0.*(twa).*n_2;

        r_3=r(j,:)+h/2*K2_r;
        T_3=T(j,:)+h/2*K2_T;
        n_3=n(j,:)+h/2*K2_n;
        b_3=b(j,:)+h/2*K2_b;
        
        K3_r = v0*T_3;
        K3_T = v0*(kappa).*n_3;
        K3_n =((-v0*kappa).*T_3) +(v0*b_3.*(twa));
        K3_b =-v0.*(twa).*n_3;

        r_4=r(j,:)+h*K3_r;
        T_4=T(j,:)+h*K3_T;
        n_4=n(j,:)+h*K3_n;
        b_4=b(j,:)+h*K3_b;

        K4_r = v0*T_4;
        K4_T = v0*(kappa).*n_4;
       K4_n =((-v0*kappa).*T_4) +(v0*b_4.*(twa));
        K4_b =-v0.*(twa).*n_4;

        r(j+1,:)=r(j,:)+h/6*(K1_r+2*K2_r+2*K3_r+K4_r);
        T(j+1,:)=T(j,:)+h/6*(K1_T+2*K2_T+2*K3_T+K4_T);
        n(j+1,:)=n(j,:)+h/6*(K1_n+2*K2_n+2*K3_n+K4_n);
        b(j+1,:)=b(j,:)+h/6*(K1_b+2*K2_b+2*K3_b+K4_b);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tstart=tend;                            
tend=tstart+Lstep;  
x1 = r(:,[1]);y1 = r(:,[2]); z1 = r(:,[3]);      %% x&y values of trajectory r at single step
x2 = T(:,[1]);y2 = T(:,[2]); z2 = T(:,[3]);     
x3 = n(:,[1]);y3 = n(:,[2]); z3 = n(:,[3]);
x4 = b(:,[1]);y4 = b(:,[2]); z4 = b(:,[3]);
rx=x1(end);ry=y1(end);rz=z1(end);dis=[rx  ry  rz];cdis=norm(dis);
%at the end of single step and calculating distance to target after single
%step

Tx=x2(end);Ty=y2(end);Tz=z2(end);   %% x&y values of trajectory T,n at the end of single step.
nx=x3(end);ny=y3(end);nz=z3(end);
bx=x4(end);by=y4(end);bz=z4(end);
%%assigning x&y values of the end of final step as initial condtions
%to the start of the next step (r0,t,n).
r0 = [rx;ry;rz];           %%assigning x&y values of the end of final step as initial condtions to nxt step. 
T0 = [Tx;Ty;Tz];                       
n0 = [nx;ny;nz];
b0 = [bx;by;bz];
ic = [r0;T0;n0;b0];
fsignal=(c0./norm(dis));    
Reward=rw*(1/osignal-1/fsignal) ;          %%calculate reward as a difference between two points
Rewardmat(i,ep)=Reward;       %% storing reward values in reward matrix
odis=cdis;                       %% assigning old distance as current distance.      % sitmulus at current postion.
sdelta=fsignal-osignal;           % difference in sitmulus between current postion and old postion.
sdeltamat(i,ep)=sign(sdelta);     %% storing sign of s delta difference.
osignal=fsignal;                  %% assigning current signal as old signal.
kmat(i,ep)=kn;                    %% storing kn values
twmat(i,ep)=twn;

for u=1:N_t
rx1mat(vg,1)=x1(u,1);                   
ry1mat(vg,1)=y1(u,1);
rz1mat(vg,1)=z1(u,1);
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
[~,stw]=min(abs(twrange-twn));
skmat(i,:)=sk;
stwmat(i,:)=stw;
cstate=Nstattw*2*(sk-1)+2*(stw-1)+cdel;        %%dterming current states postion in q matrix
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

r = zeros(N_t,3);
T = zeros(N_t,3);
n = zeros(N_t,3);
b = zeros(N_t,3);

r(1,:)=r0;
T(1,:)=T0;
n(1,:)=n0;
b(1,:)=b0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    twn=two+epslntw;
elseif arand==2
    kn=ko;  
    twn=two;
elseif arand==3
      kn=ko-epsln;                    %% third coulmn action.
    twn=two-epslntw;                    %% first column action.
else
    disp error1                      %% show error1
    break;
end

    for j=1:N_t        
        count=j;
        
       r_1=r(count,:);
        T_1=T(count,:);
        n_1=n(count,:);
        b_1=b(count,:);
        kappa=kn;
        twa=twn;
        
        K1_r = v0*T_1;
        K1_T = v0*(kappa).*n_1;
        K1_n =((-v0*kappa).*T_1) +(v0*b_1.*(twa));
        K1_b =-v0.*(twa).*n_1;
        
        r_2=r(j,:)+h/2*K1_r;
        T_2=T(j,:)+h/2*K1_T;
        n_2=n(j,:)+h/2*K1_n;
        b_2=b(j,:)+h/2*K1_b;
        
        K2_r = v0*T_2;
        K2_T = v0*(kappa).*n_2;
        K2_n =((-v0*kappa).*T_2) +(v0*b_2.*(twa));
        K2_b =-v0.*(twa).*n_2;

        r_3=r(j,:)+h/2*K2_r;
        T_3=T(j,:)+h/2*K2_T;
        n_3=n(j,:)+h/2*K2_n;
        b_3=b(j,:)+h/2*K2_b;
        
        K3_r = v0*T_3;
        K3_T = v0*(kappa).*n_3;
        K3_n =((-v0*kappa).*T_3) +(v0*b_3.*(twa));
        K3_b =-v0.*(twa).*n_3;

        r_4=r(j,:)+h*K3_r;
        T_4=T(j,:)+h*K3_T;
        n_4=n(j,:)+h*K3_n;
        b_4=b(j,:)+h*K3_b;
        
        K4_r = v0*T_4;
        K4_T = v0*(kappa).*n_4;
        K4_n =((-v0*kappa).*T_4) +(v0*b_4.*(twa));
        K4_b =-v0.*(twa).*n_4;

        r(j+1,:)=r(j,:)+h/6*(K1_r+2*K2_r+2*K3_r+K4_r);
        T(j+1,:)=T(j,:)+h/6*(K1_T+2*K2_T+2*K3_T+K4_T);
        n(j+1,:)=n(j,:)+h/6*(K1_n+2*K2_n+2*K3_n+K4_n);
        b(j+1,:)=b(j,:)+h/6*(K1_b+2*K2_b+2*K3_b+K4_b);
    end
    
    tstart=tend;                            
    tend=tstart+Lstep;  
    x1 = r(:,[1]);y1 = r(:,[2]); z1 = r(:,[3]);      %% x&y values of trajectory r at single step
    x2 = T(:,[1]);y2 = T(:,[2]); z2 = T(:,[3]);     
    x3 = n(:,[1]);y3 = n(:,[2]); z3 = n(:,[3]);
    x4 = b(:,[1]);y4 = b(:,[2]); z4 = b(:,[3]);
    
    rx=x1(end);ry=y1(end);rz=z1(end);
    Tx=x2(end);Ty=y2(end);Tz=z2(end);   %% x&y values of trajectory T,n at the end of single step.
    nx=x3(end);ny=y3(end);nz=z3(end);
    bx=x4(end);by=y4(end);bz=z4(end);

   r0 = [rx;ry;rz];           %%assigning x&y values of the end of final step as initial condtions to nxt step. 
   T0 = [Tx;Ty;Tz];                       
   n0 = [nx;ny;nz];
   b0 = [bx;by;bz];
   ic = [r0;T0;n0;b0];
    r(1,:)=r0;
    T(1,:)=T0;
    n(1,:)=n0;
    b(1,:)=b0;
    dis=[rx  ry  rz];
    cdis=norm(dis);
    fsignal=(c0./norm(dis));         % sitmulus at current postion.
    sdelta=fsignal-osignal;           % difference in sitmulus between current postion and old postion.
    Reward=rw*(1/osignal-1/fsignal);  %%calculate reward as a difference between two points
    sdeltamat(i)=sign(sdelta);        %% storing sign of s delta difference.
    osignal=fsignal;                  %% assigning current signal as old signal.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   kmat(i,ep)=kn;                     %% storing kn value
   twmat(i,ep)=twn;
u=1;
for u=1:N_t
rx1mat(vg,1)=x1(u,1);                   
ry1mat(vg,1)=y1(u,1);
rz1mat(vg,1)=z1(u,1);
vg=vg+1;
end
%%%%%%%
if sign(sdelta)==1
    cdel=1;
elseif sign(sdelta)==-1
    cdel=2;
else
    disp error2
    break;
end
[~,sk]=min(abs(krange-kn));
[~,stw]=min(abs(twrange-twn));
skmat(i,:)=sk;
stwmat(i,:)=stw;
fstate=Nstattw*2*(sk-1)+2*(stw-1)+cdel; 
qmat(cstate,arand)=(qmat(cstate,arand))+alpha.*(Reward+gama.*(max(qmat(fstate,:))-qmat(cstate,arand))); %%updating current q values
cstate=fstate; %assigning future states as current state
fstatemat(i,:)=cstate;
if abs(kn)>kmax 
    disp ('wrong interval K')
    break;
elseif abs(twn)>twmax
     disp ('wrong interval tw')
     break;
end
cdismat(i,:)=cdis;
i=i+1;         %counter updated
ko=kn;         %assigning k new as k old
two=twn;
if cdis<50
 break         % break loop if distance to target is less than 1
end
if i>5000       % break loop if iterations more than given value.
    break
end
end

for ep=1:1
 figure
plot3(rx1mat(:,ep),ry1mat(:,ep),rz1mat(:,ep))
hold on;plot3(0,0,0,'r*');plot3(rx1mat(end),ry1mat(end),rz1mat(end),'y*');plot3(rox,roy,roz,'g*');hold off
axis equal
box on; grid on;
title('3D Radial concentration field trajectory r(t)')
xlabel('x(t)') 
ylabel('y(t)')
zlabel('z(t)')
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
