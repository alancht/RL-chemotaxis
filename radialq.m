function dy = radialq(y,q,ro,c0,kn,v0)
r = y(1:2);
T = y(3:4);
n = y(5:6);
rdot = v0*T;
Tdot = v0*(kn).*n;
ndot =(-v0*(kn).*T);
dy = [rdot;Tdot;ndot];
end