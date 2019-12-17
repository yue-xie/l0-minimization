function [f0,f1,f2,card0,card1,card2,time0,time1,time2,KKT_res0,KKT_res1,KKT_res2,rho_intrac,iternum0,iternum1,iternum2] = hiddenconvex_vs_nonhiddenconvex_v3(n,A,b,gamma,S,V)
timemax = 200;
AA = A'*A;
%% ADMM-revised
y_initial = [ones(n,1);zeros(2*n,1)];
rho = 1;
delta = rho/2;
y = y_initial;
alpha = zeros(3*n,1);
P = [eye(n),sqrt(2)*eye(n),eye(n);eye(n),-sqrt(2)*eye(n),eye(n);-sqrt(2)*eye(n),zeros(n,n),sqrt(2)*eye(n)]/2;
Q = [eye(n),zeros(n,2*n);zeros(n,n),V,zeros(n,n);zeros(n,2*n),eye(n)];
G = P*Q;
iter = 0;
GAMMA = 1.01;
tic
while(1)
    fprintf('The %d iteration.\n',iter+1);
    %update x
    q = G'*[-2*A'*b+rho*(alpha(1:n)/rho-y(1:n));2*A'*b+rho*(alpha(n+1:2*n)/rho-y(n+1:2*n));rho*(alpha(2*n+1:3*n)/rho-y(2*n+1:3*n))-gamma*ones(n,1)];
    q1 = q(1:n);
    l1 = norm(q1);
    q2 = q(n+1:2*n);
    q3 = q(2*n+1:3*n);
    l3 = norm(q3);
    z2 = -q2./(rho+4*diag(S));
    thresh = 0;
    if l1 > thresh && l3 > thresh
        z1 = -(l1+l3)*q1/(2*rho*l1);
        z3 = -(l1+l3)*q3/(2*rho*l3);
    end
    if l1 <= thresh && l3 <= thresh
        z1 = zeros(n,1);
        z3 = zeros(n,1);
    end
    if l1 > thresh && l3 <= thresh
        z1 = -q1/(2*rho);
        z3 = -q1/(2*rho);  % can be modified
    end
    if l1 <= thresh && l3 > thresh
        z1 = -q3/(2*rho);  % can be modified
        z3 = -q3/(2*rho);
    end
    z = [z1;z2;z3];
    x = G*z;
    %update y
    y0 = y;
    y = x + alpha/rho;
    y1 = y(1:n);
    y2 = y(n+1:2*n);
    y3 = y(2*n+1:3*n);
    y1 = max(y1,zeros(n,1));
    y2 = max(y2,zeros(n,1));
    y3 = max(y3,zeros(n,1));
    y3 = min(y3,ones(n,1));
    y = [y1;y2;y3];
    %update alpha
    alpha = alpha + rho*(x-y);
    iter = iter + 1;
    %keep monitoring the other residuals
    res5 = norm(x-y);
    res6 = rho*norm(y-y0);
    %conditional updating
    if (rho <= 2000) && (res5/res6*rho^2*sqrt(2) + delta > rho)
        rho = GAMMA * rho;
    end
    KKT_res = norm([res5,res6],inf)
    %stopping criterion
    if KKT_res <= 1e-4 || toc > timemax
        break;
    end
end
time0 = toc;
iternum0 = iter;
KKT_res0 = KKT_res;
x_temp = y;
x1 = x_temp(1:n);
x2 = x_temp(n+1:2*n);
spar_temp = 0;
for t = 1:n
    if abs(x1(t)-x2(t)) < 1e-5
        spar_temp = spar_temp + 1;
    end
end
f0 = (A*(x1-x2)-b)'*(A*(x1-x2)-b) + gamma*(n - spar_temp);
card0 = n - spar_temp;
%% ADMM-perturb
y_initial = [ones(n,1);zeros(2*n,1)];
alpha = 1e-3;
rho = 1/(2*alpha);
mu = 3/alpha;
y = y_initial;
x = y;
lambda = zeros(3*n,1);
P = [eye(n),sqrt(2)*eye(n),eye(n);eye(n),-sqrt(2)*eye(n),eye(n);-sqrt(2)*eye(n),zeros(n,n),sqrt(2)*eye(n)]/2;
Q = [eye(n),zeros(n,2*n);zeros(n,n),V,zeros(n,n);zeros(n,2*n),eye(n)];
G = P*Q;
iter = 0;
tic
while(1)
    fprintf('The %d iteration.\n',iter+1);
    lambda = lambda/2;
    %update x
    x0 = x;
    q = G'*([-2*A'*b+rho*(lambda(1:n)/rho-y(1:n));2*A'*b+rho*(lambda(n+1:2*n)/rho-y(n+1:2*n));rho*(lambda(2*n+1:3*n)/rho-y(2*n+1:3*n))-gamma*ones(n,1)] - mu*x0);
    q1 = q(1:n);
    l1 = norm(q1);
    q2 = q(n+1:2*n);
    q3 = q(2*n+1:3*n);
    l3 = norm(q3);
    z2 = -q2./( (rho+mu) +4*diag(S));
    thresh = 0;
    if l1 > thresh && l3 > thresh
        z1 = -(l1+l3)*q1/(2*(rho+mu)*l1);
        z3 = -(l1+l3)*q3/(2*(rho+mu)*l3);
    end
    if l1 <= thresh && l3 <= thresh
        z1 = zeros(n,1);
        z3 = zeros(n,1);
    end
    if l1 > thresh && l3 <= thresh
        z1 = -q1/(2*(rho+mu));
        z3 = -q1/(2*(rho+mu));  % can be modified
    end
    if l1 <= thresh && l3 > thresh
        z1 = -q3/(2*(rho+mu));  % can be modified
        z3 = -q3/(2*(rho+mu));
    end
    z = [z1;z2;z3];
    x = G*z;
    %update y
    y0 = y;
    y = x + lambda/rho;
    y1 = y(1:n);
    y2 = y(n+1:2*n);
    y3 = y(2*n+1:3*n);
    y1 = max(y1,zeros(n,1));
    y2 = max(y2,zeros(n,1));
    y3 = max(y3,zeros(n,1));
    y3 = min(y3,ones(n,1));
    y = [y1;y2;y3];
    %update alpha
    lambda0 = lambda;
    lambda = lambda + rho*(x-y);    
    iter = iter + 1;
    %keep monitoring the other residuals
    res5 = norm(lambda - 2*lambda0)/rho;
    res6 = norm(rho*(y-y0)+mu*(x-x0));
    KKT_res = norm([res5,res6],inf)
    %stopping criterion
    if (rho >= 1/(2*1e-3)) && (KKT_res <= 1e-2 || toc > timemax)
        break;
    end
end
time1 = toc;
iternum1 = iter;
KKT_res1 = max(norm(x-y),res6);
x_temp = y;
x1 = x_temp(1:n);
x2 = x_temp(n+1:2*n);
spar_temp = 0;
for t = 1:n
    if abs(x1(t)-x2(t)) < 1e-5
        spar_temp = spar_temp + 1;
    end
end
f1 = (A*(x1-x2)-b)'*(A*(x1-x2)-b) + gamma*(n - spar_temp);
card1 = n - spar_temp;

%% ADMM-nonconvex subproblem-baron
y_initial = [ones(n,1);zeros(2*n,1)];
rho = 1;
Ab2 = 2*A'*b;
y = y_initial;
x = y_initial;
alpha = zeros(3*n,1);
iter = 0;
h0 = inf;
tau = 0.9;
GAMMA = 1.01;
rho_max = 500;
tic
while(1)
    fprintf('The %d iteration.\n',iter+1);
    %update x
    x_initial = x;
    fun = @(x)(x - y + alpha/rho)'*(x - y + alpha/rho);
    lb = zeros(3*n,1);
    ub = [Inf*ones(2*n,1);ones(n,1)];
    nlcon = @(x)(x(1:n)+x(n+1:2*n))'*x(2*n+1:3*n);
    x = baron(fun,[],[],[],lb,ub,nlcon,0,0,[],x_initial,baronset('MaxTime',200));
    %update y
    y0 = y;
    M = [rho*eye(n)+2*AA,-2*AA;-2*AA,rho*eye(n)+2*AA];
    y12 = M\( rho*x(1:2*n) + alpha(1:2*n) + [Ab2;-Ab2] );
    y3 = x(2*n + 1:3*n) + alpha(2*n+1:3*n)/rho + gamma/rho*ones(n,1);
    y = [y12;y3];
    %update alpha
    alpha = alpha + rho*(x-y);
    iter = iter + 1;
    %keep monitoring the other residuals
    res5 = norm(x-y);
    res6 = rho*norm(y-y0);
    KKT_res = norm([res5,res6],inf)
    if rho < rho_max && res5 >= 1e-2 && (res5 > tau * h0 || iter == 1)
        rho = GAMMA * rho;
    end
    h0 = res5;
    %stopping criterion
    if KKT_res <= 1e-4 || toc > timemax
        break;
    end
end
time2 = toc;
iternum2 = iter;
rho_intrac = rho;
KKT_res2 = KKT_res;
x_temp = x;
x1 = x_temp(1:n);
x2 = x_temp(n+1:2*n);
spar_temp = 0;
for t = 1:n
    if abs(x1(t)-x2(t)) < 1e-5
        spar_temp = spar_temp + 1;
    end
end
f2 = (A*(x1-x2)-b)'*(A*(x1-x2)-b) + gamma*(n - spar_temp);
card2 = n - spar_temp;
end