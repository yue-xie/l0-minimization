function [f_IHT,f_IHTWS,f_admm,time_IHT,time_IHTWS,time_ADMM,card_IHT,card_IHTWS,card_admm,e1,e2,e3,KKTres_admm,rho_admm,time_rhomax,ratio] = Algorithms4(n,A,b,gamma,S,V,xstar)
%% Iterative Hard Threshold
epsilon = 1e-6;
Lambda = sqrt(max(diag(S)));
v = b/(1.1*Lambda);
Phi = A/(1.1*Lambda);
lambda = gamma/((1.1^2)*(Lambda^2));
H = zeros(n,1);
ub = inf;
tic
for k = 1:50
    y = min(k-1,1)*randn(n,1);
    count = 0;
    while(1)
        count = count + 1
        y0 = y;
        y = y0 + Phi'*(v-Phi*y0);
        for i = 1:n
            if abs(y(i)) <= sqrt(lambda)
                H(i) = 0;
            else
                H(i) = y(i);
            end
        end
        y = H;
        if norm(y-y0) < epsilon
            break;
        end
    end
    x_temp = y;
    card_temp = n;
    for i = 1:n
        if abs(x_temp(i)) <= 1e-8
            card_temp = card_temp - 1;
        end
    end
    f_temp = (A*x_temp-b)'*(A*x_temp-b)+gamma*card_temp;
    if f_temp < ub
        x_IHT = x_temp;
        ub = f_temp;
        f_IHT = f_temp;
        card_IHT = card_temp;
    end
end
time_IHT = toc;
e1 = (A*(x_IHT-xstar))'*(A*(x_IHT-xstar))/((A*xstar)'*(A*xstar));
%% Iterative Hard Thresholding with warm-start
tic
r = b;
y = zeros(n,1);
cost0 = b'*b;
Alpha = zeros(n,1);
spar = n;
% matching persuit
while(1)
    Alpha_max = 0;
    for i = 1:n
        Alpha(i) = r'*A(:,i)/(A(:,i)'*A(:,i));
        if abs(Alpha(i)) > Alpha_max
            index = i;
            Alpha_max = abs(Alpha(i));
        end
    end
    y0 = y;
    y(index) = y0(index) + Alpha(index);
    if abs(y0(index))<1e-8
        if abs(y(index)) >= 1e-8
            spar = spar - 1;
        end
    else
        if abs(y(index)) < 1e-8
            spar = spar + 1;
        end
    end
    cost = (A*y - b)'*(A*y - b) + gamma*(n-spar);
    if cost > cost0
        break
    end
    cost0 = cost;
    r = r - Alpha(index)*A(:,index);
end
start = y0;
% Iterative Hard Threshold
epsilon = 1e-6;
Lambda = sqrt(max(diag(S)));
v = b/(1.1*Lambda);
Phi = A/(1.1*Lambda);
lambda = gamma/((1.1^2)*(Lambda^2));
H = zeros(n,1);
y = start;
count = 0;
while(1)
    count = count + 1
    y0 = y;
    y = y0 + Phi'*(v-Phi*y0);
    for i = 1:n
        if abs(y(i)) <= sqrt(lambda)
            H(i) = 0;
        else
            H(i) = y(i);
        end
    end
    y = H;
    if norm(y-y0) < epsilon
        break;
    end
end
x_IHTWS = y;
card_IHTWS = n;
for i = 1:n
    if abs(x_IHTWS(i)) < 1e-8
        card_IHTWS = card_IHTWS - 1;
    end
end
f_IHTWS = (A*x_IHTWS-b)'*(A*x_IHTWS-b)+gamma*card_IHTWS;
time_IHTWS = toc;
e2 = (A*(x_IHTWS-xstar))'*(A*(x_IHTWS-xstar))/((A*xstar)'*(A*xstar));
%% ADMM-revised
y = [ones(n,1);zeros(2*n,1)];
rho = gamma;
delta = rho/2;
alpha = zeros(3*n,1);
P = [eye(n),sqrt(2)*eye(n),eye(n);eye(n),-sqrt(2)*eye(n),eye(n);-sqrt(2)*eye(n),zeros(n,n),sqrt(2)*eye(n)]/2;
Q = [eye(n),zeros(n,2*n);zeros(n,n),V,zeros(n,n);zeros(n,2*n),eye(n)];
G = P*Q;
z = zeros(3*n,1);
iter = 0;
GAMMA = 1.01;
flag = 1;
rho_bar = 2000;
iter_rhomax = 0;
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
    if (rho <= rho_bar) && res5*rho^2*sqrt(2) > (rho-delta)*res6
        rho = GAMMA * rho;
    elseif rho > rho_bar && flag == 1
        iter_rhomax = iter;
        flag = 0;
    end
    
    %stopping criterion
    KKT_res = norm([res5,res6],inf)
    if KKT_res <= 1e-4 || toc > 300
        break;
    end
end
time_ADMM = toc;
rho_admm = rho;
KKTres_admm = KKT_res;
time_rhomax = iter_rhomax / iter;
ratio = res6/res5/rho;
x = y;
x1 = x(1:n);
x2 = x(n+1:2*n);
x3 = x(2*n+1:3*n);
spar = 0;
for t = 1:n
    if abs(x1(t)-x2(t)) < 1e-8
        spar = spar + 1;
    end
end
f = (A*(x1-x2)-b)'*(A*(x1-x2)-b) + gamma*(n - spar);
card_admm = n-spar;
if KKT_res <= 1e-4
    f_admm = (A*(x1-x2)-b)'*(A*(x1-x2)-b) + gamma*(n - sum(x3));
    card_admm = n-sum(x3);
else
    f_admm = f;
end
x_admm = x1 - x2;
e3 = (A*(x_admm-xstar))'*(A*(x_admm-xstar))/((A*xstar)'*(A*xstar));
end