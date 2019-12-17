m = 10;
n0 = [20,50,100];
M0 = [1,5];
gamma0 = [1,10];
Table = [];
for iter1 = 1:3
    n = n0(iter1);
    for iter2 = 1:2
        M = M0(iter2);
        xstar = 120*(rand(n,1))-60;
        sparsity = 0;
        for t = 1:n
            if xstar(t) > M || xstar(t) < -M || xstar(t)==0
                xstar(t) = 0;
                sparsity = sparsity + 1;
            end
        end
        A = randn(m,n);
        [V,S] = eig(A'*A);
        c1 = norm(V'*V - eye(n)); % Check orthogonality of V
        b = A*xstar + sqrt(xstar'*xstar/10)*randn(m,1);
        for iter3 = 1:2
            gamma = gamma0(iter3);
            [f0,f1,f2,card0,card1,card2,time0,time1,time2,KKT_res0,KKT_res1,KKT_res2,rho_intrac,iternum0,iternum1,iternum2] = hiddenconvex_vs_nonhiddenconvex_v3(n,A,b,gamma,S,V);
            data = [m,n,n-sparsity,gamma,f0,f1,f2,card0,card1,card2,time0,time1,time2,KKT_res0,KKT_res1,KKT_res2,rho_intrac,iternum0,iternum1,iternum2,c1];
            Table = [Table;data];
        end
    end
end