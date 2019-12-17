m = 256;
n = 1024;
SNR = [5,10];
M = [1,5,15];
gamma = [0.1,1,10,50];
Table = [];
for k = 1:2
    for i = 1:3
        A = randn(m,n);
        AA = A'*A;
        [V,S] = eig(AA);
        c1 = norm(V'*V - eye(n)); %check orthogonality
        xstar = 120*(rand(n,1))-60;
        sparsity = 0;
        for t = 1:n
            if xstar(t) > M(i) || xstar(t) < -M(i) || xstar(t)==0
                xstar(t) = 0;
                sparsity = sparsity + 1;
            end
        end
        b = A*xstar + sqrt(xstar'*xstar/SNR(k))*randn(m,1);
        for j = 1:4
            [f_IHT,f_IHTWS,f_admm,time_IHT,time_IHTWS,time_ADMM,card_IHT,card_IHTWS,card_admm,e1,e2,e3,KKTres_admm,rho_admm,time_rhomax,ratio] = Algorithms4(n,A,b,gamma(j),S,V,xstar);
            data = [SNR(k),sparsity,gamma(j),f_IHT,f_IHTWS,f_admm,time_IHT,time_IHTWS,time_ADMM,card_IHT,card_IHTWS,card_admm,e1,e2,e3,KKTres_admm,rho_admm,time_rhomax,ratio,c1];
            Table = [Table;data];
        end
    end
end