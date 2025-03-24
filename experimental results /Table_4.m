clc
clear all

addpath("../algorithms/");
addpath("../wrappers/")
addpath("../tensor_toolbox/")

dt = [3,3,5,3,5,3,5];
n = 50;
diff = [1, 0.1, 0.1, 0.01, 0.01, 0.0001,0.001];
cond_d = zeros(7,1);
NE_err = zeros(7,1);
Exp_err = zeros(7,1);
Imp_err = zeros(7,1);

for idx_d = 1:7
    d = dt(idx_d)
    T = sinsums_scaled(d,n,diff(idx_d));
    X = sinsum_full(d,n);
    for i = 1:d
        T.U{i} = (T.U{i});
        X.U{i} = (X.U{i});
    end
    
    true_err = norm(full(T) - full(X)) / norm(X);
        
    % generate random ktensor fac tors
    Ty = cell(d-1,1);
    Xy = cell(d-1,1);
    for i = 1 : d-1
        Ty{i} = T.U{i};
        Xy{i} = X.U{i};
    end
        
    true = T.U{d};
    K = khatrirao(Ty);
    condK = cond(K);
    
    % test for normal equation Grams
    G = Ty{1}'*Ty{1};
    for i = 2:d-1
        G = G.*(Ty{i}'*Ty{i});
    end
    
    
    % precompute cross products with F
    C = Ty{1}'*Xy{1};
    for k=2:length(Xy)
        C = C .* (Ty{k}'*Xy{k});
    end
    
    % XX = (G \ C * X.U{d}')';
    % T.U{d} = XX;
    T.U{d} = X.U{d} * (C' / G);
    
    normal_err = norm(full(T) - full(X)) / norm(X);
    normal_fwd_err = norm(true-T.U{d})/norm(true)
    
    
    Kx = khatrirao(Xy);
    [Qk,Rk] = qr(K,0);
    
    
    
    Kx = Qk' * Kx;
    XXX = Rk \ (Kx * X.U{d}');
    T.U{d} = XXX';
    
    exp_err = norm(full(T) - full(X)) / norm(X);
    
    exp_fwd_err = norm(true-T.U{d})/norm(true)
    
    [Qp,Qhatp,Rp,ttc,ttp] = kr_qr(Ty);
    
    [D,tttc,tttp] = apply_kr_qr(Qp,Qhatp,Xy,X.U{d});
    
    XX = (Rp \ D)';
    T.U{d} = XX;
    
    pairwise_err = norm(full(T) - full(X)) / norm(X);
    pairwise_fwd_err = norm(true-T.U{d})/norm(true);
    cond_d(idx_d) = condK;
    NE_err(idx_d) = normal_fwd_err;
    Exp_err(idx_d) = exp_fwd_err;
    Imp_err(idx_d) = pairwise_fwd_err;
end

fwd_err = table(dt', diff',cond_d,NE_err,Exp_err,Imp_err)


