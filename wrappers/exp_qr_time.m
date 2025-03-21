function time = exp_qr_time(T,X,d,n)
    %%% Test for explicit QR
    t_back_solve = 0;
    t_factor_QR = 0;
    t_apply_factor_QR = 0;
    t_QR_R = 0;
    t_apply_QR_R = 0;
    
    
    %%QR of Factor matrices
    tic
    QF = cell(d-1,1);
    RF = cell(d-1,1);
    for i = 1 : d-1
        [QF{i},RF{i}] = qr(T.U{i},0);
    end
    t = toc; t_factor_QR = t_factor_QR + t;
    
    %%QR on R
    tic
    Rk = khatrirao(RF);
    [Q0,R0] = qr(Rk,0);
    t = toc; t_QR_R = t_QR_R + t;
    
    XXy = cell(d-1,1);
    %%Apply factor QR to RHS
    tic
    for i =  1:d-1
        XXy{i} = QF{i}' * X.U{i};
    end
    t = toc; t_apply_factor_QR = t_apply_factor_QR + t;
    
    Kx = khatrirao(XXy);
    %%Apply R's QR
    tic
    Kx = Q0' * Kx;
    t = toc; t_apply_QR_R = t_apply_QR_R + t;
    
    %%solve time
    tic
    XXX = R0 \ (Kx * X.U{d}');
    T.U{d} = XXX';
    t = toc; t_back_solve = t_back_solve + t;
      
    expt = [t_apply_factor_QR,t_factor_QR,t_apply_QR_R,t_QR_R,t_back_solve];
    time = expt;
end
