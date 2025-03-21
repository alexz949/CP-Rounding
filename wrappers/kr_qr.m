function [Q,Q_hat,R,tc,tp] = kr_qr(F)
%  F: cell array of factors of coefficient KR prod
%  Q: cell array of factors QR's Q 
%  Q_hat: cell array of pairwise Q
%  R: upper triangular matrix 

d = length(F);

tc = 0;
tp = 0;
Q = cell(d,1);
T = cell(d,1);
Q_hat = cell(d-1,1);

%QR on factor matrices
tic
for i = 1:d
    [Q{i},T{i}] = qr(F{i},0);
end
t = toc; tc = tc + t;

tic
%QR on R
for i = 1:d-1
    [Q_hat{i},rr] = qr(khatrirao(T{i},T{i+1}),0);
    
    T{i+1} = rr;
end
t = toc; tp = tp + t;

R = rr;


end