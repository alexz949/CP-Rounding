function [D,tc,tp] = apply_kr_qr(Q,Q_hat,B,C)
%  Q: cell array of factors QR's Q 
%  Q_hat: cell array of pairwise Q
%  B: cell array of factors on RHS
%  C: RHS transpose
%  D: QtB
d = length(Q);
n = size(Q{1},2);
tc = 0;
tp = 0;
tic

%update RHS with Q'
for i = 1:d
    B{i} = Q{i}' * B{i};
end
t = toc; tc = tc + t;



test = cell(d-1,1);

%update pairwise
tic
for i = 1:d-1
    B{i+1} = khatrirao(B{i}, B{i+1});
    % possible to update D as well
    test{i} = Q_hat{i}' * B{i+1};
    B{i+1} = test{i};

end
t = toc; tp = tp + t;

D = B{d};
D = D * C';






end
