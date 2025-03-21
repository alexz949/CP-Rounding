function [T,diff] = sinsums(d,n,scale)
% create rank d ktensor representation of sin of sums tensor with d modes
% d: number of modes of the tensor, end rank
% n: dimension of tensor
% T: ktensor

if nargin < 3
    scale = 1;
end

lambda = ones(d,1);

x = linspace(0,2*pi,n)';
a = linspace(0,pi/d*(d-1)/scale,d);
diff = a(2) - a(1);

offs = cell(d,d);

for i = 1:d
    for j = 1:d
        if i == j
            offs{i,j} = sin(x);
        else
            offs{i,j} = sin(x+(a(j)-a(i)))/sin(a(j)-a(i));
        end
    end
end

% form factor matrices
A = cell(1,d);
for i = 1:d
    y = offs{1,i};
    for j = 2:d
        y = [y,offs{j,i}];
    end
    A{i} = y;
end

T = ktensor(lambda,A);

end