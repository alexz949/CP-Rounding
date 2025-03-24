clc
clear all

warning("off", "all")
maxNumCompThreads(1);

addpath("../algorithms/");
addpath("../wrappers/")
addpath("../tensor_toolbox/")

d = 3 %number of modes
n = 10000
r = 50  % rank
maxiter = 20;
tol = 0;
em = 'fast';  %error method options:  'fast', 'full' and 'lowmem' ('fast' method provides an estimate rather than an exact error, sacrificing precision for speed.)
pl = 5;
%rng()
flag = false;


%% generating input tensor
eta = 1e-5;
kappa = 1e10;
A = gallery('randsvd',[n,r], kappa);

[~,S,V] = svd(A,"econ");

cores = cell(d,1);
noise = cell(d,1);
for i = 1:d
    [U,~] = qr(randn(n,r),0);
    cores{i} = U*S*V;
    noise{i} = normrnd(0,1,n,9*r);
end
X  = ktensor(cores);
X.lambda = X.lambda/norm(X);
X = normalize(X,0);

N = ktensor(noise);
N.lambda = N.lambda/norm(N);
N = normalize(N,0);
T  = X + eta*N; 
T = normalize(T,0);
    
% single precision input tensor    
T_single = T;
T_single.lambda = single(T.lambda);
for j =1:d
    T_single.U{j} = single(T.U{j});
end
    
%% Initializaing cores
U = cell(d,1);
for j = 1:d
    U{j} = randn(n,r);
end
U_single = cell(d,1);
for j =1:d
    U_single{j} = single(U{j});
end



%% CP-ALS_single
[M_als_single,U_als_single,out_als_single] = cp_als_time(T_single,r,'init',U_single,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);


%CP-ALS-QR-New-Single (QR Implicit)
[M_imp_single,U_imp_single,out_imp_single] = cp_als_qr_new(T_single,r,'init',U_single,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);

if flag == true
    %CP-ALS-QR-Single (QR Explicit)
    [M_exp_single,U_exp_single,out_exp_single] = cp_als_qr(T_single,r,'init',U_single,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
end

%% double
%CP-ALS-Double
[M_als_double,U_als_double,out_als_double] = cp_als_time(T,r,'init',U,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);


%CP-ALS-QR-New-Double (QR Implicit)
[M_imp_double,U_imp_double,out_imp_double] = cp_als_qr_new(T,r,'init',U,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);

if flag == true
    % %CP-ALS-QR-Double (QR Explicit)
    [M_exp_double,U_exp_double,out_exp_double] = cp_als_qr(T,r,'init',U,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
end

als_err_single = out_als_single.relerr;
als_err_double = out_als_double.relerr;
imp_qr_err_single = out_imp_single.relerr;
imp_qr_err_double = out_imp_double.relerr;

if flag == true
    exp_qr_err_single = out_exp_single.relerr;
    exp_qr_err_double = out_exp_double.relerr;
end

als_single_time    = out_als_single.times;
als_double_time    = out_als_double.times;
imp_qr_single_time = out_imp_single.times;
imp_qr_double_time = out_imp_double.times;

if flag == true
    exp_qr_single_time = out_exp_single.times;
    exp_qr_double_time = out_exp_double.times;
end

total_time_als_single    = out_als_single.total_times;
total_time_als_double    = out_als_double.total_times;
total_time_imp_qr_single = out_imp_single.total_times;
total_time_imp_qr_double = out_imp_double.total_times;


if flag == true
    total_time_exp_qr_single{i} = out_exp_single.total_times;
    total_time_exp_qr_double{i} = out_exp_double.total_times;
end

%% ploting figure 5

%%% data prep for time plot
mean_tals_double = mean(als_double_time(1:end,:),1);
mean_tals_single = mean(als_single_time(1:end,:),1);
mean_timp_double = mean(imp_qr_double_time(1:end,:),1);
mean_timp_single = mean(imp_qr_single_time(1:end,:),1);

if flag == true
    mean_texp_double = mean(als_data.exp_qr_double_time(2:end,:),1);
    mean_texp_single = mean(als_data.exp_qr_single_time(2:end,:),1);
end

 
als_double = zeros(1,6);
als_single = zeros(1,6);
imp_double = zeros(1,6);
imp_single = zeros(1,6);
exp_double = zeros(1,6);
exp_single = zeros(1,6);

for i = 1:1
    als_double(i,:) = [mean_tals_double(i,1), mean_tals_double(i,2),0,0,mean_tals_double(i,3),mean_tals_double(i,4)];
    als_single(i,:) = [mean_tals_single(i,1), mean_tals_single(i,2),0,0,mean_tals_single(i,3),mean_tals_single(i,4)];
    imp_double(i,:) = [mean_timp_double(i,1),mean_timp_double(i,2),mean_timp_double(i,3),mean_timp_double(i,4),mean_timp_double(i,5),mean_timp_double(i,6)];
    imp_single(i,:) = [mean_timp_single(i,1),mean_timp_single(i,2),mean_timp_single(i,3),mean_timp_single(i,4),mean_timp_single(i,5),mean_timp_single(i,6)];
    if flag == true
        exp_double(i,:) = [mean_texp_double(i,1),mean_texp_double(i,2),mean_texp_double(i,3),mean_texp_double(i,4),mean_texp_double(i,5),mean_texp_double(i,6)];
        exp_single(i,:) = [mean_texp_single(i,1),mean_texp_single(i,2),mean_texp_single(i,3),mean_texp_single(i,4),mean_texp_single(i,5),mean_texp_single(i,6)];
    end
end

%% prep data
sdata = [];
for i = 1:1
    if flag == true
        sdata = [sdata; als_single(i,:);als_double(i,:);imp_single(i,:);imp_double(i,:);exp_single(i,:);exp_double(i,:)];
    else
        sdata = [sdata; als_single(i,:);als_double(i,:);imp_single(i,:);imp_double(i,:)];
    end
end


nx = size(als_err_single,1);
fig =figure;

% Adjust PaperPosition (in inches)
set(fig, 'PaperPosition', [0, 0, 14, 9]);  % Set the size (width=8 inches, height=6 inches)
% Set the PaperSize to the same dimensions to ensure no clipping
set(fig, 'PaperSize', [14, 9]);
screenSize= get(0,'ScreenSize');
set(gcf,'Position', screenSize);

subplot(1,2,1)
semilogy(1:nx,als_err_double,'-.o','Color','#0A71ED','linewidth',2.4,'MarkerSize',11.5), hold on
semilogy(1:nx,imp_qr_err_double,':*','Color','#0A71ED','linewidth',2.3,'MarkerSize',11)
semilogy(1:nx,als_err_single,'-.^','Color','#FF0000','linewidth',2.2,'MarkerSize',10) 
semilogy(1:nx,imp_qr_err_single,':+','Color','#FF0000','linewidth',2.1,'MarkerSize',10)



xlabel('iteration number')
ylabel('Relative residual error')
ylim([min(ylim) 1.1*max(ylim)])
l= legend('NE Double','QR Imp Double','NE Single', 'QR Imp Single'); 
l.FontSize = 18;
l.Location = 'northwest';
set(gca,'fontsize',25)



subplot(1,2,2)
x_positions = [2,3,4,5];
bar(x_positions,sdata(:,1:end-1),'stacked')
ylabel('Time (secs)')
if flag == true
    xticklabels({'NE Single', 'NE Double','QR Imp Single','QR Imp double','QR Exp Single','QR Exp double'})
else
    xticks(x_positions);
    xticklabels({'NE Single', 'NE Double','QR Imp Single','QR Imp Double'})
    xtickangle(45);
end
l = legend('MTTKRP/TTM','Gram/QR','Pairwise QR','Apply Pairwise QR','Back solve/NE solve','Err');
l.FontSize = 18;
l.Location = 'northeast';
set(gca,'fontsize',25)
% saveas(gcf,'Fig_5.pdf')



