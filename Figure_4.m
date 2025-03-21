clc
clear all

als_data = struct;
n = 8 %dimension of tensor
em = 'lowmem'  %error method options:  'fast', 'full' and 'lowmem' ('fast' method provides an estimate rather than an exact error, sacrificing precision for speed.)
pl = 10;

figure

for i =1:2
    if i == 1
        rng(3)
    else
        rng(2)
    end
    
    
    %%% 10-way 
    d = 10; %number of modes
    r = 10; %CP rank
    maxiter = 40;
    tol = 0;
    
    % generate sin of sums tensor
    T = sinsums(d,n);
    X = sinsum_full(d,n);
    
    % Perform CP rounding
    % CP-ALS
    [M_als10,U_als10,out_als10] = cp_als_time(X,r,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
    % CP-ALS-QR-new (QR Implicit)
    [M_imp10,U_imp10,out_imp10] = cp_als_qr_new(X,r,'init',U_als10,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
    
    als_data.als10 = out_als10.relerr;
    als_data.imp10 = out_imp10.relerr;
    
    %%% 7-way
    d = 7; %number of modes
    r = 7; %CP rank
    maxiter = 40;
    tol = 0;
    
    % generate sin of sums tensor
    T = sinsums(d,n);
    X = sinsum_full(d,n);
    
    % Perform CP rounding
    % CP-ALS
    [M_als7,U_als7,out_als7] = cp_als_time(X,r,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
    % CP-ALS-QR-new (QR Implicit)
    [M_imp7,U_imp7,out_imp7] = cp_als_qr_new(X,r,'init',U_als7,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
    
    als_data.als7 = out_als7.relerr;
    als_data.imp7 = out_imp7.relerr;
    
    %%% 5-way
    d = 5; %number of modes
    r = 5;
    maxiter = 40;
    tol = 0;
    
    % generate sin of sums tensor
    T = sinsums(d,n);
    X = sinsum_full(d,n);
    
    % Perform CP rounding
    % CP-ALS
    [M_als5,U_als5,out_als5] = cp_als_time(X,r,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
    % CP-ALS-QR-new (QR Implicit)
    [M_imp5,U_imp5,out_imp5] = cp_als_qr_new(X,r,'init',U_als5,'maxiters',maxiter,'tol',tol,'printitn',pl,'errmethod',em);
    
    als_data.als5 = out_als5.relerr;
    als_data.imp5 = out_imp5.relerr;

    % ploting
    subplot(1,2,i)
    semilogy(1:40,als_data.als5,'-.','Color','#EDD80B','linewidth',1), hold on
    semilogy(1:40,als_data.imp5,':','Color','#EDD80B','linewidth',1)
    
    semilogy(1:40,als_data.als7,'-.','Color','#0A71ED','linewidth',1)
    semilogy(1:40,als_data.imp7,':','Color','#0A71ED','linewidth',1)
    semilogy(1:40,als_data.als10,'-.','Color','#F33E09','linewidth',1)
    semilogy(1:40,als_data.imp10,':','Color','#F33E09','linewidth',1)
    ylim([1e-15 10])
    set(gca,'fontsize',10)
    %title('Relative Error for Different d (1st trial)') 
    xlabel('iteration number')
    ylabel('Relative residual error')

    if i == 1
        title('1st trial')
        legend('5-way NE','5-way QR Imp','7-way NE','7-way QR Imp','10-way NE','10-way QR Imp','fontsize',6)
    else
        title('2nd trial')
    end
end

set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);
saveas(gcf,'Fig_4.pdf')
