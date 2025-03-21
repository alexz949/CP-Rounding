clc
clear all
maxNumCompThreads(1);

addpath("./algorithms/");
addpath("./wrappers/")
addpath("./tensor_toolbox/")

n = 2000;
per_data = struct;
per_data.res = zeros(24,5);
runs = 50;

for it = 1:runs
    tp = 1;
    for d = 3:8
        % generating sinsum data
        T = sinsums(d,n);
        X = sinsum_full(d,n);

        % Extracting factor matrices out of T and X 
        Ty = cell(d-1,1);
        Xy = cell(d-1,1);
        for i = 1 : d-1
            Ty{i} = T.U{i};
            Xy{i} = X.U{i};
        end
        
        %test for normal equation
        expt = exp_qr_time(T,X,d,n);
      
        %test for normal equation
        t_back_solve = 0;
        t_gram = 0;
        t_QR_R = 0;
        t_apply_QR_R = 0;
        t_apply_gram = 0;
        
        %Grams
        tic
        G = Ty{1}'*Ty{1};
        for i = 2:d-1
            G = G.*(Ty{i}'*Ty{i});
        end
        t = toc; t_gram = t_gram + t;
        % precompute cross products with F
        tic
        C = Ty{1}'*Xy{1};
        for k=2:length(Xy)
            C = C .* (Ty{k}'*Xy{k});
        end
        t = toc; t_apply_gram = t_apply_gram + t;
        
        tic
        T.U{d} = X.U{d} * (C' / G);
        t = toc; t_back_solve = t_back_solve + t;
        
        nort = [t_apply_gram,t_gram,t_apply_QR_R,t_QR_R,t_back_solve];
        
        t_krp = 0;
        t_back_solve = 0;
        t_gram = 0;
        t_apply_gram = 0;
        
        
        %test for pairwise elimination (Implicit QR)
        [Qp,Qhatp,Rp,ttc,ttp] = kr_qr(Ty);
        [D,tttc,tttp] = apply_kr_qr(Qp,Qhatp,Xy,X.U{d});
       
        
        tic
        XX = (Rp\D);
        T.U{d} = XX';
        t = toc; t_back_solve = t_back_solve+t;
        
        t_factor_QR = ttc;
        t_QR_R = ttp;
        t_apply_factor_QR = tttc;
        t_apply_QR_R = tttp;
      
        part = [t_apply_factor_QR,t_factor_QR,t_apply_QR_R,t_QR_R,t_back_solve];
        
        if it > 1
            per_data.res(tp,:) = per_data.res(tp,:) + part;
            per_data.res(tp+1,:) = per_data.res(tp+1,:) + nort;
            per_data.res(tp+2,:) = per_data.res(tp+2,:) + expt;
        end
        tp = tp + 4;
    end
end

per_data.res = per_data.res /(runs-1);
sum_mat = zeros(1,24);
for i = 1:24
    sum_mat(i) = sum(per_data.res(i,:));
end

x = [3,4,5,6,7,8];
y1 = zeros(1,6);
y2 = zeros(1,6);
y3 = zeros(1,6);
yy1 = 1;
for i = 1:6
    y1(yy1) = sum_mat(4*(i-1)+1);
    y2(yy1) = sum_mat(4*(i-1)+2);
    y3(yy1) = sum_mat(4*(i-1)+3);
    yy1 = yy1+ 1;
end

figure()
semilogy(x,y1,':','linewidth',2,'Marker','o'), hold on
semilogy(x,y2,'--','linewidth',2,'Marker','o')
semilogy(x,y3,'-.','linewidth',2,'Marker','o')


ylabel("Time (secs)")
xlabel("d")
set(gca, 'XTick', 3:8)
l = legend("QR Imp", "NE","QR Exp");
l.Location = 'northwest';
set(gca,'fontsize',16);
set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);
saveas(gcf,'Fig_1.pdf')
%%
%%% Figure 2
bd_mat = zeros(18,5);
j = 1;
for i = 1:24
    if i ~= 3
        if i  ~= 7
            if i ~= 11
                if i ~= 15
                    if i~= 19
                        if i~= 23
                            bd_mat(j,:) = per_data.res(i,:);
                            j = j +  1;
                        end
                    end
                end
            end
        end
    end                 
end

figure()
a=gca;
bar([bd_mat(1:3,:); bd_mat(4:6,:); bd_mat(7:9,:);bd_mat(10:12,:);bd_mat(13:15,:);bd_mat(16:18,:)],'stacked');
ylabel('Time (secs)')

xlabel('d')
xticks(0:18);
xticklabels({'','QR Imp','NE','','QR Imp','NE','','QR Imp','NE','','QR Imp','NE','','QR Imp','NE','','QR Imp','NE'});
a.XRuler.TickLabelGapOffset = 15;   
a.YRuler.TickLabelGapOffset = 15;
a.XTickLabelRotation = 90;
v = -0.00009;
text(1,v,'d=3','fontsize',12)
text(4,v,'d=4','fontsize',12)
text(7,v,'d=5','fontsize',12)
text(10,v,'d=6','fontsize',12)
text(13,v,'d=7','fontsize',12)
text(16,v,'d=8','fontsize',12)

l = legend('$Q^\top$B/$A^\top$B','QR/Gram','Pairwise Apply QR','Pairwise QR','Back solve/NE solve','Interpreter','latex');
l.Direction = 'reverse';
l.Location = 'northwest';
set(gca,'fontsize',14);
set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);
saveas(gcf,'Fig_2.pdf')
