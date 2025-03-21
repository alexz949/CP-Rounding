clc
clear all
maxNumCompThreads(1);

addpath("./algorithms/");
addpath("./wrappers/")
addpath("./tensor_toolbox/")

%% initialize 
als_time = struct;
exp_qr_time = struct;
pw_qr_time = struct;

%% 5 way
d = 5;
r = 5;
n = [10000,15000,30000];
maxiter = 10;
tol = 1e-10;



als_time.r5 = zeros(length(n),5);
exp_qr_time.r5 = zeros(length(n),7);
pw_qr_time.r5 = zeros(length(n),7);

for i = 1:length(n)
    nk = n(i);
    T = sinsum_full(d,nk);
    
    % ranodmly initialize the core tensor
    Uinit = cell(d,1);
    for j = 1:d
        Uinit{j} = rand(nk,r);
    end
    
    % Perform CP roudning
    [Mals5,~,outals5]  = cp_als_time(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    [Mqr5,~,outqr5]    = cp_als_qr(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    [Mqr_pw5,~,outpw5] = cp_als_qr_new(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');

     % compute average iteration time
    exp_qr_time.r5(i,:) = mean(outqr5.times(2:end,:),1);
    pw_qr_time.r5(i,:)  = mean(outpw5.times(2:end,:),1);
    als_time.r5(i,:)    = mean(outals5.times(2:end,:),1);
end

%%% data prep
nals5 = zeros(3,6);
nqr5 = zeros(3,6);
npw5 = zeros(3,6);

for i = 1:3
    nals5(i,:) = [als_time.r5(i,1), als_time.r5(i,2),0,0,als_time.r5(i,3),(als_time.r5(i,4)+als_time.r5(i,5))];
    nqr5(i,:) = [exp_qr_time.r5(i,1),exp_qr_time.r5(i,2),exp_qr_time.r5(i,3),exp_qr_time.r5(i,4),exp_qr_time.r5(i,5),(exp_qr_time.r5(i,6)+exp_qr_time.r5(i,7))];
    npw5(i,:) = [pw_qr_time.r5(i,1),pw_qr_time.r5(i,2),pw_qr_time.r5(i,3),pw_qr_time.r5(i,4),pw_qr_time.r5(i,5),(pw_qr_time.r5(i,6)+pw_qr_time.r5(i,7))];
end

%% 3 Way
d = 3;
r = 3;
n = [10000,15000,30000];
maxiter = 20;
tol = 1e-10;


als_time.r3 = zeros(length(n),5);
exp_qr_time.r3 = zeros(length(n),7);
pw_qr_time.r3 = zeros(length(n),7);

for i = 1:length(n)
    nk = n(i);
    T = sinsum_full(d,nk);

    % ranodmly initialize the core tensor
    Uinit = cell(d,1);
    for j = 1:d
        Uinit{j} = rand(nk,r);
    end
    
    % Perform CP roudning
    [Mals3,~,outals3]  = cp_als_time(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    [Mqr3,~,outqr3]    = cp_als_qr(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    [Mqr_pw3,~,outpw3] = cp_als_qr_new(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');

     % compute average iteration time
    exp_qr_time.r3(i,:) = mean(outqr3.times(2:end,:),1);
    pw_qr_time.r3(i,:)  = mean(outpw3.times(2:end,:),1);
    als_time.r3(i,:)    = mean(outals3.times(2:end,:),1);
end

%%% data prep
nals3 = zeros(3,6);
nqr3 = zeros(3,6);
npw3 = zeros(3,6);
for i = 1:3
    nals3(i,:) = [als_time.r3(i,1), als_time.r3(i,2),0,0,als_time.r3(i,3),als_time.r3(i,4)+als_time.r3(i,5)];
    nqr3(i,:)  = [exp_qr_time.r3(i,1),exp_qr_time.r3(i,2),exp_qr_time.r3(i,3),exp_qr_time.r3(i,4),exp_qr_time.r3(i,5),(exp_qr_time.r3(i,6)+exp_qr_time.r3(i,7))];
    npw3(i,:)  = [pw_qr_time.r3(i,1),pw_qr_time.r3(i,2),pw_qr_time.r3(i,3),pw_qr_time.r3(i,4),pw_qr_time.r3(i,5),(pw_qr_time.r3(i,6)+pw_qr_time.r3(i,7))];
end


%% 7-way
d = 7;
r = 7;
n = [10000,15000,30000];
maxiter = 10;
tol = 1e-10;

als_time.r7 = zeros(length(n),5);
exp_qr_time.r7 = zeros(length(n),7);
pw_qr_time.r7 = zeros(length(n),7);

for i = 1:length(n)
    nk = n(i);
    T = sinsum_full(d,nk);
    
    % ranodmly initialize the core tensor
    Uinit = cell(d,1);
    for j = 1:d
        Uinit{j} = rand(nk,r);
    end
    
    % Perform CP roudning
    [Mals7,~,outals7]  = cp_als_time(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    [Mqr7,~,outqr7]    = cp_als_qr(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    [Mqr_pw7,~,outpw7] = cp_als_qr_new(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');

    % compute average iteration time
    exp_qr_time.r7(i,:) = mean(outqr7.times(2:end,:),1);
    pw_qr_time.r7(i,:)  = mean(outpw7.times(2:end,:),1);
    als_time.r7(i,:)    = mean(outals7.times(2:end,:),1);
end

%%% data prep
nals7 = zeros(3,6);
nqr7 = zeros(3,6);
npw7 = zeros(3,6);
for i = 1:3
    npw7(i,:) = [pw_qr_time.r7(i,1),pw_qr_time.r7(i,2),pw_qr_time.r7(i,3),pw_qr_time.r7(i,4),pw_qr_time.r7(i,5),(pw_qr_time.r7(i,6)+pw_qr_time.r7(i,7))];
    nals7(i,:) = [als_time.r7(i,1), als_time.r7(i,2),0,0,als_time.r7(i,3),als_time.r7(i,4)+als_time.r7(i,5)];
    nqr7(i,:) = [exp_qr_time.r7(i,1),exp_qr_time.r7(i,2),exp_qr_time.r7(i,3),exp_qr_time.r7(i,4),exp_qr_time.r7(i,5),(exp_qr_time.r7(i,6)+exp_qr_time.r7(i,7))];
end

%% 9-way
% generate tensor
d = 9;


r = 9;
n = [10000,15000,30000];
maxiter = 10;
tol = 1e-10;


als_time.r9 = zeros(length(n),5);
exp_qr_time.r9 = zeros(length(n),7);
pw_qr_time.r9 = zeros(length(n),7);

for i = 1:length(n)
    nk = n(i);
    T = sinsum_full(d,nk);
    
    % ranodmly initialize the core tensor
    Uinit = cell(d,1);
    for j = 1:d
        Uinit{j} = rand(nk,r);
    end

    % Perform CP rounding
    [Mals9,~,outals9]  = cp_als_time(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    %[Mqr9,~,outqr9]   = cp_als_qr(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');
    [Mqr_pw9,~,outpw9] = cp_als_qr_new(T,r,'init',Uinit,'maxiters',maxiter,'tol',tol,'printitn',1,'errmethod','fast');

    % compute average iteration time
    %tqr.r9(i,:) = mean(outqr9.times(2:end,:),1);
    pw_qr_time.r9(i,:) = mean(outpw9.times(2:end,:),1);
    als_time.r9(i,:) = mean(outals9.times(2:end,:),1);
end

%%% data prep
nals9 = zeros(3,6);
nqr9 = zeros(3,6);
npw9 = zeros(3,6);
for i = 1:3
    nals9(i,:) = [als_time.r9(i,1), als_time.r9(i,2),0,0,als_time.r9(i,3),als_time.r9(i,4)+als_time.r9(i,5)];
    %nqr9(i,:) = [tqr.r9(i,1),tqr.r9(i,2),tqr.r9(i,3),tqr.r9(i,4),tqr.r9(i,5),(tqr.r9(i,6)+tqr.r9(i,7))];
    nqr9(i,:) = [0,0,0,0,0,0];
    npw9(i,:) = [pw_qr_time.r9(i,1),pw_qr_time.r9(i,2),pw_qr_time.r9(i,3),pw_qr_time.r9(i,4),pw_qr_time.r9(i,5),(pw_qr_time.r9(i,6)+pw_qr_time.r9(i,7))];
end


%% prep data for ploting
sdata7 = [];
for i = 1:3
    sdata7 = [sdata7; npw7(i,:);nals7(i,:);nqr7(i,:); zeros(1,6)];
end

sdata3 = [];
for i = 1:3
    sdata3 = [sdata3; npw3(i,:);nals3(i,:);nqr3(i,:); zeros(1,6)];
end

sdata5 = [];
for i = 1:3
    sdata5 = [sdata5; npw5(i,:);nals5(i,:);nqr5(i,:); zeros(1,6)];
end

sdata9 = [];
for i = 1:3
    sdata9 = [sdata9; npw9(i,:); nals9(i,:);nqr9(i,:); zeros(1,6)];
end


kt_data = struct;
kt_data.sdata7 = sdata7;
kt_data.sdata5 = sdata5;
kt_data.sdata3 = sdata3;   
kt_data.sdata9 = sdata9;

%%% 9-way 
figure,
subplot(2,2,4)
bar(kt_data.sdata9(:,1:end-1),'stacked')
title('9-way')
a = gca;
xlabel('n')
ylabel('Time (secs)','FontSize',14)
xticks([0:12])
xticklabels({'','QR Imp','NE','','','QR Imp','NE','','','QR Imp','NE',''})
a.XRuler.TickLabelGapOffset = 15;   
a.YRuler.TickLabelGapOffset = 15;
a.XAxis.FontSize = 12;
a.XTickLabelRotation = 90;

v = -0.06;
text(1,v,'10000','fontsize',12)
text(5,v,'15000','fontsize',12)
text(9,v,'30000','fontsize',12)

     
%%% 7-way      
subplot(2,2,3)
bar(kt_data.sdata7(:,1:end-1),'stacked')
title('7-way')
a = gca;
xlabel('n')
ylabel('Time (secs)','FontSize',14)
xticks([0:12])
xticklabels({'','QR Imp','NE','QR Exp','','QR Imp','NE','QR Exp','','QR Imp','NE','QR Exp'})
a.XRuler.TickLabelGapOffset = 15;   
a.YRuler.TickLabelGapOffset = 15;
a.XTickLabelRotation = 90;
a.XAxis.FontSize = 12;
ylim([0 0.42])
v = -0.02;
text(1,v,'10000','fontsize',12)
text(5,v,'15000','fontsize',12)
text(9,v,'30000','fontsize',12)


%%% 3-way
subplot(2,2,1)
bar(kt_data.sdata3(:,1:end-1),'stacked')
title('3-way')
a = gca;
xlabel('n')
ylabel('Time (secs)','FontSize',14)
xticks([0:12])
ylim([0 0.0085])
xticklabels({'','QR Imp','NE','QR Exp','','QR Imp','NE','QR Exp','','QR Imp','NE','QR Exp'})
a.XRuler.TickLabelGapOffset = 15;   
a.YRuler.TickLabelGapOffset = 15;
a.XTickLabelRotation = 90;
a.XAxis.FontSize = 12;
v = -0.0005;
text(1,v,'10000','Fontsize',12)
text(5,v,'15000','Fontsize',12)
text(9,v,'30000','Fontsize',12)
l = legend('MTTKRP/TTM','Gram/QR','Pairwise QR','Apply Pairwise QR','Back solve/NE solve','Other');
l.FontSize = 5.8;
l.Location = 'northwest';


%%% 5-way
subplot(2,2,2)
bar(kt_data.sdata5(:,1:end-1),'stacked')
title('5-way')
a = gca;
xlabel('n')
ylabel('Time (secs)','FontSize',14)
%ylim([0 430])
xticks([0:12])
xticklabels({'','QR Imp','NE','QR Exp','','QR Imp','NE','QR Exp','','QR Imp','NE','QR Exp'})
a.XRuler.TickLabelGapOffset = 15;   
a.YRuler.TickLabelGapOffset = 15;
a.XTickLabelRotation = 90;
a.XAxis.FontSize = 12;
v = -0.002;
text(1,v,'10000','fontsize',12)
text(5,v,'15000','fontsize',12)
text(9,v,'30000','fontsize',12)
set(gcf,'Units','inches');
screenposition = get(gcf,'Position');
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);
saveas(gcf,'Fig_3.pdf')





