clear all 
clc
format compact;
warning off;

P_train = xlsread('data','training set','B2:R844')';T_train = xlsread('data','training set','S2:S844')';

P_test=xlsread('data','test set','B2:R215')';T_test=xlsread('data','test set','S2:S215')';

[Pn_train,inputps] = mapminmax(P_train,-1,1);
Pn_test = mapminmax('apply',P_test,inputps);

[Tn_train,outputps] = mapminmax(T_train,-1,1);
Tn_test = mapminmax('apply',T_test,outputps);



fobj = @(x) fun(x,Tn_train,Pn_train); 

dim = 2;

lb = [0,0];
ub = [100,100];


pop =10; 
Max_iteration=50;           

[Alpha_score,Alpha_pos,Convergence_curve]=GWO(pop,Max_iteration,lb,ub,dim,fobj); 
c = Alpha_pos(1, 1);  
g = Alpha_pos(1, 2); 
figure
plot(Convergence_curve,'linewidth',1.5);
grid on;


cmd = ['-c ', num2str(c), ' -g ', num2str(g) , ' -s 3 -p 0.01'];
model = libsvmtrain(Tn_train',Pn_train',cmd);

[predict,mse,~] = libsvmpredict(Tn_test',Pn_test',model);
test = mapminmax('reverse',predict',outputps);
% test = predict';
N1=length(T_test);
T_test1=T_test;
R2 = (N1*sum(test.*T_test1)-sum(test)*sum(T_test1))^2/((N1*sum((test).^2)-(sum(test))^2)*(N1*sum((T_test1).^2)-(sum(T_test1))^2)); 
figure
plot(1:N1,T_test,'b:*',1:N1,test,'r-o')
title(string)



