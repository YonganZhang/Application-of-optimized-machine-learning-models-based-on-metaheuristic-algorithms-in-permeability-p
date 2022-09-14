clear all 
clc



P_train = xlsread('data','training set','B2:R844')';T_train = xlsread('data','training set','S2:S844')';

P_test=xlsread('data','test set','B2:R215')';T_test=xlsread('data','test set','S2:S215')';

[inputn,inputps] = mapminmax(P_train,-1,1);
inputn_test = mapminmax('apply',P_test,inputps);

[outputn,outputps] = mapminmax(T_train,-1,1);
outputn_test = mapminmax('apply',T_test,outputps);

inputnum=size(P_train,1);
hiddennum=50;
outputnum=size(T_train,1);


net=newff(inputn,outputn,hiddennum);


dim=inputnum * hiddennum + hiddennum*outputnum + hiddennum + outputnum;
Max_iteration=50;  
pop=100;  
lb=-5;  
ub=5;   
fobj = @(x) fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);
[Leader_score,Leader_pos,Convergence_curve]=WOA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化
figure
plot(Convergence_curve,'linewidth',1.5);
grid on

x=Leader_pos;
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;


net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;


[net,tr]=train(net,inputn,outputn);


test=sim(net,inputn_test);
test = mapminmax('reverse',test,outputps);
N1=length(T_test);
R2 = (N1*sum(test.*T_test)-sum(test)*sum(T_test))^2/((N1*sum((test).^2)-(sum(test))^2)*(N1*sum((T_test).^2)-(sum(T_test))^2)); 
figure
plot(1:N1,T_test,'b:*',1:N1,test,'r-o')
title(string)

