
function [fitness] = fun(x,Tn_train,Pn_train)
    cmd = ['-c ', num2str(x(1)), ' -g ', num2str(x(2)) , ' -s 3 -p 0.01'];
    model = libsvmtrain(Tn_train',Pn_train',cmd);
   [predict,mse,~] = libsvmpredict(Tn_train',Pn_train',model);
    if size(mse,1) == 0 
        fitness = 1;
    else
        fitness = mse(2);
    end
end