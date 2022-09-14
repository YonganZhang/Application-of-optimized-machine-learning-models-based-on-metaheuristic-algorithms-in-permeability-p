%% ��Ӧ�Ⱥ������Իع�mse��Ϊ��Ӧ��ֵ
function [fitness] = fun(x,Tn_train,Pn_train)
    cmd = ['-c ', num2str(x(1)), ' -g ', num2str(x(2)) , ' -s 3 -p 0.01'];%���볬����
    model = libsvmtrain(Tn_train',Pn_train',cmd);
   [predict,mse,~] = libsvmpredict(Tn_train',Pn_train',model);
    if size(mse,1) == 0 
        fitness = 1;
    else
        fitness = mse(2);
    end
end
%%
% LIBSVMѵ��ʱ����ѡ��Ĳ����ܶ࣬������
% 
% -s svm���ͣ�SVM�������ͣ�Ĭ��0)
% ��������0 �� C-SVC�� 1 �Cv-SVC�� 2 �C һ��SVM�� 3 �� e-SVR�� 4 �� v-SVR
% -t �˺������ͣ��˺����������ͣ�Ĭ��2��
% ��������0 �C ���Ժ˺�����u��v 
% ��������1 �C ����ʽ�˺�������r*u��v + coef0)^degree
% ��������2 �C RBF(�����)�˺�����exp(-r|u-v|^2��
% ��������3 �C sigmoid�˺�����tanh(r*u��v + coef0)
% -d degree���˺����е�degree���ã���Զ���ʽ�˺�������Ĭ��3��
% -g r(gamma�����˺����е�gamma�������ã���Զ���ʽ/rbf/sigmoid�˺�������Ĭ��1/k��kΪ�������)
% -r coef0���˺����е�coef0���ã���Զ���ʽ/sigmoid�˺���������Ĭ��0)
% -c cost������C-SVC��e -SVR��v-SVR�Ĳ�������ʧ��������Ĭ��1��
% -n nu������v-SVC��һ��SVM��v- SVR�Ĳ�����Ĭ��0.5��
% -p p������e -SVR ����ʧ����p��ֵ��Ĭ��0.1��
% -m cachesize������cache�ڴ��С����MBΪ��λ��Ĭ��40��
% -e eps�������������ֹ�оݣ�Ĭ��0.001��
% -h shrinking���Ƿ�ʹ������ʽ��0��1��Ĭ��1��
% -wi weight�����õڼ���Ĳ���CΪweight*C (C-SVC�е�C) ��Ĭ��1��
% -v n: n-fold��������ģʽ��nΪfold�ĸ�����������ڵ���2
%% libsvmpredict�����ķ���ֵ
% �������ȷ�ʡ��ع�ľ��������ع��ƽ�����ϵ��������ȡ���ǵڶ���