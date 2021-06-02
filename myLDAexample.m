clear variables
clc
% change load and save path all over the document please.
%                v <-- this is a back slash (\)
path = "C:\Users\fuge\Google Drive\";
sub = "kw"; % faz fce fcg fcj kw
load(strcat(path, "GoogleColab\AMUSE_", sub, "\bf2_X3D.mat"))
load(strcat(path, "GoogleColab\AMUSE_", sub, "\b_Y.mat"))
load(strcat(path, "GoogleColab\AMUSE_", sub, "\bf2_X3D_val.mat"))
load(strcat(path, "GoogleColab\AMUSE_", sub, "\b_Y_val.mat"))

%%
clearvars -except b_X3D_train b_X3D_val b_Y_train b_Y_val

%%
sampling_division = 8; % <-- downsample as you wish (if you don't, good luck)
PEnter = 0.1; % <-- CHANGE
PRemove = 0.15; % <-- CHANGE

% put all channels on one line
b_X_train = reshape(b_X3D_train, size(b_X3D_train,1), []);
b_X_val = reshape(b_X3D_val, size(b_X3D_val,1), []);

% resample the signal
k = 1:size(b_X_train,2)/sampling_division;
k = k*sampling_division;

% FIT THE SWLDA (STEPWISE METHOD for feature selection)
[b,se,pval,finalmodel,stats,nextstep,history] = stepwisefit(b_X_train(:,k), ...
    b_Y_train, 'PEnter',PEnter, 'PRemove',PRemove, 'Scale','on');
% get the indeces selected by the stepwise fit
true_indeces = find(finalmodel);
k = k(true_indeces);

% FIT THE LDA
W = myLDA(b_X_train(:,k), b_Y_train);

% calculate the llkelihood, then the probability using the logistic funct
% on the training set
L_train = [ones(size(b_Y_train)), b_X_train(:,k)] * W';
P_train = exp(L_train) ./ repmat(sum(exp(L_train),2),[1 2]);
% separate at P = 0.5
[argvalue_train, argmax_train] = max(P_train,[],2);
argmax_train = argmax_train - 1;
% confusion matrix and accuracy
C_train = confusionmat(b_Y_train,argmax_train);
acc_train = sum(diag(C_train))/sum(C_train,'all');

% calculate the llkelihood, then the probability using the logistic funct
% on the test set
L_val = [ones(size(b_Y_val)), b_X_val(:,k)] * W';
P_val = exp(L_val) ./ repmat(sum(exp(L_val),2),[1 2]);
% separate at P = 0.5
[argvalue_val, argmax_val] = max(P_val,[],2);
argmax_val = argmax_val - 1;
% confusion matrix and accuracy
C_val = confusionmat(b_Y_val,argmax_val);
acc_val = sum(diag(C_val))/sum(C_val,'all');

%% This histogram should represent the valuable channels
histogram(k, 45)