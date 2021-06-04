clear variables
clc
% change load and save path all over the document please.
%         v <-- this is a back slash (\)
path = "C:\Users\fuge\Google Drive\";
load(strcat(path, "GoogleColab\AMUSE_VPfaz.mat"))
load(strcat(path, "GoogleColab\coeffFilt8080.mat"))

X = data{1,1}.X; % training eeg matrix
meanX = mean(X);
Xdetrended = X - meanX; % remove the mean ch by ch
b_X2D_train = filtfilt(b2, a2, Xdetrended);
b_Y_train = bbci_mrk{1, 1}.y(1,:)';

X = data{1,2}.X; % test eeg matrix
meanX = mean(X);
Xdetrended = X - meanX; % remove the mean ch by ch
b_X2D_test = filtfilt(b2, a2, Xdetrended);
b_Y_test = bbci_mrk{1, 2}.y(1,:)';

tStim_train = data{1,1}.trial;
tStim_test = data{1,2}.trial;

sample_train = size(b_Y_train, 1);
sample_test = size(b_Y_test, 1);

T = 256;
idx_samp = 1 : T;

base_T = 44 - 1; % time samples to average and subtract from the main samples
idx_samp_base = -base_T : 0;

chh = [6:41, 43:51]; % choosen channels
C = length(chh);

% TRAIN cut the signals and order it in a matrix (samples, C, T)
b_X3D_train = zeros(sample_train, C, T);
b_X3D_train_baseline = zeros(sample_train, C, base_T + 1);
for samp = 1 : sample_train
    b_X3D_train(samp, :, :) = b_X2D_train(idx_samp + tStim_train(samp), chh)';
    b_X3D_train_baseline(samp, :, :) = b_X2D_train(idx_samp_base + tStim_train(samp), chh)';
end
baseline(:,:,1) = mean(b_X3D_train_baseline, 3);
b_X3D_train = b_X3D_train - baseline;

clearvars baseline

% TEST cut the signals and order it in a matrix (samples, C, T)
b_X3D_test = zeros(sample_test, C, T);
b_X3D_test_baseline = zeros(sample_test, C, base_T + 1);
for samp = 1 : sample_test
    b_X3D_test(samp, :, :) = b_X2D_test(idx_samp + tStim_test(samp), chh)';
    b_X3D_test_baseline(samp, :, :) = b_X2D_test(idx_samp_base + tStim_test(samp), chh)';
end
baseline(:,:,1) = mean(b_X3D_test_baseline, 3);
b_X3D_test = b_X3D_test - baseline;

%%
clearvars -except b_X3D_train b_X3D_test b_Y_train b_Y_test data

%%
sampling_division = 8; % <-- downsample as you wish (if you don't, good luck)
PEnter = 0.1; % <-- CHANGE
PRemove = 0.15; % <-- CHANGE

% put all channels on one line
b_X_train = reshape(b_X3D_train, size(b_X3D_train,1), []);
b_X_test = reshape(b_X3D_test, size(b_X3D_test,1), []);

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
L_test = [ones(size(b_Y_test)), b_X_test(:,k)] * W';
P_test = exp(L_test) ./ repmat(sum(exp(L_test),2),[1 2]);
% separate at P = 0.5
[argvalue_test, argmax_test] = max(P_test,[],2);
argmax_test = argmax_test - 1;
% confusion matrix and accuracy
C_test = confusionmat(b_Y_test,argmax_test);
acc_test = sum(diag(C_test))/sum(C_test,'all');

%%
clearvars -except b_X3D_train b_X3D_test b_Y_train b_Y_test data C_train C_test acc_train acc_test k

%% This histogram should represent the valuable channels
h = histogram(k, 45);
vect = h.BinLimits;
xtk = linspace(max(vect)/46/2, max(vect) - max(vect)/46/2, 45);
set(gca, 'xtick', xtk, 'xticklabel', data{1, 1}.channels, 'FontSize', 6)


