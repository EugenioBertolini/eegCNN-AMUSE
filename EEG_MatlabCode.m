clear variables
clc
% change load and save path all over the document please.
%                v <-- this is a back slash (\)
path = "blablabla\";
load(strcat(path, "GoogleColab\AMUSE_VPfaz.mat"))
X = data{1,1}.X;
meanX = mean(X);
Xdetrended = X - meanX; % remove the mean ch by ch
ch = 28; % choose Cz (the 28th channel of the EEG) for plotting
fs = 250;

%% FILTER DESIGN 2
% filterDesigner
Fpass = 40;          % Passband Frequency
Fstop = 50;          % Stopband Frequency
Apass = 0.0001;     % Passband Ripple (dB)
Astop = 100;          % Stopband Attenuation (dB)

% Butter
match = 'passband';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdButter = design(h, 'butter', 'MatchExactly', match);

% Cheby1
match = 'passband';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdCheby1 = design(h, 'cheby1', 'MatchExactly', match);

% Cheby2
match = 'stopband';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdCheby2 = design(h, 'cheby2', 'MatchExactly', match);

% Ellip
match = 'both';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdEllip = design(h, 'ellip', 'MatchExactly', match);

% Firpm
Dstop = 10^(-Astop/20);    % Stopband Attenuation
% dens  = 20;              % Density Factor b  = firpm(N, Fo, Ao, W, {dens});
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(fs/2), [1 0], [Apass, Dstop]);
HdFirpm = dfilt.dffir(bf);


%%%%%%%%%%%%%%%%%
[bb,ab] = sos2tf(HdButter.sosMatrix,HdButter.ScaleValues);
X_b = filtfilt(bb, ab, Xdetrended);

[b1,a1] = sos2tf(HdCheby1.sosMatrix,HdCheby1.ScaleValues);
X_1 = filtfilt(b1, a1, Xdetrended);

[b2,a2] = sos2tf(HdCheby2.sosMatrix,HdCheby2.ScaleValues);
X_2 = filtfilt(b2, a2, Xdetrended);

[be,ae] = sos2tf(HdEllip.sosMatrix,HdEllip.ScaleValues);
X_e = filtfilt(be, ae, Xdetrended);

bf  = firpm(N, Fo, Ao, W);
X_f = filtfilt(bf, 1, Xdetrended);

% save the filter's coeff
save(strcat(path, "GoogleColab\coeffFilt", ...
    num2str(-20*log10(Apass)), num2str(Astop),".mat"), 'ab', 'a1', ...
    'a2', 'ae', 'bb', 'b1', 'b2', 'be', 'bf')

%% Plot in frequency domain
% fvtool(bb,ab, b1,a1, b2,a2, be,ae, bf,1)
% fvtool(bb,ab)
% fvtool(b1,a1)
% fvtool(b2,a2)
% fvtool(be,ae)
% fvtool(bf,1)
% 
% figure(4)
% tiledlayout('flow', 'TileSpacing','none', 'Padding','none')
% 
% nexttile, hold on
% plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
% plot(linspace(1,250,1024), mag2db(abs(fft(X_b(:, ch)',1024))))
% plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
% plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
% xlim([1 125])
% ylim([20, 80])
% title("butter")
% 
% nexttile, hold on
% plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
% plot(linspace(1,250,1024), mag2db(abs(fft(X_1(:, ch)',1024))))
% plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
% plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
% xlim([1 125])
% ylim([20, 80])
% title("cheb1")
% 
% nexttile, hold on
% plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
% plot(linspace(1,250,1024), mag2db(abs(fft(X_2(:, ch)',1024))))
% plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
% plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
% xlim([1 125])
% ylim([20, 80])
% title("cheb2")
% 
% nexttile, hold on
% plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
% plot(linspace(1,250,1024), mag2db(abs(fft(X_e(:, ch)',1024))))
% plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
% plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
% xlim([1 125])
% ylim([20, 80])
% title("ellip")
% 
% nexttile, hold on
% plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
% plot(linspace(1,250,1024), mag2db(abs(fft(X_f(:, ch)',1024))))
% plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
% plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
% xlim([1 125])
% ylim([20, 80])
% title("firpm")

%% PLOT SIGNAL IN TIME domain
% figure(6)
% tiledlayout('flow', 'TileSpacing','none', 'Padding','none')
% visual = [1,100] + 5000;
% 
% nexttile, hold on
% plot(Xdetrended(:, ch))
% plot(X_b(:, ch), '.-.', 'LineWidth',4)
% plot(X_1(:, ch), '.-.', 'LineWidth',3)
% plot(X_2(:, ch), '.-.', 'LineWidth',2)
% plot(X_e(:, ch), '.-.', 'LineWidth',1.5)
% plot(X_f(:, ch), '.-.')
% xlim(visual)
% legend("NORMAL","butter","cheb1","cheb2","ellip","firpm")
% 
% title(strcat("Fpass = ",num2str(Fpass), " | Fstop = ",num2str(Fstop), ...
%     " | Ripple = ",num2str(Apass), " | Attenuation = ",num2str(Astop), "dB"))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% n0 = normalized version               baseline
% n1 = normalized version               normalize([],[],3)
% f0 = filtered version                 (butter) with n0
% f1 = filtered version                 (chebichev 1) with n0
% f2 = filtered version                 (chebichev 2) with n0
% f3 = filtered version                 (elliptic) with n0
% f4 = filtered version                 (firpm) with n0
% filter without norm --> f0nNO, f1nNO, ...
% filter with n1 --> f0n1, f1n1, ...
clear variables
clc
close all

% choose the subjects
subje = ["faz"; "fce"; "fcg"; "fcj"; "kw"];
% choose which filters to apply
versio = ["n0"; "f0"; "f1"; "f2"; "f3"; "f4"];
verbose = 0;
for subsub = 1:5
    for verver = 1:6
        verbose = verbose + 1
        vers = versio(verver);
        
        % choose the order of the filters (you generated them with the
        % previous piece of code. Make sure that Apass=0.0001, Astop=100 if
        % you want to use this load.
        load(strcat(path, "GoogleColab\coeffFilt80100.mat"))
        % for this commented one: Apass=0.001, Astop=40.
        % load("C:\Users\fuge\Google Drive\GoogleColab\coeffFilt6040.mat")

        load(strcat(path, "GoogleColab\AMUSE_VP", subje(subsub), ".mat"))

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MULTICLASS
        sample_train = size(data{1,1}.y, 2)/6;
        sample_val = size(data{1,2}.y, 2)/6;
        T = 384; % time-length sample
        base_T = 44 - 1; % time samples to average and subtract from the main samples
        chh = [6:41, 43:51]; % choosen channels
        C = length(chh);

        % TRAIN X MULTICLASS
        X1 = data{1,1}.X;
        meanX = mean(X1);
        Xdetrended = X1 - meanX;
        tStim = data{1,1}.trial;
        
        % choose the filtering method
        if strcmp(vers, 'f0')
            X = filtfilt(bb, ab, Xdetrended);
        elseif strcmp(vers, 'f1')
            X = filtfilt(b1, a1, Xdetrended);
        elseif strcmp(vers, 'f2')
            X = filtfilt(b2, a2, Xdetrended);
        elseif strcmp(vers, 'f3')
            X = filtfilt(be, ae, Xdetrended);
        elseif strcmp(vers, 'f4')
            X = filtfilt(bf, 1, Xdetrended);
        else
            X = Xdetrended;
        end

        % cut the signals and order it in a matrix (samples, C, T)
        m_X3D_train = zeros(sample_train, C, T);
        m_X3D_train_baseline = zeros(sample_train, C, base_T + 1);
        for samp = 1 : sample_train
            idx_samp = 1 : T;
            m_X3D_train(samp, :, :) = X(idx_samp + tStim((samp-1)*6 + 1), chh)';

            idx_samp = -base_T : 0;
            m_X3D_train_baseline(samp, :, :) = X(idx_samp + tStim((samp-1)*6 + 1), chh)';
        end
        
        % choose the normalization method: n0 is the standard for filtered
        % signals        
        if strcmp(vers, 'n1')
            m_X3D_train = normalize(m_X3D_train, 3);
        else
            baseline(:,:,1) = mean(m_X3D_train_baseline, 3);
            m_X3D_train = m_X3D_train - baseline;
        end

        clearvars baseline m_X3D_train_baseline

        % VAL X MULTICLASS
        X1 = data{1,2}.X;
        meanX = mean(X1);
        Xdetrended = X1 - meanX;
        tStim = data{1,2}.trial;

        % choose the filtering method
        if strcmp(vers, 'f0')
            X = filtfilt(bb, ab, Xdetrended);
        elseif strcmp(vers, 'f1')
            X = filtfilt(b1, a1, Xdetrended);
        elseif strcmp(vers, 'f2')
            X = filtfilt(b2, a2, Xdetrended);
        elseif strcmp(vers, 'f3')
            X = filtfilt(be, ae, Xdetrended);
        elseif strcmp(vers, 'f4')
            X = filtfilt(bf, 1, Xdetrended);
        else
            X = Xdetrended;
        end

        % cut the signals and order it in a matrix (samples, C, T)
        m_X3D_val = zeros(sample_val, C, T);
        m_X3D_val_baseline = zeros(sample_val, C, base_T + 1);
        for samp = 1 : sample_val
            idx_samp = 1 : T;
            m_X3D_val(samp, :, :) = X(idx_samp + tStim((samp-1)*6 + 1), chh)';

            idx_samp = -base_T : 0;
            m_X3D_val_baseline(samp, :, :) = X(idx_samp + tStim((samp-1)*6 + 1), chh)';
        end

        % choose the normalization method: n0 is the standard for filtered
        % signals        
        if strcmp(vers, 'n1')
            m_X3D_val = normalize(m_X3D_val, 3);
        else
            baseline(:,:,1) = mean(m_X3D_val_baseline, 3);
            m_X3D_val = m_X3D_val - baseline;
        end

        clearvars baseline m_X3D_val_baseline

        % SAVE
        save(strcat(path, "GoogleColab\AMUSE_", subje(subsub), ...
            "\m", vers, "_X3D.mat"), 'm_X3D_train')
        save(strcat(path, 'GoogleColab\AMUSE_', subje(subsub), ...
            "\m", vers, "_X3D_val.mat"), 'm_X3D_val')




        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % BINARY
        sample_train = size(data{1,1}.y, 2);
        sample_val = size(data{1,2}.y, 2);
        
        T = 256; % time-length sample
        base_T = 44 - 1; % time samples to average and subtract from the main samples
        
        % TRAIN X BINARY
        X1 = data{1,1}.X;
        meanX = mean(X1);
        Xdetrended = X1 - meanX;
        tStim = data{1,1}.trial;

        % choose the filtering method
        if strcmp(vers, 'f0')
            X = filtfilt(bb, ab, Xdetrended);
        elseif strcmp(vers, 'f1')
            X = filtfilt(b1, a1, Xdetrended);
        elseif strcmp(vers, 'f2')
            X = filtfilt(b2, a2, Xdetrended);
        elseif strcmp(vers, 'f3')
            X = filtfilt(be, ae, Xdetrended);
        elseif strcmp(vers, 'f4')
            X = filtfilt(bf, 1, Xdetrended);
        else
            X = Xdetrended;
        end

        % cut the signals and order it in a matrix (samples, C, T)
        b_X3D_train = zeros(sample_train, C, T);
        b_X3D_train_baseline = zeros(sample_train, C, base_T + 1);
        for samp = 1 : sample_train
            idx_samp = 1 : T;
            b_X3D_train(samp, :, :) = X(idx_samp + tStim(samp), chh)';

            idx_samp = -base_T : 0;
            b_X3D_train_baseline(samp, :, :) = X(idx_samp + tStim(samp), chh)';
        end
        
        % choose the normalization method: n0 is the standard for filtered
        % signals        
        if strcmp(vers, 'n1')
            b_X3D_train = normalize(b_X3D_train, 3);
        else
            baseline(:,:,1) = mean(b_X3D_train_baseline, 3);
            b_X3D_train = b_X3D_train - baseline;
        end

        clearvars baseline b_X3D_train_baseline

        % VAL X BINARY
        X1 = data{1,2}.X;
        meanX = mean(X1);
        Xdetrended = X1 - meanX;
        tStim = data{1,2}.trial;

        % choose the filtering method
        if strcmp(vers, 'f0')
            X = filtfilt(bb, ab, Xdetrended);
        elseif strcmp(vers, 'f1')
            X = filtfilt(b1, a1, Xdetrended);
        elseif strcmp(vers, 'f2')
            X = filtfilt(b2, a2, Xdetrended);
        elseif strcmp(vers, 'f3')
            X = filtfilt(be, ae, Xdetrended);
        elseif strcmp(vers, 'f4')
            X = filtfilt(bf, 1, Xdetrended);
        else
            X = Xdetrended;
        end

        % cut the signals and order it in a matrix (samples, C, T)
        b_X3D_val = zeros(sample_val, C, T);
        b_X3D_val_baseline = zeros(sample_val, C, base_T + 1);
        for samp = 1 : sample_val
            idx_samp = 1 : T;
            b_X3D_val(samp, :, :) = X(idx_samp + tStim(samp), chh)';

            idx_samp = -base_T : 0;
            b_X3D_val_baseline(samp, :, :) = X(idx_samp + tStim(samp), chh)';
        end

        % choose the normalization method: n0 is the standard for filtered
        % signals
        if strcmp(vers, 'n1')
            b_X3D_val = normalize(b_X3D_val, 3);
        else
            baseline(:,:,1) = mean(b_X3D_val_baseline, 3);
            b_X3D_val = b_X3D_val - baseline;
        end

        clearvars baseline b_X3D_val_baseline

        % SAVE
        save(strcat(path, "GoogleColab\AMUSE_", subje(subsub), ...
            "\b", vers, "_X3D.mat"), 'b_X3D_train')
        save(strcat(path, "GoogleColab\AMUSE_", subje(subsub), ...
            "\b", vers, "_X3D_val.mat"), 'b_X3D_val')

        clearvars -except subje versio subsub verver verbose

    end
end