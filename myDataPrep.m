clear variables
clc

%% LOAD EEG FOR VISUALIZATION
% change load and save path all over the document please.
%         v <-- this is a back slash (\)
path = "C:\Users\fuge\Google Drive\";
load(strcat(path, "GoogleColab\AMUSE_VPfaz.mat"))
X = data{1,1}.X; % training eeg matrix
meanX = mean(X);
Xdetrended = X - meanX; % remove the mean ch by ch
ch = 28; % choose Cz (the 28th channel of the EEG) for visualization
fs = 250;

%% FILTER DESIGN (filterDesigner) [different filters, same caracteristics]
Fpass = 40;          % Passband Frequency
Fstop = 50;          % Stopband Frequency
Apass = 0.0001;      % Passband Ripple
Astop = 80;          % Stopband Attenuation (dB)

% Butter
match = 'passband';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdButter = design(h, 'butter', 'MatchExactly', match);
[bb,ab] = sos2tf(HdButter.sosMatrix,HdButter.ScaleValues);

% Cheby1
match = 'passband';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdCheby1 = design(h, 'cheby1', 'MatchExactly', match);
[b1,a1] = sos2tf(HdCheby1.sosMatrix,HdCheby1.ScaleValues);

% Cheby2
match = 'stopband';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdCheby2 = design(h, 'cheby2', 'MatchExactly', match);
[b2,a2] = sos2tf(HdCheby2.sosMatrix,HdCheby2.ScaleValues);

% Ellip
match = 'both';      % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdEllip = design(h, 'ellip', 'MatchExactly', match);
[be,ae] = sos2tf(HdEllip.sosMatrix,HdEllip.ScaleValues);

% Firpm
% dens = 20;         % Density Factor b  = firpm(N, Fo, Ao, W, {dens});
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(fs/2), [1 0], [Apass, 10^(-Astop/20)]);
bf  = firpm(N, Fo, Ao, W);
HdFirpm = dfilt.dffir(bf);

%% Save the filters' coefficients
save(strcat(path, "GoogleColab\coeffFilt", ...
    num2str(-20*log10(Apass)), num2str(Astop),".mat"), ...
    'ab', 'a1', 'a2', 'ae', 'bb', 'b1', 'b2', 'be', 'bf')

%% Plot in frequency domain
fvtool(bb,ab)
fvtool(b1,a1)
fvtool(b2,a2)
fvtool(be,ae)
fvtool(bf,1)

%%
X_b = filtfilt(bb, ab, Xdetrended);
X_1 = filtfilt(b1, a1, Xdetrended);
X_2 = filtfilt(b2, a2, Xdetrended);
X_e = filtfilt(be, ae, Xdetrended);
X_f = filtfilt(bf, 1, Xdetrended);

%%
figure(4)
tiledlayout('flow', 'TileSpacing','tight', 'Padding','tight')

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_b(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("butter")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_1(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheb1")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_2(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheb2")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_e(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("ellip")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_f(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("firpm")

%% PLOT SIGNAL IN TIME domain
figure(6)
tiledlayout('flow', 'TileSpacing','tight', 'Padding','none')
visual1 = [1,256] + 5000; % range of visualization
visual2 = [1,45] + 10000; % range of visualization

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_b(:, ch), '.-.', 'LineWidth',4)
plot(X_1(:, ch), '.-.', 'LineWidth',3)
plot(X_2(:, ch), '.-.', 'LineWidth',2)
plot(X_e(:, ch), '.-.', 'LineWidth',1.5)
plot(X_f(:, ch), '.-.')
xlim(visual1)
legend("NORMAL","butter","cheb1","cheb2","ellip","firpm")
title(strcat("Fpass = ",num2str(Fpass), " | Fstop = ",num2str(Fstop), ...
    " | Ripple = ",num2str(Apass), " | Attenuation = ",num2str(Astop), "dB"))

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_b(:, ch), '.-.', 'LineWidth',4)
plot(X_1(:, ch), '.-.', 'LineWidth',3)
plot(X_2(:, ch), '.-.', 'LineWidth',2)
plot(X_e(:, ch), '.-.', 'LineWidth',1.5)
plot(X_f(:, ch), '.-.')
xlim(visual2)
legend("NORMAL","butter","cheb1","cheb2","ellip","firpm")
title(strcat("Fpass = ",num2str(Fpass), " | Fstop = ",num2str(Fstop), ...
    " | Ripple = ",num2str(Apass), " | Attenuation = ",num2str(Astop), "dB"))

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_b(:, ch), '.-.', 'LineWidth',4)
plot(X_1(:, ch), '.-.', 'LineWidth',3)
plot(X_2(:, ch), '.-.', 'LineWidth',2)
plot(X_e(:, ch), '.-.', 'LineWidth',1.5)
plot(X_f(:, ch), '.-.')
xlim(visual1+8000)
legend("NORMAL","butter","cheb1","cheb2","ellip","firpm")
title(strcat("Fpass = ",num2str(Fpass), " | Fstop = ",num2str(Fstop), ...
    " | Ripple = ",num2str(Apass), " | Attenuation = ",num2str(Astop), "dB"))

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_b(:, ch), '.-.', 'LineWidth',4)
plot(X_1(:, ch), '.-.', 'LineWidth',3)
plot(X_2(:, ch), '.-.', 'LineWidth',2)
plot(X_e(:, ch), '.-.', 'LineWidth',1.5)
plot(X_f(:, ch), '.-.')
xlim(visual2+8000)
legend("NORMAL","butter","cheb1","cheb2","ellip","firpm")
title(strcat("Fpass = ",num2str(Fpass), " | Fstop = ",num2str(Fstop), ...
    " | Ripple = ",num2str(Apass), " | Attenuation = ",num2str(Astop), "dB"))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% |/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\
clear variables
clc

%% FILTER DESIGN 2 (filterDesigner) [same filters, different order]
Fpass = 40;          % Passband Frequency
Fstop = 50;          % Stopband Frequency
filtType = 'cheby2'; % Filter Type (if you change it, watch out)

% Cheby2
Apass = 0.0001;      % Passband Ripple
Astop = 80;          % Stopband Attenuation (dB)
match = 'stopband';  % Band to match exactly
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, fs);
HdCheby2 = design(h, filtType, 'MatchExactly', match);
[b2,a2] = sos2tf(HdCheby2.sosMatrix,HdCheby2.ScaleValues);

% Cheby2 ord3
Astop = 20;  % Stopband Attenuation (dB)
h  = fdesign.lowpass('N,Fst,Ast', 3, Fstop, Astop, fs);
HdCheby3 = design(h, filtType);
[b3,a3] = sos2tf(HdCheby3.sosMatrix,HdCheby3.ScaleValues);

% Cheby2 ord4
Astop = 30;  % Stopband Attenuation (dB)
h  = fdesign.lowpass('N,Fst,Ast', 4, Fstop, Astop, fs);
HdCheby4 = design(h, filtType);
[b4,a4] = sos2tf(HdCheby4.sosMatrix,HdCheby4.ScaleValues);

% Cheby2 ord5
Astop = 40;  % Stopband Attenuation (dB)
h  = fdesign.lowpass('N,Fst,Ast', 5, Fstop, Astop, fs);
HdCheby5 = design(h, filtType);
[b5,a5] = sos2tf(HdCheby5.sosMatrix,HdCheby5.ScaleValues);

% Cheby2 ord7
Astop = 60;  % Stopband Attenuation (dB)
h  = fdesign.lowpass('N,Fst,Ast', 7, Fstop, Astop, fs);
HdCheby7 = design(h, filtType);
[b7,a7] = sos2tf(HdCheby7.sosMatrix,HdCheby7.ScaleValues);

% Cheby2 ord10
Astop = 60;  % Stopband Attenuation (dB)
h  = fdesign.lowpass('N,Fst,Ast', 10, Fstop, Astop, fs);
HdCheby10 = design(h, filtType);
[b10,a10] = sos2tf(HdCheby10.sosMatrix,HdCheby10.ScaleValues);

% Cheby2 ord15
Astop = 80;  % Stopband Attenuation (dB)
h  = fdesign.lowpass('N,Fst,Ast', 15, Fstop, Astop, fs);
HdCheby15 = design(h, filtType);
[b15,a15] = sos2tf(HdCheby15.sosMatrix,HdCheby15.ScaleValues);

% Cheby2 ord20 (= original Cheby2)
Astop = 80;  % Stopband Attenuation (dB)
h  = fdesign.lowpass('N,Fst,Ast', 20, Fstop, Astop, fs);
HdCheby20 = design(h, filtType);
[b20,a20] = sos2tf(HdCheby20.sosMatrix,HdCheby20.ScaleValues);

%% Save the filters' coefficients
save(strcat(path, "GoogleColab\coeffFilt3457.mat"), ...
    'a3', 'a4', 'a5', 'a7', 'a10', 'a15', 'a20', ...
    'b3', 'b4', 'b5', 'b7', 'b10', 'b15', 'b20')

%% Plot in frequency domain
fvtool(b2,a2)
fvtool(b3,a3)
fvtool(b4,a4)
fvtool(b5,a5)
fvtool(b7,a7)
fvtool(b10,a10)
fvtool(b15,a15)
fvtool(b20,a20)

%%
X_2 = filtfilt(b2, a2, Xdetrended);
X_3 = filtfilt(b3, a3, Xdetrended);
X_4 = filtfilt(b4, a4, Xdetrended);
X_5 = filtfilt(b5, a5, Xdetrended);
X_7 = filtfilt(b7, a7, Xdetrended);
X_10 = filtfilt(b10, a10, Xdetrended);
X_15 = filtfilt(b15, a15, Xdetrended);
X_20 = filtfilt(b20, a20, Xdetrended);

%%
figure(4)
tiledlayout(2, 4, 'TileSpacing','tight', 'Padding','tight')

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_2(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 8080")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_3(:, ch)',1024))))
plot([Fpass-10, Fpass-10] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 ord03")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_4(:, ch)',1024))))
plot([Fpass-10, Fpass-10] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 ord04")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_5(:, ch)',1024))))
plot([Fpass-10, Fpass-10] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 ord05")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_7(:, ch)',1024))))
plot([Fpass-10, Fpass-10] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 ord07")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_10(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 ord10")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_15(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 ord15")

nexttile, hold on
plot(linspace(1,250,1024), mag2db(abs(fft(Xdetrended(:, ch)',1024))))
plot(linspace(1,250,1024), mag2db(abs(fft(X_20(:, ch)',1024))))
plot([Fpass, Fpass] ,[-5, 100], '.-.','LineWidth', 2)
plot([Fstop, Fstop] ,[-5, 100], 'r.-.', 'LineWidth', 1)
xlim([1 125])
ylim([20, 80])
xlabel("frequency (Hz)")
ylabel("Magnitude (dB)")
title("cheby2 ord20")

%% PLOT SIGNAL IN TIME domain
figure(6)
tiledlayout('flow', 'TileSpacing','tight', 'Padding','none')
visual1 = [1,256] + 5000;
visual2 = [1,45] + 10000;

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_3(:, ch), '.-.', 'LineWidth',4)
plot(X_4(:, ch), '.-.', 'LineWidth',3)
plot(X_5(:, ch), '.-.', 'LineWidth',2)
plot(X_7(:, ch), '.-.', 'LineWidth',1.5)
plot(X_10(:, ch), '.-.')
plot(X_15(:, ch), '.-.')
plot(X_20(:, ch), '.-.')
xlim(visual1)
legend("NORMAL","ord3","ord4","ord5","ord7","ord10","ord15","ord20")
title("Cheby2, different filters' orders")

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_3(:, ch), '.-.', 'LineWidth',4)
plot(X_4(:, ch), '.-.', 'LineWidth',3)
plot(X_5(:, ch), '.-.', 'LineWidth',2)
plot(X_7(:, ch), '.-.', 'LineWidth',1.5)
plot(X_10(:, ch), '.-.')
plot(X_15(:, ch), '.-.')
plot(X_20(:, ch), '.-.')
xlim(visual2)
legend("NORMAL","ord3","ord4","ord5","ord7","ord10","ord15","ord20")
title("Cheby2, different filters' orders")

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_3(:, ch), '.-.', 'LineWidth',4)
plot(X_4(:, ch), '.-.', 'LineWidth',3)
plot(X_5(:, ch), '.-.', 'LineWidth',2)
plot(X_7(:, ch), '.-.', 'LineWidth',1.5)
plot(X_10(:, ch), '.-.')
plot(X_15(:, ch), '.-.')
plot(X_20(:, ch), '.-.')
xlim(visual1+8000)
legend("NORMAL","ord3","ord4","ord5","ord7","ord10","ord15","ord20")
title("Cheby2, different filters' orders")

nexttile, hold on
plot(Xdetrended(:, ch))
plot(X_3(:, ch), '.-.', 'LineWidth',4)
plot(X_4(:, ch), '.-.', 'LineWidth',3)
plot(X_5(:, ch), '.-.', 'LineWidth',2)
plot(X_7(:, ch), '.-.', 'LineWidth',1.5)
plot(X_10(:, ch), '.-.')
plot(X_15(:, ch), '.-.')
plot(X_20(:, ch), '.-.')
xlim(visual2+8000)
legend("NORMAL","ord3","ord4","ord5","ord7","ord10","ord15","ord20")
title("Cheby2, different filters' orders")
