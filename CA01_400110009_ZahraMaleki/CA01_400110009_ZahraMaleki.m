
clear;
clc;

%% 1.1 - do not run this cell, for recording only
fs = 40000; 
bitDepth = 16; 
duration = 10; 
recorder = audiorecorder(fs, bitDepth, 1);
%record
recordblocking(recorder, duration);
%to data
data = getaudiodata(recorder);
%save
save('D:\electrical eng\term 6\communication sys\CA01_400110009_ZahraMaleki\recorded_audio.mat', 'data', 'fs');
audiowrite('D:\electrical eng\term 6\communication sys\CA01_400110009_ZahraMaleki\recorded_audio.wav', data, fs);

%% use the recorded audio
fs = 40000; 
bitDepth = 16; 
duration = 10; 

data=audioread("D:\electrical eng\term 6\communication sys\CA01_400110009_ZahraMaleki\recorded_audio.wav");
%% 1.2
nfft = 1024;
fs = 40000; 
window = nfft; 
noverlap = 0; 

[s, f, t] = spectrogram(data, window, noverlap, nfft, fs);
figure;
spectrogram(data, window, noverlap, nfft, fs);
figure;
imagesc(t, f, 10*log10(abs(s)));
xlabel('t (s)');
title('spectrum');
colorbar.Label.String = 'dB';

%% 1.3
spectrogram(data, window, noverlap, nfft, fs, 'centered')
%% 1.4
windowTypes = {'rectwin', 'hann', 'hamming', 'blackman', 'gaussian'};
window = hann(nfft);
figure;
spectrogram(data, window, noverlap, nfft, fs, 'centered');
%% 1.5
%auto correlaion
autocorr_result = xcorr(data);
power = max(autocorr_result);
disp(['Power: ' num2str(power)]);
%% 1.6
nfft = 10000;
spectrogram(data, window, noverlap, nfft, fs, 'centered');
%% 1.7
normalized_data = data / sqrt(max(autocorr_result));
normalized_corr = xcorr(normalized_data);
normalized_power = max(normalized_corr);
disp(['normalized power= ', num2str(normalized_power)]);
soundsc(normalized_data,fs);
%% 1.8
fs_min = 35000;
nfft = 1024;
window = hann(nfft); 
noverlap = 0; 

spectrogram(data, window, noverlap, nfft, fs_min, 'centered');
%% 1.9
[y, Fs] = audioread("D:\electrical eng\term 6\communication sys\CA01_400110009_ZahraMaleki\Paper Plane.mp3");

samples = round(Fs * 10);
segment = y(1:samples, :);

%save
save('D:\electrical eng\term 6\communication sys\CA01_400110009_ZahraMaleki\segment.mat', 'segment', 'Fs');
audiowrite('D:\electrical eng\term 6\communication sys\CA01_400110009_ZahraMaleki\segment.wav', segment, Fs);

%choose channel
music=segment(:, 1);
nfft = 1024;
window = hann(nfft); 
noverlap = 0; 
figure;
spectrogram(music, window, noverlap, nfft, Fs, 'centered');

%down sampeling
autocorr_music = xcorr(music);
normalized_music = music / sqrt(max(autocorr_music));

music_corr = xcorr(normalized_music);
music_power = max(music_corr);
disp(['normalized power= ', num2str(music_power)]);
soundsc(normalized_music,Fs);

Fs_min = 41000;
figure;
spectrogram(normalized_music, window, noverlap, nfft, Fs_min, 'centered');

%% 2.1
syms z 
H = 0.05* (1-z^-800+z^-1600+z^-2400-z^-3200+z^-4000);
hz = iztrans(H);
t=1:40000;
n=t;
hz_num = double(subs(hz));
%% 
outputSignal = (conv(normalized_data', hz_num))';
%listen
soundsc(outputSignal,fs);

spectrogram(outputSignal, window, noverlap, nfft, fs, 'centered');
corr = xcorr(outputSignal);
power_channel = max(corr);
disp(['power_channel = ', num2str(power_channel)]);
figure;
plot(outputSignal');
%% 2.2
%noise = sqrt(0.1) * randn(1, length(outputSignal));
noise = 0 * randn(1, length(outputSignal));
noise_added=outputSignal + noise';
figure;
spectrogram(noise_added, window, noverlap, nfft, fs, 'centered');
corr_noise = xcorr(noise_added);
power_channel_noise = max(corr_noise);
disp(['power_channel_noise = ', num2str(power_channel_noise)]);
figure;
plot(noise_added');

%% 2.3
syms z 
Heq = 20* (1+z^-800-z^-1600-z^-2400+z^-3200-z^-4000);
hz_eq = iztrans(Heq);
t=1:40000;
n=t;
hz_num_eq = double(subs(hz_eq));
%% 
outputSignal_eq = (conv(noise_added', hz_num_eq))';
%listen
soundsc(outputSignal_eq,fs);
figure;
spectrogram(outputSignal_eq, window, noverlap, nfft, fs, 'centered');
corr_eq = xcorr(outputSignal_eq);
power_channel_eq = max(corr_eq);
%% 2.4
filterDesign = designfilt('lowpassfir', 'PassbandFrequency', .47, ...
                          'StopbandFrequency', .50, 'PassbandRipple', ...
                          1, 'StopbandAttenuation', 80);
filterCoeffs = filterDesign.Coefficients;               
filteredOutput = conv(noise_added, filterCoeffs);
figure;
spectrogram(filteredOutput, window, noverlap, nfft, fs, 'centered');
corr_fil = xcorr(filteredOutput);
power_channel_fil = max(corr_fil);
disp(['power_channel_fil = ', num2str(power_channel_fil)]);
figure;
plot(filteredOutput');

%listen
soundsc(outputSignal_eq,fs);
%% 2.5

filterDesign = designfilt('lowpassfir', 'PassbandFrequency', .20, ...
                          'StopbandFrequency', .25, 'PassbandRipple', ...
                          1, 'StopbandAttenuation', 80);
filterCoeffs = filterDesign.Coefficients;               
filteredOutput = conv(noise_added, filterCoeffs);
figure;
spectrogram(filteredOutput, window, noverlap, nfft, fs, 'centered');
corr_fil = xcorr(filteredOutput);
power_channel_fil = max(corr_fil);
disp(['power_channel_fil = ', num2str(power_channel_fil)]);

















