clc;
clear;
%% 1.1
fs = 10e3;    
t = 0:1/fs:10;   
fc = 500;   
delta_f = 60; 

x = 4*heaviside(t) - 7*heaviside(t-4) + 3*heaviside(t-5) - 3*heaviside(t-8);
integral_x = cumtrapz(t, x);

xc = cos(2*pi*fc*t + 2*pi*delta_f*integral_x);

figure;
subplot(2, 1, 1);
plot(t, x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal');


subplot(2, 1, 2);
plot(t, integral_x);
xlabel('Time (s)');
ylabel('Integral');
title('Integral of Original Signal');

%% 1.2

figure;
plot(t, xc);
xlabel('Time (s)');
ylabel('Amplitude');
title('FM Modulated Signal');
xlim([0 0.01])

%% 1.3

xc = fmmod(x, fc, fs, delta_f);

figure;
plot(t, xc);
xlabel('Time (s)');
ylabel('Amplitude');
title('FM Modulated Signal');
xlim([0 0.01])
%% 1.4

figure;
subplot(2, 1, 1);
spectrogram(x, hamming(256), 128, 512, fs, 'yaxis');
title('Spectrogram of Original Signal');

subplot(2, 1, 2);
spectrogram(xc, hamming(256), 128, 512, fs, 'yaxis');
title('Spectrogram of FM Modulated Signal');
%% 1.6

lpf_cutoff = 80;
lpf_order = 1;

xd = diff(xc) * fs;
yd1 = abs(xd);

[b, a] = butter(lpf_order, lpf_cutoff / (fs), 'low');
yd2 = filtfilt(b, a, yd1);

yd3 = fmdemod(xc,fc, fs, lpf_cutoff);

t1 = (0:numel(yd2)-1) / fs;
t2 = (0:numel(yd3)-1) / fs;
figure;
subplot(2,1,1);
plot(t1, yd2);
xlabel('Time');
title('Demodulated Signal using the System');

subplot(2,1,2);
plot(t2, yd3);
xlabel('Time');
title('Demodulated Signal using fmdemod');

%% 2
fs = 10e3;    
t = 0:1/fs:10;   
fc = 500; 
x = 4*heaviside(t) - 7*heaviside(t-4) + 3*heaviside(t-5) - 3*heaviside(t-8);
x_modulated = pmmod(x, fc, fs,10 ); 

figure;
plot(t, x_modulated, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('x_m(t)');
title('Modulated Signal');
xlim([0 0.01])

figure;
subplot(2, 1, 1);
spectrogram(x, hamming(256), 128, 512, fs, 'yaxis');
title('Spectrogram of Original Signal');

subplot(2, 1, 2);
spectrogram(x_modulated, hamming(256), 128, 512, fs, 'yaxis');
title('Spectrogram of FM Modulated Signal');




