clc;
clear;
%% 1.1

t = -0.001:0.000000001:0.001;
hm = -2 *cos(2*pi*8*10^3*t)-1/4 *sin(2*pi*(10*10^3)*t)+1/4 *sin(2*pi*(6*10^3)*t)-cos(2*pi*(12*10^3)*t)-cos(2*pi*(4*10^3)*t)-3/4  *sin(5*pi*8*10^3*t)+3/4  *sin(pi*8*10^3*t)-sin(2*pi*(32*10^3)*t)+sin(2*pi*(8*10^3)*t);
figure;
plot(t,hm);
title('1.1');
xlabel('t');
ylabel('Hilbert m(t)');
grid on

%% 1.2

m = 2 *sin(2*pi*8*10^3*t)-1/4 *cos(2*pi*(10*10^3)*t)+1/4 *cos(2*pi*(6*10^3)*t)+sin(2*pi*(12*10^3)*t)+sin(2*pi*(4*10^3)*t)-3/4  *cos(5*pi*8*10^3*t)+3/4  *cos(pi*8*10^3*t)-cos(2*pi*(32*10^3)*t)+cos(2*pi*(8*10^3)*t);
hm2 = hilbert(m);

figure;
plot(t,imag(hm2));
title('1.2');
xlabel('t');
ylabel('Hilbert m(t)');
grid on

%% 1.3

figure;
subplot(2,2,1);
plot(t,real(hm2))
title('real part')

subplot(2,2,2);
plot(t,imag(hm2))
title('imag part')

subplot(2,2,[3,4]);
plot(t,m);
title('m(t)')

%% 1.4
k = 1;
w = -2*pi*10*k:0.01:2*pi*10*k;

t = -0.1:0.001:0.1;
m = 2 *sin(2*pi*8*10^3*t)-1/4 *cos(2*pi*(10*10^3)*t)+1/4 *cos(2*pi*(6*10^3)*t)+sin(2*pi*(12*10^3)*t)+sin(2*pi*(4*10^3)*t)-3/4  *cos(5*pi*8*10^3*t)+3/4  *cos(pi*8*10^3*t)-cos(2*pi*(32*10^3)*t)+cos(2*pi*(8*10^3)*t);

%Fm = (dirac(w+2*pi*8*k) - dirac(w-2*pi*8*k) - 1/8 * dirac(w+2*pi*10*k) - 1/8 * dirac(w-2*pi*10*k) + 1/8 * dirac(w+2*pi*6*k) + 1/8 * dirac(w-2*pi*6*k) + 1/2 * dirac(w+2*pi*12*k) - 1/2 * dirac(w-2*pi*12*k) +1/2 * dirac(w+2*pi*4*k) - 1/2 * dirac(w-2*pi*4*k) - 3/8 * dirac(w+5*pi*8*k) - 3/8 * dirac(w-5*pi*8*k) + 3/8 * dirac(w+pi*8*k) + 3/8 * dirac(w-pi*8*k) - 1/2 * dirac(w+2*pi*32*k) - 1/2 * dirac(w-2*pi*32*k) + 1/2 * dirac(w+2*pi*8*k) + 1/2 * dirac(w-2*pi*8*k));

%figure;
%stem(w, Fm);
%xlabel('w');
%ylabel('Amplitude');
%title('Fm');

M = fft(m);

Fs = 1 / (t(2) - t(1)); 
N = length(t); 
f = linspace(-Fs/2, Fs/2, N); 


figure;
subplot(2,1,1);
plot(f, abs(M));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Magnitude Spectrum 2 sided');

subplot(2,1,2);
plot(f, angle(M));
xlabel('Frequency (Hz)');
ylabel('Phase');

figure;
subplot(2,1,1);
plot(f, real(M));
xlabel('Frequency (Hz)');
ylabel('real');

subplot(2,1,2);
plot(f, imag(M));
xlabel('Frequency (Hz)');
ylabel('imaginery');

%% 1.5

k = 1;
w = -2*pi*10*k:0.01:2*pi*10*k;

t = -0.1:0.001:0.1;
m = 2 *sin(2*pi*8*10^3*t)-1/4 *cos(2*pi*(10*10^3)*t)+1/4 *cos(2*pi*(6*10^3)*t)+sin(2*pi*(12*10^3)*t)+sin(2*pi*(4*10^3)*t)-3/4  *cos(5*pi*8*10^3*t)+3/4  *cos(pi*8*10^3*t)-cos(2*pi*(32*10^3)*t)+cos(2*pi*(8*10^3)*t);

ana = hilbert(m);
hilbert_m = imag(ana);
%Fm = (dirac(w+2*pi*8*k) - dirac(w-2*pi*8*k) - 1/8 * dirac(w+2*pi*10*k) - 1/8 * dirac(w-2*pi*10*k) + 1/8 * dirac(w+2*pi*6*k) + 1/8 * dirac(w-2*pi*6*k) + 1/2 * dirac(w+2*pi*12*k) - 1/2 * dirac(w-2*pi*12*k) +1/2 * dirac(w+2*pi*4*k) - 1/2 * dirac(w-2*pi*4*k) - 3/8 * dirac(w+5*pi*8*k) - 3/8 * dirac(w-5*pi*8*k) + 3/8 * dirac(w+pi*8*k) + 3/8 * dirac(w-pi*8*k) - 1/2 * dirac(w+2*pi*32*k) - 1/2 * dirac(w-2*pi*32*k) + 1/2 * dirac(w+2*pi*8*k) + 1/2 * dirac(w-2*pi*8*k));

%figure;
%stem(w, Fm);
%xlabel('w');
%ylabel('Amplitude');
%title('Fm');

M_ans = fft(ana);
M_h = fft(hilbert_m);

Fs = 1 / (t(2) - t(1)); 
N = length(t); 
f = linspace(-Fs/2, Fs/2, N); 


figure;
subplot(2,1,1);
plot(f, abs(M_ans));
xlabel('Frequency (Hz)');
ylabel('Magnitude Analytic');

subplot(2,1,2);
plot(f, angle(M_ans));
xlabel('Frequency (Hz)');
ylabel('Phase Analytic');

figure;
subplot(2,1,1);
plot(f, real(M_ans));
xlabel('Frequency (Hz)');
ylabel('real Analytic');

subplot(2,1,2);
plot(f, imag(M_ans));
xlabel('Frequency (Hz)');
ylabel('imaginery Analytic');


figure;
subplot(2,1,1);
plot(f, abs(M_h));
xlabel('Frequency (Hz)');
ylabel('Magnitude Hilbert');

subplot(2,1,2);
plot(f, angle(M_h));
xlabel('Frequency (Hz)');
ylabel('Phase Hilbert');

figure;
subplot(2,1,1);
plot(f, real(M_h));
xlabel('Frequency (Hz)');
ylabel('real Hilbert');

subplot(2,1,2);
plot(f, imag(M_h));
xlabel('Frequency (Hz)');
ylabel('imaginery Hilbert');

%% 2.1
t = -10:0.01:10; 
m = exp(-3*t) .* (heaviside(t-6) - heaviside(t-9)) + exp(3*t) .* (heaviside(-t-6) - heaviside(-t-9)); % تابع m(t)


figure;
plot(t, m);
xlabel('time');
ylabel('m(t)');

M = fftshift(fft(m));
Fs = 1 / (t(2) - t(1)); 
f = (-Fs/2 : Fs/length(t) : Fs/2 - Fs/length(t));

figure;
plot(f, abs(M));
xlabel('Frequency');

ylabel('Magnitude');

%% 2.2

Fs = 100; 
t2 = -10:1/Fs:10;
m2 = exp(-3*t) .* (heaviside(t-6) - heaviside(t-9)) + exp(3*t) .* (heaviside(-t-6) - heaviside(-t-9)); % Signal m(t)

figure;
scatter(t2, m2);
xlabel('time');
ylabel('m(t)');

%% 2.3

A = 0.5; 
fc = 10; 

M_modulated = ( 1 + A * m ).* cos(2*pi*fc*t);

figure;
plot(t, M_modulated);
xlabel('Time');
ylabel('Modulated Signal');
xlim([-0.2, 0.2]);

%% 2.4

M = fftshift(fft(M_modulated));
Fs = 1 / (t(2) - t(1)); 
f = (-Fs/2 : Fs/length(t) : Fs/2 - Fs/length(t));

figure;
plot(f, abs(M));
xlabel('Frequency');

ylabel('Magnitude');

%% 2.6
time_constant = 1.5;
initial_output = 0;

output_signal = rc_circuit(M_modulated, time_constant, initial_output);

plot(t, output_signal);
xlabel('Time');
title('RC Circuit Response');

%% 2.8

M_modulated2 = ( 1 + A * m ).* cos(2*pi*fc*t);

M_abs = abs(fftshift(fft(M_modulated2))) .* heaviside(f) ;
X = ifft(M_abs);

plot(f, M_abs);
xlabel('Frequency');
ylabel('Magnitude');

%% 2.9

M_modulated3 = ( 1 + A * m ).* cos(2*pi*fc*t) .* cos(2*pi*fc*t);

M_abs_2 = fftshift(fft(M_modulated3)) .* (heaviside(f+10)-heaviside(f-10));
X = ifft(M_abs_2);

figure;
plot(f, abs(M_abs_2));
xlabel('Frequency');
ylabel('Magnitude');

figure;
plot(t, X);

%% 2.10

y = ammod(M_modulated,fc,100);
figure;
plot(t, y);
xlim([-0.2 0,2])

%% 2.11
X = fft(m);

X_ssb = X;
X_ssb(1:length(X)/2) = 0;  
x_ssb = ifft(X_ssb);

figure;
subplot(2, 1, 1);
plot(t, x_ssb);
title('ssb Signal');
xlabel('Time');
ylabel('Amplitude');

subplot(2, 1, 2);
f = (-Fs/2:Fs/length(X_ssb):Fs/2-Fs/length(X_ssb));
plot(f, abs(fftshift(X_ssb)));
title('SSB Signal in the Frequency Domain');
xlabel('Frequency');
ylabel('Magnitude');

%% 2.12

M_modulated4 = x_ssb .* cos(2*pi*fc*t) .* (heaviside(f+10)-heaviside(f-10));

M_abs_4 = fftshift(fft(M_modulated4));
X = ifft(M_abs_4);

figure;
subplot(2, 1, 1);
plot(t, X);
xlabel('Time');
ylabel('Amplitude');

subplot(2, 1, 2);
f = (-Fs/2:Fs/length(X_ssb):Fs/2-Fs/length(X_ssb));
plot(f, abs(M_abs_4));
xlabel('Frequency');
ylabel('Magnitude');

%% 2.14- recorsing
fs = 40000; 
bitDepth = 16; 
duration = 0.5; 
recorder = audiorecorder(fs, bitDepth, 1);
%record
recordblocking(recorder, duration);
%to data
data = getaudiodata(recorder);
%save
save('D:\electrical eng\term 6\communication sys\CA02_400110009_ZahraMaleki\recorded_audio.mat', 'data', 'fs');
audiowrite('D:\electrical eng\term 6\communication sys\CA02_400110009_ZahraMaleki\recorded_audio.wav', data, fs);
%% 
fs = 40000; 
bitDepth = 16; 
duration = 10; 
nfft = 1024;
fs = 40000; 
window = nfft; 
noverlap = 0; 

data=audioread("D:\electrical eng\term 6\communication sys\CA02_400110009_ZahraMaleki\recorded_audio.wav");

spectrogram(data, window, noverlap, nfft, fs);
%% 2.14

t = linspace(0, 0.5, 20000);
A = 0.5; 
fc = 1000; 

M_modulated = ( 1 + A * data ) .* cos(2*pi*fc .* t');
%% 
nfft = 1024;
fs = 40000; 
window = nfft; 
noverlap = 0; 

figure;
plot(t', M_modulated);
xlabel('Time');
ylabel('Modulated Signal');

figure;
spectrogram(M_modulated);
%% 
t1= t' ;
Fs = 1 / (t1(2) - t1(1)); 
f = (-Fs/2 : Fs/length(t) : Fs/2 - Fs/length(t));

M_abs = abs(fftshift(fft(M_modulated))) .* heaviside(f) ;
X1 = ifft(M_abs);

figure;
plot(f, M_abs);
xlabel('Frequency');
ylabel('Magnitude abs-lpf');

M_modulated3 = M_modulated .* cos(2*pi*fc .*t');

M_abs_2 = fftshift(fft(M_modulated3)) .* (heaviside(f+300)-heaviside(f-300));
X2 = ifft(M_abs_2);

figure;
plot(f, abs(M_abs_2));
xlabel('Frequency');
ylabel('Magnitude cos-lpf');

figure;
plot(t', X2);

y = ammod(M_modulated,fc,100);
figure;
plot(t', y);
ylabel('Magnitude ammod');

%% 2.6- function

function output = rc_circuit(input_signal, time_constant, initial_output)
    output = zeros(size(input_signal));  % Initialize the output with zeros
    output(1) = initial_output;  % Set the initial output value

    for i = 2:length(input_signal)
        dt = input_signal(i) - input_signal(i-1);  % Time step
        d_output = (input_signal(i) - output(i-1)) / sqrt(time_constant);  % Change in output
        output(i) = output(i-1) + d_output * dt;  % Update the output
    end
end








