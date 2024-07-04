clc;
clear;
%% 1

N = 500;       
Ts = 1;    
Tb = 50;    
A = 1;        

binary_sequence = randi([0, 1], 1, N);

t_pulse = 0:Ts:Tb-Ts;
pulse = A * square(2*pi*t_pulse/Tb);

signal = zeros(1, N*Tb/Ts);

for i = 1:N
    
    start_index = (i-1)*(Tb/Ts) + 1;
    end_index = start_index + (Tb/Ts) - 1;
    signal(start_index:end_index) = binary_sequence(i) * pulse;
end

t = Ts:Ts:N*Tb;
plot(t, signal);
xlabel('Time (seconds)');
ylabel('Signal');
ylim([-2 2]);

effective_bandwidth = 1 / Tb;
disp(['Effective Bandwidth: ', num2str(effective_bandwidth), ' Hz']);

%% 2

Ts = 1;      
eta = 2;        

t = 0:Ts:eta-1;
white_noise = randn(size(t));

power_noise = (norm(white_noise)^2) / (eta*Ts);

disp(['Power of Sampled White Noise: ', num2str(power_noise)]);

%% 3,4
SNRe_min = -10; 
SNRe_max = 10;

SNRe = SNRe_min + (SNRe_max - SNRe_min) * rand(1);
disp(['SNRe: ', num2str(SNRe)]);

bpsk_pulse_sequence1 = zeros(1, 500*Tb/Ts);
for i = 1:500
    start_index = (i-1)*(Tb/Ts) + 1;
    end_index = start_index + (Tb/Ts) - 1;
    if binary_sequence(i) == 1
        bpsk_pulse_sequence1(start_index:end_index) = A * sqrt(10^(SNRe/10));
    else
        bpsk_pulse_sequence1(start_index:end_index) = -A * sqrt(10^(SNRe/10));
    end
end

binary_sequence2 = randi([0, 1], 1, 500);

bpsk_pulse_sequence2 = zeros(1, 500*Tb/Ts);
for i = 1:500
    start_index = (i-1)*(Tb/Ts) + 1;
    end_index = start_index + (Tb/Ts) - 1;
    if binary_sequence2(i) == 1
        bpsk_pulse_sequence2(start_index:end_index) = A * sqrt(10^(SNRe/10));
    else
        bpsk_pulse_sequence2(start_index:end_index) = -A * sqrt(10^(SNRe/10));
    end
end

figure;
t_pulse = Ts:Ts:500*Tb;
plot(t_pulse, bpsk_pulse_sequence1);
xlabel('Time');
ylabel('BPSK1');
figure;
plot(t_pulse, bpsk_pulse_sequence2);
xlabel('Time');
ylabel('BPSK2');

%% 5

noise_sequence = randn(1, 500*Tb/Ts);

bpsk_noise_sequence1 = bpsk_pulse_sequence1 + noise_sequence;
bpsk_noise_sequence2 = bpsk_pulse_sequence2 + noise_sequence;

t_noise = Ts:Ts:500*Tb;

figure;
plot(t_noise, bpsk_noise_sequence1);
xlabel('t');
ylabel('v1(t)');

figure;
plot(t_noise, bpsk_noise_sequence2);
xlabel('t');
ylabel('v2(t)');
%% 6
symb_I1 = bpsk_noise_sequence1 .* cos(10*pi*t_noise/Tb);
symb_Q1 = bpsk_noise_sequence1 .* sin(10*pi*t_noise/Tb);

figure;
scatter(symb_I1, symb_Q1);
xlabel('I1');
ylabel('Q1');
grid on;
sample1= symb_I1 + 1j*symb_Q1;

symb_I2 = bpsk_noise_sequence2 .* cos(10*pi*t_noise/Tb);
symb_Q2 = bpsk_noise_sequence2 .* sin(10*pi*t_noise/Tb);

figure;
scatter(symb_I2, symb_Q2);
xlabel('I1');
ylabel('Q1');
grid on;
sample2= symb_I2 + 1j*symb_Q2;

constellation = comm.ConstellationDiagram('Title','BPSK','ShowTrajectory',false,'SamplesPerSymbol',1);
%% 

close all
framlen=512;
for i=0:(numel(sample1)/framlen)-1 
    slice= i*framlen+1 : framlen*(i+1);
    framsamples = sample1(slice).';
    step(constellation,(framsamples(1:framlen)));
    
    pause(0.1)
    
end

%% 
close all
framlen=512;
for i=0:(numel(sample2)/framlen)-1 
    slice= i*framlen+1 : framlen*(i+1);
    framsamples = sample2(slice).';
    step(constellation,(framsamples(1:framlen)));
    
    pause(0.1)
    
end

%% 7

filterCoeff = [0.2, 0.3, 0.4, 0.3, 0.2]; 

filteredSignal1 = conv(symb_I1 + 1j * symb_Q1, filterCoeff);
filteredSignal2 = conv(symb_I2 + 1j * symb_Q2, filterCoeff);

filteredNoise1 = conv(bpsk_noise_sequence1 .* cos(10*pi*t_noise/Tb), filterCoeff);
filteredNoise2 = conv(bpsk_noise_sequence2 .* sin(10*pi*t_noise/Tb), filterCoeff);

n=round(Tb / Ts);
downsampledSignal1 = filteredSignal1(1:n:end);
downsampledSignal2 = filteredSignal2(1:n:end);
downsampledNoise1 = filteredNoise1(1:n:end);
downsampledNoise2 = filteredNoise2(1:n:end);

t_downsampled = Ts * n * (0:length(downsampledSignal1)-1);

figure;
plot(t_downsampled, downsampledNoise1);
xlabel('Time');
ylabel('Downsampled1');

figure;
plot(t_downsampled, downsampledNoise2);
xlabel('Time');
ylabel('Downsampled2');

%% 8

threshold_estimate1 = 0.8;
threshold_estimate2 = 0.7;

estimated_bits1 = zeros(size(downsampledSignal1));
estimated_bits2 = zeros(size(downsampledSignal2));

for i = 1:length(downsampledSignal1)
    if abs(downsampledSignal1(i)) > threshold_estimate1
        estimated_bits1(i) = 1;
    else
        estimated_bits1(i) = 0;
    end
end

for i = 1:length(downsampledSignal2)
    if abs(downsampledSignal2(i)) > threshold_estimate2
        estimated_bits2(i) = 1;
    else
        estimated_bits2(i) = 0;
    end
end

fprintf('Threshold Estimate 1: %.4f\n', threshold_estimate1);
fprintf('Threshold Estimate 2: %.4f\n', threshold_estimate2);

figure;
plot(estimated_bits1);
title('Estimated Bits1');
xlabel('Sample Index');
ylabel('Bit Value');
ylim([-0.2 1.2]);
xlim([0 500]);
figure;
plot(estimated_bits2);
title('Estimated Bits2');
xlabel('Sample Index');
ylabel('Bit Value');
ylim([-0.2 1.2]);
xlim([0 500]);

%% 9
SNRe_min = -10; 
SNRe_max = 10;
num_points = 6;

SNR_values = linspace(SNRe_min, SNRe_max, num_points);
BER = zeros(1, num_points);

for j = 1:num_points
    
    SNRe = SNR_values(j);
    
    bpsk_pulse_sequence_9 = zeros(1, 500*Tb/Ts);
    for i = 1:500
        start_index = (i-1)*(Tb/Ts) + 1;
        end_index = start_index + (Tb/Ts) - 1;
        if binary_sequence(i) == 1
            bpsk_pulse_sequence_9(start_index:end_index) = A * sqrt(10^(SNRe/10));
        else
            bpsk_pulse_sequenc_9(start_index:end_index) = -A * sqrt(10^(SNRe/10));
        end
    end

    noise_sequence = randn(1, 500*Tb/Ts);

    bpsk_noise_sequence9 = bpsk_pulse_sequence_9 + noise_sequence;

    symb_I1 = bpsk_noise_sequence9 .* cos(10*pi*t_noise/Tb);
    symb_Q1 = bpsk_noise_sequence9 .* sin(10*pi*t_noise/Tb);

    filterCoeff = [0.2, 0.3, 0.4, 0.3, 0.2]; 

    filteredSignal9 = conv(symb_I1 + 1j * symb_Q1, filterCoeff);

    filteredNoise9 = conv(bpsk_noise_sequence9 .* cos(10*pi*t_noise/Tb), filterCoeff);

    n=round(Tb / Ts);
    downsampledSignal9 = filteredSignal1(1:n:end);
    downsampledNoise9 = filteredNoise1(1:n:end);

    threshold_estimate9 = 0.8;

    estimated_bits9 = zeros(size(downsampledSignal9)-1);

    for i = 1:length(downsampledSignal9)-1
        if abs(downsampledSignal9(i)) > threshold_estimate9
            estimated_bits9(i) = 1;
        else
            estimated_bits9(i) = 0;
        end
    end
    error_count = sum(abs(estimated_bits9 - binary_sequence));
    BER(j) = error_count / 500;
end

figure;
stem(SNR_values, BER);
title('Bit Error Rate (BER) vs. SNR');
xlabel('SNR (dB)');
ylabel('BER');

%% 10

SNRe_min = -10;
SNRe_max = 10;
num_points = 6;

SNR_values = linspace(SNRe_min, SNRe_max, num_points);
BER = zeros(1, num_points);

for j = 1:num_points
    
    SNRe = SNR_values(j);
    
    error_count_total = 0;
    
    for realization = 1:50
        
        bpsk_pulse_sequence_9 = zeros(1, 500*Tb/Ts);
        for i = 1:500
            start_index = (i-1)*(Tb/Ts) + 1;
            end_index = start_index + (Tb/Ts) - 1;
            if binary_sequence(i) == 1
                bpsk_pulse_sequence_9(start_index:end_index) = A * sqrt(10^(SNRe/10));
            else
                bpsk_pulse_sequence_9(start_index:end_index) = -A * sqrt(10^(SNRe/10));
            end
        end

        noise_sequence = randn(1, 500*Tb/Ts);

        bpsk_noise_sequence9 = bpsk_pulse_sequence_9 + noise_sequence;

        symb_I1 = bpsk_noise_sequence9 .* cos(10*pi*t_noise/Tb);
        symb_Q1 = bpsk_noise_sequence9 .* sin(10*pi*t_noise/Tb);

        filterCoeff = [0.2, 0.3, 0.4, 0.3, 0.2]; 

        filteredSignal9 = conv(symb_I1 + 1j * symb_Q1, filterCoeff);

        filteredNoise9 = conv(bpsk_noise_sequence9 .* cos(10*pi*t_noise/Tb), filterCoeff);

        n = round(Tb / Ts);
        downsampledSignal9 = filteredSignal9(1:n:end);
        downsampledNoise9 = filteredNoise9(1:n:end);

        threshold_estimate9 = 0.8;

        estimated_bits9 = zeros(size(downsampledSignal9)-1);

        for i = 1:length(downsampledSignal9)-1
            if abs(downsampledSignal9(i)) > threshold_estimate9
                estimated_bits9(i) = 1;
            else
                estimated_bits9(i) = 0;
            end
        end
        
        error_count = sum(abs(estimated_bits9 - binary_sequence));
        error_count_total = error_count_total + error_count;
    end
    
    BER(j) = error_count_total / (500 * 50);
    
end

figure;
stem(SNR_values, BER);
title('Bit Error Rate (BER) vs. SNR');
xlabel('SNR (dB)');
ylabel('BER');

%% 11

offset_frequency = 0.001 * 1/Ts;
t_noise_offset = t_noise + offset_frequency;

symb_I11 = bpsk_noise_sequence1 .* cos(10*pi*t_noise_offset/Tb);
symb_Q11 = bpsk_noise_sequence1 .* sin(10*pi*t_noise_offset/Tb);

figure;
scatter(symb_I11, symb_Q11);
xlabel('I11');
ylabel('Q11');
grid on;
sample11= symb_I11 + 1j*symb_Q11;

constellation = comm.ConstellationDiagram('Title','BPSK','ShowTrajectory',false,'SamplesPerSymbol',1);


%% 
close all
framlen=512;
for i=0:(numel(sample11)/framlen)-1 
    slice= i*framlen+1 : framlen*(i+1);
    framsamples = sample11(slice).';
    step(constellation,(framsamples(1:framlen)));
    
    pause(0.1)
    
end
%% 12/13 in the pdf



