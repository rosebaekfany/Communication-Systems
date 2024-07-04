% ch_input = randn(1, 100); 
% fs = 1000;
% t = 0:1/fs:1-1/fs; 
% f_sin = 10; 
% ch_input = sin(2*pi*f_sin*t); 
ch_input = zeros(1, 100);
ch_input(50) = 1; % Dirac function
ch_type = 'Ray TV FF'; 
snr = 20; 
Tm = 0.1; 
fd = 0.01; 

ch_output = simulate_channel(ch_input, ch_type, snr, Tm, fd);

function ch_output = simulate_channel(ch_input, ch_type, snr, Tm, fd)

    noise = sqrt(0.5)*(randn(size(ch_input)) + 1i*randn(size(ch_input)));
    ch_input_noisy = awgn(ch_input, snr, 'measured');

    ch_output = [];

    switch ch_type
        case 'Ray TI FF'
            % Time-invariant Rayleigh channel with flat fading
            h = (randn + 1i*randn)/sqrt(2);
            ch_output = h * ch_input_noisy;

        case 'Ray TI FS'
            % Time-invariant Rayleigh channel with frequency selective fading
            L = round(Tm*length(ch_input)); 
            h = (randn(L,1) + 1i*randn(L,1))/sqrt(2*L);
            ch_output = filter(h, 1, ch_input_noisy);

        case 'Ray TV FS'
            % Time-varying Rayleigh channel with frequency selective fading
            L = round(Tm*length(ch_input));
            for k = 1:length(ch_input_noisy)
                h = (randn(L,1) + 1i*randn(L,1))/sqrt(2*L) .* exp(1i*2*pi*fd*k);
                segment = ch_input_noisy(max(1, k-L+1):k);
                if length(segment) < L
                    segment = [zeros(L-length(segment),1); segment(:)]; 
                end
                if size(segment)==[1 10]
                    segment = segment.';
                end
                ch_output(k) = h.' * segment;
            end

        case 'Ric TV FS'
            % Time-varying Rician channel with frequency selective fading
            L = round(Tm*length(ch_input)); % Number of taps
            K = 10; 
            for k = 1:length(ch_input_noisy)
                h_LOS = sqrt(K/(K+1));
                h_NLOS = (randn(L,1) + 1i*randn(L,1))/sqrt(2*L*(K+1));
                h = h_LOS + h_NLOS .* exp(1i*2*pi*fd*k);
                segment = ch_input_noisy(max(1, k-L+1):k);
                if length(segment) < L
                    segment = [zeros(L-length(segment),1); segment(:)];
                end
                if size(segment)==[1 10]
                    segment = segment.';
                end
                ch_output(k) = h.' * segment;
            end

        case 'Awgn'
            % AWGN channel
            ch_output = ch_input_noisy;

        case 'Ray TV FF'
            % Time-varying Rayleigh channel with flat fading
            ch_output = zeros(size(ch_input_noisy));
            for k = 1:length(ch_input_noisy)
                h = (randn + 1i*randn)/sqrt(2) * exp(1i*2*pi*fd*k);
                ch_output(k) = h * ch_input_noisy(k);
            end

        otherwise
            error('Unknown channel type');
    end
    ch_output_fft = fft(ch_output);
    fs = 1000;
    figure;
    subplot(2, 1, 1);
    plot(abs(ch_output));
    title(['Impulse Response of ' ch_type ' Channel']);
    xlabel('Sample Index');
    ylabel('Magnitude');
    grid on;

    subplot(2, 1, 2);
    f = (-fs/2):(fs/length(ch_output_fft)):(fs/2)-(fs/length(ch_output_fft));
    plot(f, abs(fftshift(ch_output_fft)));
    title(['Fourier Transform of ' ch_type ' Channel Output']);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    grid on;

end
