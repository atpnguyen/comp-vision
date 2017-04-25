clear all; close all; clc;

%% Task 1
Fhat = zeros(128,128);
p = 125;
q = 1;
Fhat(p,q) = 1;

% showgrey(Fhat)
F = ifft2(Fhat);

Fabsmax = max(abs(F(:)));
figure;
subplot(2,2,1);
showgrey(real(F), 64, -Fabsmax,Fabsmax);
title('real(F)');
subplot(2,2,2);
showgrey(imag(F), 64, -Fabsmax, Fabsmax);
title('imag(F)');
subplot(2,2,3);
showgrey(abs(F), 64, -Fabsmax, Fabsmax);
title('abs(F)');
subplot(2,2,4);
showgrey(angle(F), 64, -pi, pi);
title('angle(F)');

%% 1.3 Basic functions

u = 80;
v = 1;
sz = 128;
figure;
fftwave(u, v, sz);


%% 1.4 Linearity
clear all; close all; clc;

figure;
F = [zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';
H = F + 2 * G;

% F = [ones(16, 16) zeros(16,112); zeros(56, 128)];

% F = zeros(128);
% F(100) = 1;

% Fourier transform
Fhat = fft2(F);
Ghat = fft2(G);
Hhat = fft2(H);

% Fourier spectra
subplot(121); showgrey(F, 64); 
subplot(122); showgrey(log(1 + abs(Fhat))); input('Press any key to continue');
subplot(121); showgrey(G, 64); 
subplot(122); showgrey(log(1 + abs(Ghat))); input('Press any key to continue');
subplot(121); showgrey(H, 64); 
subplot(122); showgrey(log(1 + abs(Hhat))); input('Press any key to continue');

% center Fourier spectrum on zero-frequency component
showgrey(abs(fftshift(Hhat)), 200);
%}

%% 1.5 Multiplication
clear all; close all; clc;

F = [zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';

figure;
showgrey(F .* G);
title('Multiplication in spatial domain');
figure;
multFFT = fft2(F .* G);
subplot(121); showfs(multFFT);
title('Multiplication in frequency domain');
Fhat = fft2(F);
Ghat = fft2(G);
Fconv = conv2(Fhat,Ghat, 'full');
subplot(122); showfs(Fconv);


%% 1.6 Scaling

F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* ...
[zeros(128, 48) ones(128, 32) zeros(128, 48)];
figure;
subplot(121);
showgrey(F);
title({['Scaling'],['Spatial domain']});
subplot(122);
showfs(fft2(F));
title('Fourier domain');


%% 1.7 Rotation

alpha = 60;
G = rot(F, alpha);
figure;
subplot(121);showgrey(G);
title({['Rotation'],['Spatial domain']});
axis on;
%FFT
Ghat = fft2(G);
% figure;
% showfs(Ghat);
Hhat = rot(fftshift(Ghat), -alpha );
subplot(122);showgrey(log(1 + abs(Hhat)));
title('Fourier domain');


%% 1.8 Information in Magnitude vs. Phase
img = few128;
figure; 
subplot(121); showgrey(img);
powFFT = pow2image(img,10e-10);
subplot(122); showgrey(powFFT);
figure;
showgrey(randphaseimage(img));

%% 2.3 Filtering
clear all; close all; clc;
img = deltafcn(128,128);
% img = zeros(128,128);
% img(1) = 1;
% img = few128;
N = length(img);

figure;
subplot(231);
showgrey(img);

%
t = 100;
psf = gaussfft(img,t);
% psf = discgaussfft(img,t);

% plot impulse response
% figure;
subplot(122);
covarM = variance(psf);
t = [1:N];
surf(t,t,reshape(psf,N,N));
title(['Impulse response of Gaussian kernel; Variance = ' num2str(covarM(1,1))]);
%}

% filtering loop
%{
n = 2;
for t = [1,4,16,64,256]
    psf = gaussfft(img,t);
%     psf = discgaussfft(img,t);

    % plot impulse response
    % figure;
    subplot(2,3,n);
    covarM = variance(psf);
    % t = [1:N];
    % surf(t,t,reshape(psf,N,N));
    % title(['Impulse response of Gaussian kernel; Variance = ' num2str(covarM(1,1))]);
    showgrey(psf);
    title(['Filtered image; t = ' num2str(t)]);
    n = n +1;
end
%}
%% 3.0 Smoothing
clear all; close all; clc;

office = office256;
N = length(office);
var = 1.0; %interesting values: 1.0, 2.0, 10.0
medSize = 3; %interesting values: 3
f_cut = 0.2; % similar to JPEGs, the more freq. are cut away, the more we see the low freq. (interesting values 0.5, 0.1, 0.005)
add = gaussnoise(office, 16);
sap = sapnoise(office, 0.1, 255);

figure;
subplot(131); showgrey(office); title('Original');
subplot(132); showgrey(add); title('+ Gaussian noise');
subplot(133); showgrey(sap); title('+ Salt & Pepper noise');

figure;
subplot(231); showgrey(discgaussfft(add,var)); title('Gauss filter');
subplot(232); showgrey(medfilt(add,medSize)); title('Median filter');
subplot(233); showgrey(ideal(add,f_cut)); title('Ideal filter');
subplot(234); showgrey(discgaussfft(sap,var)); title('Gauss filter');
subplot(235); showgrey(medfilt(sap,medSize)); title('Median filter');
subplot(236); showgrey(ideal(sap,f_cut)); title('Ideal filter');
% figure;
% subplot(121); showgrey(discgaussfft(add,1)); title('Gauss filter');
% subplot(122); showgrey(discgaussfft(sap,1));
% figure;
% subplot(121); showgrey(medfilt(add,1)); title('Median filter');
% subplot(122); showgrey(medfilt(sap,1));
% figure;
% subplot(121); showgrey(ideal(add,f_cut)); title('Ideal filter');
% subplot(122); showgrey(ideal(sap,f_cut));

%% 3.2 Smoothing and subsampling
clear all; close all; clc;

img = phonecalc256;
smoothimg = img;
N=5;
var = 0.4;
for i=1:N
    if i>1
         % generate subsampled versions
        img = rawsubsample(img);
        smoothimg = gaussfft(smoothimg, var);
%         smoothimg = ideal(smoothimg, var);
        smoothimg = rawsubsample(smoothimg);
    end
    subplot(2, N, i)
    showgrey(img)
    subplot(2, N, i+N)
    showgrey(smoothimg)
end
title(['Var = ' num2str(var)]);