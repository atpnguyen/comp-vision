%% 1 Difference operators 
% close all; clear all; clc;

% deltax =  [1 0 -1; 1 0 -1; 1 0 -1]; %simple
% deltay = [1 1 1; 0 0 0; -1 -1 -1];
deltax =  [0 0 0; 1 0 -1; 0 0 0]./3;   %central
deltay = [0 1 0; 0 0 0; 0 -1 0]./3;

tools = few256;
dxtools = conv2(tools, deltax, 'valid');
dytools = conv2(tools, deltay, 'valid');
size(dxtools)
figure;
subplot(121);showgrey(dxtools); title('dxtools');
subplot(122);showgrey(dytools); title('dytools');

%%  2 Point-wise thresholding of gradient magnitudes
close all;

%{
dxtoolsconv = dxtools;
dytoolsconv = dytools;
gradmagntools = sqrt(dxtoolsconv .^2 + dytoolsconv .^2);
figure(1);subplot(121);showgrey(gradmagntools); title('Gradient Magnitude');
%TODO : compute histogram and guess threshold that yields reasonably thin
%edges when aplied to gradmagntools
figure(2);histogram(gradmagntools);
% input('Press any key to continue');
threshold = 12;
figure(1);subplot(122);showgrey((gradmagntools - threshold) > 0);title('thresholded image');
%}

figure(1);
img = godthem256;
sigma2 = 1.0;
img = discgaussfft(img, sigma2);

pixels = Lv(img); % pixels function exists. TODO: do smoothing with discgauss
subplot(121);showgrey(pixels); title('Filtering with filter2');
figure(2); histogram(pixels);
threshold = 800;
figure(1); subplot(122);showgrey((pixels - threshold) > 0);title('Thresholded w/ Filter2');
erosion = 1/3.*[0 1 0; 0 2 0; 0 1 0];

%{
img = godthem256;
img = conv2(img, erosion,'valid');
pixels = Lv(img);
threshold =1000;
figure(3); subplot(122);showgrey((pixels - threshold) > 0);
%}


%% 3 Differential geometry based edge detection: Theory TESTING
close all;

dxmask = zeros(5,5);
dxmask(3,2:4) = [-1/2 0 1/2];
dymask = dxmask';

dxxmask = zeros(5,5);
dxxmask(3,2:4) = [1 -2 1];
dyymask = dxxmask';

dxxxmask = conv2(dxmask, dxxmask, 'same');
dyyymask = conv2(dymask, dyymask, 'same');

dxymask = conv2(dxmask, dymask, 'same');
dxxymask = conv2(dxxmask, dymask, 'same');
dxyymask = conv2(dxmask, dyymask, 'same');

[x, y] = meshgrid(-5:5, -5:5);

Lxx = filter2(dxxmask, x .^3, 'valid');
Lxxx = filter2(dxxxmask, x .^3, 'valid');
Lxxy = filter2(dxxymask, x .^2 .* y, 'valid');

%% 4 Computing differential geometry descriptors
clear all; close all; clc;

%house Lvv
house = godthem256;
tools = few256;
img = house;
scaleArr = [0.0001 1.0 4.0 16.0 64.0];

figure('Name', 'House Lvv');
%house Lvv
subplot(3,2,1); showgrey(house);
for i = 1:5
scale = scaleArr(i);
subplot(3,2,i+1);str = sprintf('Scale %d',scale);contour(Lvvtilde(discgaussfft(house, scale), 'same'), [0 0]);title(str); 
axis('image') % fit the axis box tightly around data.
axis('ij') %ij is for reverse y-cordinate
end
figure('Name', 'Tools Lvv')
%tools Lvv
subplot(3,2,1); showgrey(tools);
for i = 1:5
scale = scaleArr(i);
subplot(3,2,i+1);str = sprintf('Scale %d',scale);contour(Lvvtilde(discgaussfft(tools, scale), 'same'), [0 0]);title(str); 
axis('image') % fit the axis box tightly around data.
axis('ij') %ij is for reverse y-cordinate
end

%house Lvvv
figure('Name', 'House Lvvv')
subplot(3,2,1); showgrey(house);
for i = 1:5
scale = scaleArr(i);
subplot(3,2,i+1);str = sprintf('Scale %d',scale); showgrey(Lvvvtilde(discgaussfft(house, scale), 'same') < 0);title(str);
axis('image') % fit the axis box tightly around data.
axis('ij') %ij is for reverse y-cordinate
end

figure('Name', 'Tools Lvvv')
%tools Lvv
subplot(3,2,1); showgrey(tools);
for i = 1:5
scale = scaleArr(i);
subplot(3,2,i+1);str = sprintf('Scale %d',scale); showgrey(Lvvvtilde(discgaussfft(tools, scale), 'same') < 0);title(str);
axis('image') % fit the axis box tightly around data.
axis('ij') %ij is for reverse y-cordinate
end




%% 5 Extraction of edge segments
% detect and display edges
clear all; 
close all; 
clc;

house = godthem256;
tools = few256;
img = house;
scale = 10.0; % best result scale = 4 ; less noise inside tools & different edges not combined (with higher scale)
threshold = 0; % best: 20 ; tower roof detected
shape = 'valid';

% detect edge; filter according to equations Lvv = 0 and Lvvv < 0
edgecurves = extractedge(img, scale, threshold, shape);

% overlay edges on top of original image
figure;
overlaycurves(img, edgecurves);
title(['Edge extraction with scale = ' num2str(scale) ' and threshold = ' num2str(threshold)]);


%% 6 Hough Transform
clear all;
% close all;
clc;

tools = few256;
house = godthem256;
phone = phonecalc256;
triangle = triangle128;
small = binsubsample(triangle);
hought = houghtest256;
small2 = binsubsample(hought);
img = tools;
% img = binsubsample(img); % uncomment for subsampling


scale = 4;  % smoothing parameter (sigma)
gradmagnthreshold = 50; % threshold for edge gradient magnitude
diagonal = sqrt((size(img,1)-1)^2 + (size(img,2)-1)^2); % max radius length (diagonal of image), formula for distance (x,y)
nrho = round(2*diagonal);     % number of possible values for radius length, rounds towards nearest integer
ntheta = 180;   % number of possible values for angle
nlines = 10; % number of extracted lines
verbose = 3;

figure;
showgrey(img);
title('Original image');

tic % start stopwatch
[linepar acc] = houghedgeline(img, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose);
t = toc % display current time
linepar_disp = [linepar(1,:);linepar(2,:)*180/pi];
% disp('linepar format: rho (upper row, in pixel-units); theta(lower row, in \it{degree})');

figure;
plot_lines(img, linepar); % plots lines (edges) in img
title(['Image overlayed with extracted lines; nlines = ' num2str(nlines)]);
% legend(['\rho = ' num2str(linepar(1,1)) ' \theta = ' num2str(linepar(2,1)*180/pi) '°'],...
%     ['\rho = ' num2str(linepar(1,2)) ' \theta = ' num2str(linepar(2,2)*180/pi) '°'],...
%     ['\rho = ' num2str(linepar(1,3)) ' \theta = ' num2str(linepar(2,3)*180/pi) '°']) % legend for analysing triangle
legend(['\theta = ' num2str(ntheta)],... % legend resolution of accumulator; res of rho is always ~1px
    ['Comp. time = ' num2str(t)]); % disp computational time
axis([0 size(img,1) 0 size(img,2)]);