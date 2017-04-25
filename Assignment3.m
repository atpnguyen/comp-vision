%% Q1
clear all;
%close all;
clc;

K = 8;               % number of clusters used; def: 8
L = 10;              % number of iterations; def: 10
seed = 14;           % seed used for random initialization; def: 14
scale_factor = 1.0;  % image downscale factor; def: 1.0
image_sigma = 1.0;   % image preblurring scale; def: 1.0

orange = imread('orange.jpg');
tiger1 = imread('tiger1.jpg');
tiger2 = imread('tiger2.jpg');
tiger3 = imread('tiger3.jpg');

I = tiger1;
I = imresize(I, scale_factor); %resize image
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma); % create gaussian filter
I = imfilter(I, h); %filters image I with filter h

% plot original
% figure;
% imshow(I);
% title('Original image');

tic %take time
[ segm, centers ] = kmeans_segm(I, K, L, seed);
toc
Inew = mean_segments(Iback, segm);
Iover = overlay_bounds(Iback, segm);

% plot K-means result
figure;
subplot(221);
imshow(Inew);
title(['K-means clustering; K = ' num2str(K) '; L = ' num2str(L)]);
subplot(222);
imshow(Iover);
title('K-means w/ segmentation boundaries');
subplot(223);
imshow(I);
title('Original image');

% save images
% imwrite(Inew,'result/kmeans1.png');
% imwrite(Iover,'result/kmeans2.png');

%% Q2: with for-loops to test effects of K,L,scale_factor
clear all; 
%close all; 
clc;

K = 8;               % number of clusters used; def: 8
L = 10;              % number of iterations; def: 10
seed = 14;           % seed used for random initialization; def: 14
scale_factor = 1.0;  % image downscale factor; def: 1.0
image_sigma = 1.0;   % image preblurring scale; def: 1.0

orange = imread('orange.jpg');
tiger1 = imread('tiger1.jpg');
tiger2 = imread('tiger2.jpg');
tiger3 = imread('tiger3.jpg');

I = tiger3;
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

figure;
i = 1;
for L=[5 10 15 20] % change to test different parameters
    tic
    [ segm, centers ] = kmeans_segm(I, K, L, seed);
    toc
    Inew = mean_segments(Iback, segm);
    Iover = overlay_bounds(Iback, segm);
    
    subplot(2,2,i);
    imshow(Iover);
    title(['L = ' num2str(L)]); % change according to current varying parameter
    
    i = i+1;
end

%% Q5: Mean-shift segmentation
clear all; 
%close all;
clc;

scale_factor = 0.5;       % image downscale factor; def: 0.5
spatial_bandwidth = 10.0;  % spatial bandwidth; def: 10.0
colour_bandwidth = 5.0;   % colour bandwidth; def: 5.0
num_iterations = 40;      % number of mean-shift iterations; def: 40
image_sigma = 1.0;        % image preblurring scale; def: 1.0

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
% imwrite(Inew,'result/meanshift1.png')
% imwrite(I,'result/meanshift2.png')

figure;
subplot(1,2,1); imshow(Inew);
subplot(1,2,2); imshow(I);
title(['Bandwidth = ' num2str(colour_bandwidth)]);

%% Q7: Normalised Cut
clear all;
close all; 
clc;

colour_bandwidth = 20.0; % color bandwidth; def: 20.0
radius = 3;              % maximum neighbourhood distance; def: 3
ncuts_thresh = 0.2;      % cutting threshold; maximum allowed value for Ncut(A,B); def: 0.2
min_area = 200;          % minimum area of segment: def: 200ostentation
max_depth = 8;           % maximum splitting depth; depth of recursion; def: 8
scale_factor = 0.4;      % image downscale factor; def: 0.4
image_sigma = 2.0;       % image preblurring scale; def: 2.0

I = imread('tiger3.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma); %create gaussian filter
I = imfilter(I, h); %perform filter

segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
% imwrite(Inew,'result/normcuts1.png')
% imwrite(I,'result/normcuts2.png')

% figure;
subplot(1,2,1); 
% imshow(Inew);
% subplot(1,2,2); 
imshow(I);
% title(['Bandwidth = ' num2str(colour_bandwidth) '; max_{depth} = ' num2str(max_depth) '; ncuts_{thresh} = ' num2str(ncuts_thresh)]);
title(['Radius = ' num2str(radius)]);

%% Q11: Graph cuts
% clear all;
close all;
% clc;

scale_factor = 0.5;          % image downscale factor
area = [ 80, 110, 570, 300 ]; % image region to train foreground with
K = 16;                      % number of mixture components
alpha = 8.0;                 % maximum edge cost
sigma = 10.0;                % edge cost decay factor

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
area = int16(area*scale_factor);
[ segm, prior ] = graphcut_segm(I, area, K, alpha, sigma);

Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);

% save images locally
% imwrite(Inew,'result/graphcut1.png')
% imwrite(I,'result/graphcut2.png')
% imwrite(prior,'result/graphcut3.png')

% plot
figure;
subplot(2,2,1); imshow(Inew);title('segmented image');
subplot(2,2,2); imshow(I);title('original image');
subplot(2,2,3); imshow(prior);title(' image');