
clc;
clear;
close all;

%% Step 1: Read the image
%% ima
[fileName, folderName] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'});

if isequal(fileName, 0)
    disp('No image selected!');
    return;
end

originalImage = imread(fullfile(folderName, fileName));

figure;
imshow(originalImage);
title('Original Image');


figure;
imshow(originalImage);
title('Original Image');

% Convert the image into grayscale
if size(originalImage, 3) == 3
    grayImage = rgb2gray(originalImage);
else
    grayImage = originalImage;
end

%% Step 2: Improve image quality

% Improve the contrast of the image
contrastImage = imadjust(grayImage);

% Enhance the image using histogram equalization
equalizedImage = histeq(grayImage);

% Improve local contrast using CLAHE
enhancedImage = adapthisteq(grayImage);

% Adjust image brightness using gamma correction
gammaImage = imadjust(grayImage, [], [], 0.7);

figure;
subplot(2,2,1);
imshow(contrastImage);
title('Contrast Improved');

subplot(2,2,2);
imshow(equalizedImage);
title('Histogram Equalization');

subplot(2,2,3);
imshow(enhancedImage);
title('Adaptive Histogram Equalization');

subplot(2,2,4);
imshow(gammaImage);
title('Gamma Correction');

%% Step 3: Remove unwanted noise

% Apply different filters to reduce noise

averageImage = imfilter(grayImage, ...
    fspecial('average', [3 3]));

medianImage = medfilt2(grayImage, [3 3]);

gaussianImage = imgaussfilt(grayImage, 2);

wienerImage = wiener2(grayImage, [5 5]);

bilateralImage = imbilatfilt(grayImage);

figure;
subplot(2,3,1);
imshow(grayImage);
title('Original');

subplot(2,3,2);
imshow(averageImage);
title('Average Filter');

subplot(2,3,3);
imshow(medianImage);
title('Median Filter');

subplot(2,3,4);
imshow(gaussianImage);
title('Gaussian Filter');

subplot(2,3,5);
imshow(wienerImage);
title('Wiener Filter');

subplot(2,3,6);
imshow(bilateralImage);
title('Bilateral Filter');

%% Step 4: Detect image edges

% Find the boundaries and important details in the image

sobelEdges = edge(grayImage, 'Sobel');

prewittEdges = edge(grayImage, 'Prewitt');

cannyEdges = edge(grayImage, 'Canny');

robertsEdges = edge(grayImage, 'Roberts');

logEdges = edge(grayImage, 'log');

figure;
subplot(2,3,1);
imshow(grayImage);
title('Original Image');

subplot(2,3,2);
imshow(sobelEdges);
title('Sobel Edge Detection');

subplot(2,3,3);
imshow(prewittEdges);
title('Prewitt Edge Detection');

subplot(2,3,4);
imshow(cannyEdges);
title('Canny Edge Detection');

subplot(2,3,5);
imshow(robertsEdges);
title('Roberts Edge Detection');

subplot(2,3,6);
imshow(logEdges);
title('LoG Edge Detection');

%% Step 5: Convert image into binary form

% Use different thresholding methods

binaryImage = imbinarize(grayImage);

adaptiveImage = imbinarize(grayImage, 'adaptive');

thresholdValue = graythresh(grayImage);

otsuImage = imbinarize(grayImage, thresholdValue);

figure;
subplot(1,3,1);
imshow(binaryImage);
title('Normal Thresholding');

subplot(1,3,2);
imshow(adaptiveImage);
title('Adaptive Thresholding');

subplot(1,3,3);
imshow(otsuImage);
title('Otsu Thresholding');

%% Step 6: Perform morphological operations

% These operations help to improve the segmented image

structureElement = strel('disk', 3);

erodedImage = imerode(binaryImage, structureElement);

dilatedImage = imdilate(binaryImage, structureElement);

openedImage = imopen(binaryImage, structureElement);

closedImage = imclose(binaryImage, structureElement);

filledImage = imfill(binaryImage, 'holes');

figure;
subplot(2,3,1);
imshow(erodedImage);
title('Erosion');

subplot(2,3,2);
imshow(dilatedImage);
title('Dilation');

subplot(2,3,3);
imshow(openedImage);
title('Opening');

subplot(2,3,4);
imshow(closedImage);
title('Closing');

subplot(2,3,5);
imshow(filledImage);
title('Hole Filling');

%% Step 7: Frequency domain analysis

% Convert the image into the frequency domain

imageFrequency = fft2(double(grayImage));

shiftedFrequency = fftshift(imageFrequency);

frequencySpectrum = log(1 + abs(shiftedFrequency));

figure;
imshow(frequencySpectrum, []);
title('Frequency Spectrum');

% Create a low-pass filter

[imageRows, imageColumns] = size(grayImage);

[columnGrid, rowGrid] = meshgrid(1:imageColumns, 1:imageRows);

distanceFromCenter = sqrt( ...
    (columnGrid - imageColumns/2).^2 + ...
    (rowGrid - imageRows/2).^2);

cutoffValue = 30;

lowPassFilter = double(distanceFromCenter <= cutoffValue);

% Apply the low-pass filter

lowPassResult = real(ifft2(ifftshift( ...
    shiftedFrequency .* lowPassFilter)));

% Apply a high-pass filter

highPassFilter = 1 - lowPassFilter;

highPassResult = real(ifft2(ifftshift( ...
    shiftedFrequency .* highPassFilter)));

figure;
subplot(1,2,1);
imshow(lowPassResult, []);
title('Low Pass Filtering');

subplot(1,2,2);
imshow(highPassResult, []);
title('High Pass Filtering');

%% Step 8: Image segmentation using K-means

% Divide image pixels into three groups

pixelData = im2single(grayImage);

pixelValues = pixelData(:);

numberOfClusters = 3;

clusterLabels = kmeans(pixelValues, numberOfClusters, ...
    'Replicates', 3);

segmentedImage = reshape(clusterLabels, ...
    imageRows, imageColumns);

figure;
imagesc(segmentedImage);
axis image;
axis off;
title('K-Means Image Segmentation');
colorbar;

%% Step 9: Detect important image features

% Detect corners using the Harris method

cornerPoints = detectHarrisFeatures(grayImage);

figure;
imshow(grayImage);
hold on;
plot(cornerPoints.selectStrongest(100));
title('Harris Corner Detection');

% Detect SURF features

surfPoints = detectSURFFeatures(grayImage);

figure;
imshow(grayImage);
hold on;
plot(surfPoints.selectStrongest(50));
title('SURF Feature Detection');

% Extract HOG features

[hogFeatures, hogVisualization] = ...
    extractHOGFeatures(grayImage);

figure;
imshow(grayImage);
hold on;
plot(hogVisualization);
title('HOG Feature Extraction');

%% Step 10: Analyze image texture

% Calculate texture information using GLCM

grayLevelMatrix = graycomatrix(grayImage);

textureResults = graycoprops(grayLevelMatrix, ...
    {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

disp('Image Texture Information:');
disp(textureResults);

%% Step 11: Detect objects in the image

% Remove small unwanted regions

cleanImage = bwareaopen(binaryImage, 30);

% Measure properties of detected objects

objectInformation = regionprops(cleanImage, ...
    'Area', 'Centroid', 'BoundingBox', ...
    'Perimeter', 'Eccentricity');

figure;
imshow(grayImage);
hold on;

for objectNumber = 1:length(objectInformation)

    rectangle('Position', ...
        objectInformation(objectNumber).BoundingBox, ...
        'EdgeColor', 'r', 'LineWidth', 2);

    objectCenter = objectInformation(objectNumber).Centroid;

    plot(objectCenter(1), objectCenter(2), ...
        'g+', 'MarkerSize', 10);
end

title('Detected Objects');

%% Step 12: Sharpen the image

% Highlight important details and boundaries

sharpenedImage = imsharpen(grayImage, ...
    'Radius', 2, 'Amount', 1.5);

figure;
subplot(1,2,1);
imshow(grayImage);
title('Before Sharpening');

subplot(1,2,2);
imshow(sharpenedImage);
title('After Sharpening');

%% Step 13: Image restoration

% Create a motion blur effect

motionFilter = fspecial('motion', 15, 45);

blurredImage = imfilter(grayImage, motionFilter, ...
    'conv', 'circular');

% Restore the blurred image using Wiener deconvolution

restoredImage = deconvwnr(blurredImage, ...
    motionFilter, 0.01);

figure;
subplot(1,3,1);
imshow(grayImage);
title('Original Image');

subplot(1,3,2);
imshow(blurredImage);
title('Blurred Image');

subplot(1,3,3);
imshow(restoredImage, []);
title('Restored Image');

%% Step 14: Measure image quality

% Compare the enhanced image with the original image

originalData = double(grayImage);

enhancedData = double(enhancedImage);

meanSquareError = mean( ...
    (originalData(:) - enhancedData(:)).^2);

if meanSquareError == 0
    psnrValue = Inf;
else
    psnrValue = 10 * log10(255^2 / meanSquareError);
end

ssimValue = ssim(enhancedImage, grayImage);

fprintf('\nIMAGE QUALITY RESULTS\n');

fprintf('Mean Square Error: %.4f\n', ...
    meanSquareError);

fprintf('PSNR: %.4f dB\n', psnrValue);

fprintf('SSIM: %.4f\n', ssimValue);

%% Step 15: Save the final results

imwrite(enhancedImage, 'Enhanced_Image.jpg');

imwrite(cannyEdges, 'Edge_Detection.jpg');

imwrite(otsuImage, 'Segmented_Image.jpg');

imwrite(sharpenedImage, 'Sharpened_Image.jpg');

disp('All image processing operations are completed!');
disp('Processed images have been saved successfully.');
