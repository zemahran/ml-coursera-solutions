%% Machine Learning Online Class
%%
% 
%  Exercise 7 | Principle Component Analysis and K-Means Clustering
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%  
%     pca.m
%     projectData.m
%     recoverData.m
%     computeCentroids.m
%     findClosestCentroids.m
%     kMeansInitCentroids.m
%  
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% 

clear; close all; clc
%% ================= Part 1: Find Closest Centroids ====================
%%
% 
%  To implement K-Means, we have divided the learning algorithm 
%  into two functions -- findClosestCentroids and computeCentroids.
%  First we will use findClosestCentroids.
%  Here, we initialize the initial centroids manually, but later on,
%  we will automatically select random instances from our dataset X.
%
%%
% Load an example dataset that we will be using
load('ex7data2.mat');

% Select an initial set of centroids
K = 3;
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the initial_centroids
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');
%% ===================== Part 2: Compute Means =========================
%%
% 
%  Second, we will use computeCentroids.
%
%%
fprintf('\nComputing centroids means.\n\n');
%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ])\n\n');
%% =================== Part 3: K-Means Clustering ======================
%%
% 
%  After we have completed the two functions computeCentroids and
%  findClosestCentroids, we have all the necessary pieces to run the
%  kMeans algorithm. In this part, we will run the K-Means algorithm on
%  our example dataset.
%
%%
fprintf('\nRunning K-Means clustering on example dataset.\n\n');
% Load an example dataset
load('ex7data2.mat');

% Settings for K-Means
K = 3;
max_iters = 10;
initial_centroids = [1 8; 8 1; 7 7];

% Run K-Means algorithm & plot progress
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');
%% ============= Part 4: K-Means Clustering on Pixels ===============
%%
% 
%  In this exercise, we will use K-Means to compress an image. To do this,
%  we will first run K-Means on the colors of the pixels in the image and
%  then we will map each pixel onto its closest centroid.
%
%%
fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');
%  Load an image of a bird
A = double(imread('bird_small.png'));

% Divide by 255 so that all values are in the range 0:1
A = A/255;

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on
X = reshape(A, img_size(1) * img_size(2), 3);

% Settings for K-means
K = 16; 
max_iters = 10;

% Initialize centroids randomly
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);
%% ================= Part 5: Image Compression ======================
%%
% 
%  In this part of the exercise, we will use the clusters of K-Means to
%  compress an image. To do this, we first find the closest clusters for
%  each example. After that, we will recover the image pixel values according
%  to our final computed centroids.
%
%%
fprintf('\nApplying K-Means to compress an image.\n\n');
% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% now we have represented the image X in terms of the values in idx
% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions (r,g,b)
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display both images side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed with %d colors.', K));
%% External Functions
%%
function idx = findClosestCentroids(X, centroids)
% FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])

square_diff = zeros(size(centroids,1), size(centroids,2));
sum_square_diff = zeros(size(centroids,1), 1);
idx = zeros(size(X,1), 1);

fun = @(x,c)(x-c).^2;

for i=1:size(X,1)
    square_diff = bsxfun(fun, X(i,:), centroids);
    sum_square_diff = sum(square_diff, 2);
    idx(i) = find(sum_square_diff == min(sum_square_diff), 1);
end

end

function centroids = computeCentroids(X, idx, K)
% COMPUTECENTROIDS returns the new centroids by computing the means of the 
%   data points assigned to each centroid. It returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. It returns a matrix of
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.

centroids = zeros(K, size(X,2));

for k=1:K
    centroids(k,:) = mean(X(idx==k, :));
end

end

function centroids = kMeansInitCentroids(X, K)
% KMEANSINITCENTROIDS initializes K centroids that are to be used in
%   K-Means on the dataset X. It sets centroids to randomly chosen examples from X

centroids = X(randi(size(X,1), [K, 1]), :);

end

function [centroids, idx] = runkMeans(X, initial_centroids, ...
                                      max_iters, plot_progress)
% RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
%   is a single example. It uses initial_centroids used as the
%   initial centroids. max_iters specifies the total number of interactions 
%   of K-Means to execute. plot_progress is a true/false flag that 
%   indicates if the function should also plot its progress as the 
%   learning happens. This is set to false by default. runkMeans returns 
%   centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])

% Set default value for plot_progress
if ~exist('plot_progress', 'var') || isempty(plot_progress)
    plot_progress = false;
end

% Plot the data if we are plotting progress
if plot_progress
    figure;
    hold on;
end

% Initialize values
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(size(X,1), 1);

% Run K-Means
for i=1:max_iters
    
    % Output progress
    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
    
    % For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X, centroids);
    
    % Plot progress
    if plot_progress
        plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
        previous_centroids = centroids;
    end
    
    % Given the memberships, compute new centroids
    centroids = computeCentroids(X, idx, K);
end

% Hold off if we are plotting progress
if plot_progress
    hold off;
end

end

function plotProgresskMeans(X, centroids, previous, idx, K, i)
% PLOTPROGRESSKMEANS is a helper function that displays the progress of 
%   k-Means as it is running. It is intended for use only with 2D data.
%   It plots the data points with colors assigned to each centroid cluster.
%   With the previous centroids, it also plots a line between the previous locations and
%   current locations of the centroids.

% Plot the examples
plotDataPoints(X, idx, K);

% Plot the centroids as black x's
plot(centroids(:,1), centroids(:,2), 'x', ...
     'MarkerEdgeColor','k', ...
     'MarkerSize', 10, 'LineWidth', 3);

% Plot the history of the centroids with black lines
for j=1:size(centroids,1)
    drawLine(centroids(j, :), previous(j, :), 'color', 'black');
end

% Set title
title(sprintf('Iteration number %d', i))

end