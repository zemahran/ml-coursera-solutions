%% Machine Learning Online Class
clear ; close all; clc
%% ================== Part 1: Load Example Dataset ===================
%%
% 
%  We start this exercise by using a small dataset that is easy to visualize.
%
%%
fprintf('Visualizing example dataset for PCA.\n\n');
% Load the dataset - we now have X in our environment
load ('ex7data1.mat');

% Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]); axis square;
%% =============== Part 2: Principal Component Analysis ===============
%%
fprintf('\nRunning PCA on the example dataset.\n\n');
% Normalize X
[X_norm, mu, sigma] = featureNormalize(X);

% Run PCA
[U, S] = pca(X_norm);

% Compute mu (the mean of the each feature) and Draw the eigenvectors
% centered at mean of data - these lines show the directions of
% maximum variations in the dataset
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;
fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
fprintf('\n(you should expect to see -0.707107 -0.707107)\n');
%% =================== Part 3: Dimension Reduction ===================
%%
% 
%  You should now implement the projection step to map the data onto the 
%  first k eigenvectors. The code will then plot the data in this reduced 
%  dimensional space.  This will show you what the data looks like when 
%  using only the corresponding eigenvectors to reconstruct it.
%
%%
% 
%  You should complete the code in projectData.m
%
%%
fprintf('\nDimension reduction on example dataset.\n\n');
% Plot the normalized dataset (returned from pca)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square

% Project the data onto one dimension - K = 1
K = 1;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));
fprintf('\n(this value should be about 1.481274)\n\n');
X_rec  = recoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
fprintf('\n(this value should be about  -1.047419 -1.047419)\n\n');
%  Draw lines connecting the projected data points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off
%% External Functions

function [X_norm, mu, sigma] = featureNormalize(X)
% FEATURENORMALIZE Normalizes the features in X 
%   It returns a normalized version of X where the mean value of each feature 
%   is 0 and the standard deviation is 1. This is often a good preprocessing
%   step to do when working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end

function [U, S] = pca(X)
% PCA Runs principal component analysis on the dataset X
%   It computes the eigenvectors (U) of the covariance matrix of X
%   as well as the the eigenvalues (S) on diagonal

[m, n] = size(X);
U = zeros(n);
S = zeros(n);

sigma = (1/m)*(X')*X;
[U, S, ~] = svd(sigma);

end

function Z = projectData(X, U, K)
% PROJECTDATA Computes the reduced data representation when projecting only 
%   on to the top k eigenvectors. It represents the reduced dimensional space
%   spanned by the first K columns of U. It returns the projected examples in Z.
%   For the i-th example X(i,:), the projection on to the k-th eigenvector is
%   given as follows: x = X(i, :)'; projection_k = x' * U(:, 1:k);

Z = zeros(size(X, 1), K);

U_reduced = U(:,1:K);
Z = X*U_reduced;

end

function X_rec = recoverData(Z, U, K)
% RECOVERDATA Recovers an approximation of the original data that has been
%   reduced to K dimensions by using the projected data. It returns the
%   approximate reconstruction in X_rec.
%   For the i-th example Z(i,:), the (approximate) recovered data for dimension j
%   is given as follows: v = Z(i, :)'; recovered_j = v' * U(j, 1:K)';

X_rec = zeros(size(Z, 1), size(U, 1));             
X_rec = Z * U(:, 1:K)';

end

function drawLine(p1, p2, varargin)
% DRAWLINE Draws a line from point p1 to point p2

plot([p1(1) p2(1)], [p1(2) p2(2)], varargin{:});

end