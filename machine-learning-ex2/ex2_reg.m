%% Machine Learning Online Class - Exercise 2: Logistic Regression
clear ; close all; clc
%% Load Data
%%
% 
%  The first two columns contains the X values and the third column
%  contains the label (y).
%
%%
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
%%
plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;
%% =========== Part 1: Regularized Logistic Regression ============
% * The cost function implemented is wrong. It doesn't return the correct minimized 
% cost (J).
% * TODO: ditch vectorization and try re-implementing the corresponding equations 
% using normal loops.
%%
% 
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%%
% 
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%
%%
% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));
%%
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression - expected: cost=0.693, grad= 0.0085 0.0188 0.0001 0.0503 0.0115
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost(1));
fprintf(['Gradient at initial theta (zeros) - first five values only: %f %f %f %f %f'], grad(1:5));
%%
% Compute and display cost and gradient
% with all-ones theta and lambda = 10
% expected: cost= 3.16, grad= 0.3460 0.1614 0.1948 0.2269 0.0922
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost(1));
fprintf(['Gradient at test theta - first five values only: %f %f %f %f %f'], grad(1:5));
%% External Functions
%% Feature Mapping
% One way to fit the data better is to create more features from each data point. 
% mapFeature(X1,X2) will map the features into all polynomial terms of X1 and 
% X2 up to the sixth power (higher-order ploynomials) - refer to the handout
%%
function out = mapFeature(X1, X2)

%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end
%% Regularized Logistic Regression
% Compute cost and gradient for logistic regression with regularization

function [J, grad] = costFunctionReg(theta, X, y, lambda)

%   COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);
theta(1) = 0;
J = (1/m) * (-(y') * log(h) - (1-y)' * log(1-h)) + (lambda/(2*m)) * theta.^2;
grad = (1/m) * X' * (h - y) + (lambda/m) * theta;

end