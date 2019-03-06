%% Machine Learning Online Class - Exercise 2: Logistic Regression
%%
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%  
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%  
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% 
%%
clear ; close all; clc
%% Load Data
%%
% 
%  The first two columns contains the exam scores and the third column
%  contains the label.
%
%%
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
%% ==================== Part 1: Plotting ====================
%%
% 
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.
%
%%
% Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples

plotData(X, y);

hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
%% ============ Part 2: Compute Cost and Gradient ============
%%
% 
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m
%
%%
% Add intercept term to X
[m, n] = size(X);
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% at initial theta, expected cost = 0.693
% expected updated theta values (applying gd on initial theta) = -0.1000 -12.0092 -11.2628

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf(['Gradient at initial theta (zeros): %f %f %f'], grad);
%%
% Compute and display cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];

% at test theta, expected cost = 0.218
% expected updated theta values (applying gd on test theta) = 0.043 2.566 2.647

[cost, grad] = costFunction(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);
fprintf(['Gradient at test theta: %f %f %f'], grad);
%% ============= Part 3: Optimizing using fminunc =============
%%
% 
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.
%
%%
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen - expected: theta=0.203 & grad= -25.161 0.206 0.201
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf(['theta: %f %f %f'], theta);
%%
% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
%% ============== Part 4: Predict and Accuracies ==============
%%
% 
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%%
% 
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%%
% 
%  Your task is to complete the code in predict.m
%
%%
%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 - expected value: 0.775 +/- 0.002

prob = sigmoid([1 45 85] * theta); % hypothesis
fprintf(['Probability: %f'], prob);
% Predict on all X rows (all training set) using theta
p = predict(theta, X);

% Compute accuracy on our training set - expected (approx) = 89.0
% by taking mean of all correct predictions:
% considering how many times p was right (= to y)
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
%% External Functions
%% Sigmoid Function
% Compute the sigmoid of each value of z (z can be a matrix, vector or scalar)
%%
function g = sigmoid(z)

g = zeros(size(z));
g = 1./(1+exp(-z));
    
end
%% Cost Function for Logistic Regression
% Compute cost and gradient for logistic regression, using an initial theta 
% as a parameter

function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Note: grad has the same dimensions as theta

h = sigmoid(X * theta); % hypothesis
J = (1/m) * (-(y') * log(h) - (1-y)' * log(1-h)); % minimized cost fn
grad = (1/m) * X' * (h - y); % updated theta

end
%% Predict
% Compute predictions upon our dataset: predict whether the label is 0 or 1 
% using learned logistic regression parameters

function p = predict(theta, X)

%   PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % number of training examples
p = zeros(m, 1);

% p should be set to a vector of 0's and 1's

for i = 1:m
    if (sigmoid(X(i,:) * theta)) >= 0.5
        p(i) = 1;
    else
        p(i) = 0;
    end
end

end
%% Plot Data
% Plot the data points X and y into a new figure

function plotData(X, y)

%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create new figure
figure; hold on;

% Plot the positive and negative examples on a 2D plot,
% using the option 'k+' for the positive examples and 'ko' for the negative examples

% Find indices of positive and negative examples
pos = find(y==1); neg = find(y == 0);

% Plot examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

hold off;

end
%% Plot Decsion Boundary
% Plot the data points X and y into a new figure with the decision boundary 
% defined by theta

function plotDecisionBoundary(theta, X, y)

%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end

hold off

end