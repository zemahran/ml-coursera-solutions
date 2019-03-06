%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%
%% Initialization
%%
clear; close all; clc
%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m
%%
fprintf('Running warmUpExercise ... \n\n5x5 Identity Matrix: \n');
warmUpExercise()
%% ======================= Part 2: Plotting =======================
%%
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);
%% =================== Part 3: Cost and Gradient descent ===================
%%
X = [ones(m, 1), data(:,1)]; % Add a column of ones to X, which corresponds to X node
theta = zeros(2, 1); % initialize fitting parameters

% gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');
% further testing of the cost function
J = computeCost(X, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');
fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n %f\n %f', theta(1), theta(2));
fprintf('Expected theta values (approx):\n -3.6303\n 1.1664\n');
% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure
% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);
%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
%%
fprintf('Visualizing J(theta_0, theta_1) ...\n')
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');
% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
%% External Functions
%% Data Plotting
% PLOTDATA plots the data points x and y into a new figure with axes labels 
% of population and profit
%%
function plotData(x, y)

figure;

plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in 10,000s');
xlabel('City Population in 10,000s');

end
%% Cost Function for Uni-Variate Linear Regression
% COMPUTECOST computes the cost of using theta as the parameter for linear regression 
% to fit the data points in X and y

function J = computeCost(X, y, theta)

m = length(y); % number of training examples

hypothesis = X*theta;
sqrErrors = (hypothesis-y).^2;
J = (1/(2*m)) * sum(sqrErrors);

end
%% Gradient Descent for Linear Regression
% GRADIENTDESCENT performs gradient descent to learn theta
% 
% theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate alpha

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    hypothesis = X*theta;
    theta  = theta - (alpha * (1/m) * (sum((hypothesis-y) .* X))');    

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
end

end