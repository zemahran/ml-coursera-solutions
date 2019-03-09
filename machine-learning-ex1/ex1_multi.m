%% Machine Learning Online Class
clear; close all; clc
%%
% Load data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out the first 10 data points from the dataset
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
%%
% Scale features
[X, mu, sigma] = featureNormalize(X);
fprintf(' x = [%.03f %.03f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
%%
% Add intercept term to X
X = [ones(m, 1) X];
fprintf(' x = [%.0f %.03f %.03f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
%% ================ Part 2: Gradient Descent ================
% * There is a mistake here (the two predicted prices aren't the same)
% * TODO: try different learning rates or re-check the logic (to solve the above 
% problem)
%%
fprintf('Running gradient descent ...\n');
% gradient descent settings
alpha = 0.01;
num_iters = 400;

% Init theta and run gradient descent 
theta = zeros(size(X, 2), 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0 ; 0]\nCost computed = %f\n', J);
fprintf('Theta computed by gradient descent:\n%f\n%f\n%f', theta);
%%
% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
%%
% Predict the price of a 1650 sq-ft 3-bedroom house (using the updated hypothesis: h(x) = x*theta)
normalized_features = ([1650 3]-mu) ./ sigma; % scaling both features using the same mu & sigma
price = [1 normalized_features] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
%% ================ Part 3: Normal Equation ================
%%
% Reload X (without normalization) and add intercept term X node
X = data(:, 1:2);
X = [ones(m, 1) X];

% Predict the price of a 1650 sq-ft 3-bedroom house (using the the normal equation)
theta = normalEqn(X, y);
price = [1, 1650, 3] * theta;

fprintf('Theta computed by normal equation:\n%f\n%f\n%f', theta);
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equation):\n $%f\n'], price);     
%% External Functions
%% Feature Normalization
% FEATURENORMALIZE normalizes the features in X
% 
% FEATURENORMALIZE(X) returns a normalized version of X where the mean value 
% of each feature is 0 and the standard deviation is 1. This is often a good preprocessing 
% step to do when working with learning algorithms
%%
function [X_norm, mu, sigma] = featureNormalize(X)

mu = zeros(1, size(X, 2)); % 1 x number of columns (features)
sigma = zeros(1, size(X, 2));     

for i = 1:size(X,2)
mu(i) = mean(X(:, i));
sigma(i) = std(X(:, i));
end

X_norm = (X - mu) ./ sigma;

end
%% Cost Function for Multi-Variate Linear Regression
% COMPUTECOSTMULTI computes the cost of using theta as the parameter for linear 
% regression to fit the data points in X and y

function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples

hypothesis = X*theta;
J = (1/(2*m)) * ((hypothesis-y)' * (hypothesis-y));

end
%% Gradient Descent for Linear Regression
% GRADIENTDESCENTMULTI performs gradient descent to learn theta
% 
% theta = GRADIENTDESCENTMULTI(X, y, theta, alpha, num_iters) updates theta 
% by taking num_iters gradient steps with learning rate alpha

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    hypothesis = X*theta;
    theta  = theta - (alpha * (1/m) * (sum((hypothesis-y) .* X))');    

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
%% Normal Equation
% NORMALEQN computes the closed-form solution to linear regression using the 
% normal equations (estimates values of theta in one step)

function [theta] = normalEqn(X, y)

%theta = inv(X'*X) * (X'*y); % slow & unrecommended
theta = (X'*X) \ (X'*y); % inv(A) * B = A \ B

end