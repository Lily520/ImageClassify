function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% J
h = zeros(m,1);
h = sigmoid(X * theta);
error = y .* log(h) + (1-y) .* log(1-h);
sum_error = -(1/m) * sum(error);
error_theta = (lambda /(2*m)) * (sum(theta .^ 2) - theta(1)*theta(1));
J = sum_error + error_theta;

%grad
row = size(theta,1);
g = sigmoid(X * theta);
grad(1) = (1/m) * sum((g-y) .* X(:,1));
for j = 2:row
grad(j) = (1/m) * sum((g-y) .* X(:,j)) + (lambda/m) * theta(j);




% =============================================================

end
