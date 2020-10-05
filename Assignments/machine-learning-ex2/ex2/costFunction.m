function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Cot grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%mpute the partial derivatives and se

X_Theta = X*theta;

J = (1/m)*((-1*y.'*log(sigmoid(X_Theta))+ (-1)*((1-y).'*log( 1- sigmoid(X_Theta)))));

grad = (1/m)*X.'*(sigmoid(X_Theta)-y);


% =============================================================

end
