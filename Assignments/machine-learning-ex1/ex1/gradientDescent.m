function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    hypotesis = X*theta;
    
    errors = hypotesis - y;
    
    Grad_0 = (1/m) * sum(errors.*X(:,1));

    Grad_1 = (1/m) * sum(errors.*X(:,2));

    Grad = zeros(2,1);
    
    Grad(1) = Grad_0;
    
    Grad(2) = Grad_1;
    
    theta = theta - (alpha * Grad );

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
