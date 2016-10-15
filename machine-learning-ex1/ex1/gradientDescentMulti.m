function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp = zeros(size(theta));

for iter = 1:num_iters
  for i = 1:(size(X,2))
    temp(i) = theta(i) - (alpha/m) * sum((X*theta-y).*X(:,i)); % determine new thetas
  end
theta = temp; %assign new thetas 
J_history(iter) = computeCostMulti(X, y, theta);  % store cost function after iteration
end

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    

end
