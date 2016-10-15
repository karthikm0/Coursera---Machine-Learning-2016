function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  temp_0 = theta(1) - (alpha/m) * sum(X*theta-y); % new theta_0
  temp_1 = theta(2) - (alpha/m) * sum((X*theta-y).*X(:,2)); % new theta_1
  theta(1) = temp_0; % store new thetas
  theta(2) = temp_1; % store new thetas
  J_history(iter) = computeCost(X, y, theta); % store cost function after iteration
  fprintf('%f \n', J_history(iter));
end
end
