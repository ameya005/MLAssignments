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
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%X = [ones(size(X,1),1) X];

h_temp = X*theta;
h_th = sigmoid(h_temp);

J = (-1/m) * sum( y.*log(h_th) + (1 - y).*log(1-h_th));
n = size(grad,1);

temp = h_th - y;


for(j=1:n)
	grad(j) = (1/m)*sum(temp.*X(:,j));
end	
	







% =============================================================

end
