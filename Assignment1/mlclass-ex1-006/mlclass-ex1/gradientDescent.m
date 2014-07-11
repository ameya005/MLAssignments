function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
	h_th = X * theta;
	%h_size = size(h_th)
	%y_size = size(y)
	temp = zeros(size(theta));
	[t1,t2] = size(theta);
	temp1 = theta;
	
	for(i=1:t1)
		grad = sum( (h_th - y) .* X(:,i) ) ;
		%size(grad)
		temp(i,1) = (1/m) * alpha * grad;
	end			
	theta = theta - temp;	
	
	%computeCost(X,y,theta)




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
