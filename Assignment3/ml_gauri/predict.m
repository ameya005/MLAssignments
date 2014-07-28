function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
n = size(X, 2); %X = m x n = 5000 x 400
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % p = m x 1 = 5000 x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m,1) X]; % n x (m+1) = 5000x401
z2 = a1*(Theta1'); % 5000x401*401x25
a2 = sigmoid(z2); %5000x25
a2 = [ones(m,1) a2]; %5000x26
z3 = a2*(Theta2'); %5000x26*26x10
a3 = sigmoid(z3); %5000x10
h = a3; %5000x10

[prob,p] = max(h,[],2); %5000x1






% =========================================================================


end