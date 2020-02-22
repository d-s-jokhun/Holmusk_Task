function p = NeuNet_predict(Theta1, Theta2, X)
% This function uses the parameters of the 2 layers to predict the class of
% input data

% number of observations
m = size(X, 1);

p = zeros(size(X, 1), 1); % number of output labels

h1 = NeuNet_sigmoid([ones(m, 1) X] * Theta1'); % Output from 1st layer
h2 = NeuNet_sigmoid([ones(m, 1) h1] * Theta2'); % Output from 2nd layer


[~, p] = max(h2, [], 2); % Finding the column which has the maximum output for each observation
p=p-1; % If index 1 was maximum, output=0, If index 2 was maximum, output=1


end
