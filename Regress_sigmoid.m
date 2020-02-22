function g = NeuNet_sigmoid(z)
% This function calculates the sigmoid of every element of input z

g = 1.0 ./ (1.0 + exp(-z));
end
