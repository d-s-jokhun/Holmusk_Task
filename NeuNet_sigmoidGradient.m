function g = NeuNet_sigmoidGradient(z)
% This function returns the gradient of the sigmoid function for each element of z

sig=NeuNet_sigmoid(z); % Calculating the sigmoid
g=sig.*(1-NeuNet_sigmoid(z)); % Calculating the gradient



end
