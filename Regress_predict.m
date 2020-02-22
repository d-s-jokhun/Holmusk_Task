function p = Regress_predict(theta, X)
% This function uses the parameters in theta to predict the class of
% input data
% The threshold is set at 0.5; If sigmoid(theta'*x) >=0.5, predict 1 else 0

% Binarized output based on a 0.5 threshold
p=Regress_sigmoid(X*theta) >= 0.5;


end
