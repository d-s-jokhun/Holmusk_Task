function [J, grad] = Regress_costFunctionReg(theta, X, y, lambda)
% Computes the cost and gradient for a logistic regression with regularization

m = length(y); % number of observations

% Calculating the cost of the regression using the parameters in theta
pre_J= -y.*log(Regress_sigmoid(X*theta)) - (1-y).*log(1-(Regress_sigmoid(X*theta)));
J=sum(pre_J)/m; % J is the cost

% Regularized cost
J=J+((lambda/(2*m))*sum(theta(2:end).^2));


% Calculating the gradient
a=X';
grad0=(a(1,:)*(Regress_sigmoid(X*theta)-y))/m;
grad1=(a(2:end,:)*(Regress_sigmoid(X*theta)-y))/m; % Unregularized gradients
grad1=grad1+ ((lambda/m)*theta(2:end)); % Regularized gradients for non-bias terms

grad=[grad0;grad1]; % Regularized gradients of the logistic regression



end
