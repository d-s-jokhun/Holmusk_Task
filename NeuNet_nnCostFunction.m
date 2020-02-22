function [J grad] = NeuNet_nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Neural network cost function for a two layer neural network
% Computes the cost and gradient of the network.


% Reconstructing the two layers of the network from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1); % number of obversations
         
J = 0;  % J is the cost
Theta1_grad = zeros(size(Theta1)); % gradient for the 1st layer
Theta2_grad = zeros(size(Theta2)); % gradient for the 2nd layer


Y=zeros(m,num_labels); % Y will be a 2-column binary representatio of the class
% [1,0] represents 0 and [0,1] represents 1
for count=1:m
  Y(count,y(count,1)+1)=1;
end

% Calculating the output of the network
  a1=X;
  z2=[ones(m,1),a1]*Theta1';
  a2=NeuNet_sigmoid(z2);
  z3=[ones(m,1),a2]*Theta2';
  hk=NeuNet_sigmoid(z3);
  a3=hk;

  
  
  J= (-Y.*log(hk))-((1-Y).*log(1-hk));
J=sum(sum(J))/m;  
% Unregularized cost of the network

% Implementing regularization
Theta1_sq=Theta1.^2;
Theta2_sq=Theta2.^2;
Sum_Theta_sq=sum(sum(Theta1_sq(:,2:end)))+sum(sum(Theta2_sq(:,2:end)));
Reg_term=(lambda/(2*m))*Sum_Theta_sq;

% Regularized cost
J=J+Reg_term;

% Used for back propagating the errors
DELTA1=zeros(size(Theta1));
DELTA2=zeros(size(Theta2));

for t=1:m % Looping through each observation and accumulating the cost 
  
  delta3=a3(t,:)-Y(t,:);
  
  delta2=(Theta2'*delta3');
  delta2=delta2(2:end,:).*NeuNet_sigmoidGradient(z2(t,:))';
  
  DELTA1=DELTA1+(delta2*[1,a1(t,:)]); % Cost accumulation for the the 1st layer
  DELTA2=DELTA2+(delta3'*[1,a2(t,:)]);  % Cost accumulation for the the 2nd layer
  

  

  
end


  % Regularized gradients of the first layer
  Theta1_grad=DELTA1/m;
  Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+( (lambda/m)*Theta1(:,2:end) );
  
  
  
  
  % Regularized gradients of the second layer
  Theta2_grad=DELTA2/m;
  Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+( (lambda/m)*Theta2(:,2:end) );

% Unrolling the gradients for output
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
