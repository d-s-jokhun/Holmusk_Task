function [NeuNet_TrainPredictions,NeuNet_TestPredictions,Train_Accuracy,Test_Accuracy,Weights]=NeuNet(TrainData,Train_GrndTrth,TestData,Test_GrndTrth,hidden_layer_size,lambda,MaxIterate)

% This function will train a neural network using the train data and train
% ground truth. It will also evaluate the model using the test data and
% test ground truth.
% Additional arguments include setting the number of units to be used in
% the hidden layer, the regularization parameter and the maximum iteration
% which trigers the stop condition.
% It directly/indirectly requires the following custom functions:
% sigmoidGradient, randInitializeWeights and nnCostFunction

input_layer_size  = size(TrainData,2);
num_labels = numel(unique(categorical(Train_GrndTrth)));    

% If lambda is not given, it will be taken as zero (no regularization)
if isempty(lambda)
    lambda = 0;
end



% Initializing the weights of the neural network
initial_Theta1 = NeuNet_randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = NeuNet_randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% creating costFunction, a shortcut for NeuNet_nnCostFunction such that we have to
% provide only 1 argument later (arg1: p)
costFunction = @(p) NeuNet_nnCostFunction(p, ...  % p will be the parameters to optimized
                                   input_layer_size, ...  % used to recreate the two layers from the unrolled list of parameters
                                   hidden_layer_size, ...
                                   num_labels, TrainData, Train_GrndTrth, lambda);


% Using NeuNet_fmincg to minimize costFunction by optimizing
% initial_nn_params
% The output are the optimized parameters as well as the cost
options = optimset('MaxIter', MaxIterate);  % setting the stop condition
[nn_params, cost] = NeuNet_fmincg(costFunction, initial_nn_params, options);

% Obtaining Theta1 and Theta2 from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Can be used to visualize the parameters being given importance by the
% network
Weights= (Theta1(:, 2:end));


% NeuNet_predict uses the trained network parameters and predicts the
% progression state of some input data
NeuNet_TrainPredictions = NeuNet_predict(Theta1, Theta2, TrainData); % Model prediction of the train data
NeuNet_TestPredictions = NeuNet_predict(Theta1, Theta2, TestData); % Model prediction of the test data

% Calculating the prediction accuracy for the training and test datasets
Train_Accuracy= mean(double(NeuNet_TrainPredictions == Train_GrndTrth)) * 100;
Test_Accuracy= mean(double(NeuNet_TestPredictions == Test_GrndTrth)) * 100;


end

