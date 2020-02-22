
function [Reg_TrainPredictions,Reg_TestPredictions]=RegressClassi(TrainData,TestData,Lambda,Train_GrndTrth,Test_GrndTrth)
% This function will train a Logistic regression based classifier using the
% train data and train ground truth. It will also evaluate the model using
% the test data and test ground truth.
% Additional argument include the regularization parameter
% It directly/indirectly requires the following custom functions:
% Regress_costFunctionReg, Regress_predict and Regress_sigmoid


Reg_TrainPredictions=[]; % This will contain the model predictions of the training dataset
Reg_TestPredictions=[]; % This will contain the model predictions of the test dataset

if isempty(Lambda)
    Lambda=0; % If the regularization term is not provided, it will be taken as zero (No regularization)
end

NumOfRuns=numel(Lambda); % The function will train and test a logistic regression model using each of the regularization parameter provided

for run_count=1:NumOfRuns
    
    initial_theta = zeros(size(TrainData, 2), 1);  % setting the initial weights (untrained) to zero
    
    options = optimset('GradObj', 'on', 'MaxIter', 1000,'TolX',10^-7);  % Options for fminunc (minimization function)
    [theta, J, exit_flag] = ...
        fminunc(@(t)(Regress_costFunctionReg(t, TrainData, Train_GrndTrth, Lambda(run_count))), initial_theta, options);
    % fminunc will minimize the cost as given by Regress_costFunctionReg
    % by optimizing the parameters in t
    % initial_theta is submitted as t
    
    
    % Using the optimized theta to make predictions on the trainin and test
    % datasets
    % Concatenating the prediction sets for each regularization parameter
    % provided
    Reg_TrainPredictions = [Reg_TrainPredictions,Regress_predict(theta, TrainData)];
    Reg_TestPredictions = [Reg_TestPredictions,Regress_predict(theta, TestData)];
    % Can be perallocated for speed
    
end
   fprintf('Train Accuracy: %f\n', mean(double(Reg_TrainPredictions == Train_GrndTrth)) * 100);
   % Printing the training accuracies obtained with every regularization
   % parameter provided
    
    if ~isempty(Test_GrndTrth)
        % if the test ground truth is provided
        fprintf('Test Accuracy: %f\n', mean(double(Reg_TestPredictions == Test_GrndTrth)) * 100);
        % Printing the test accuracies obtained with every 
        % regularization parameter provided
    end

end
