function [Model,TrainCorrectRate,TestCorrectRate]=model_LDA(Train_Data,Train_GrndTrth,Test_Data,Test_GrndTrth)
% This function takes in the training data and ground truth to train a
% linear classifier. It then resubmits the training data to evaluate the
% training acuracy.
% It further uses the test dataset to make predictions and compares them 
% with the test ground truth to evaluate the test accuracy.
% It additionally outputs the respective confusion matrices.


% Training the linear classifier using fitcdiscr
Model=fitcdiscr(Train_Data,Train_GrndTrth,'DiscrimType','linear');
%% Train Accuracy
figure
ConfusionMatrix = confusionchart(Train_GrndTrth,resubPredict(Model));
ConfusionMatrix.ColumnSummary = 'column-normalized';
ConfusionMatrix.RowSummary = 'row-normalized';
ConfusionMatrix.Title = 'Classifier accuracy - Train';
cp = classperf(Train_GrndTrth,predict(Model,Train_Data)); % classifier performance
TrainCorrectRate=cp.CorrectRate
%% Test Accuracy
figure
ConfusionMatrix = confusionchart(Test_GrndTrth,predict(Model,Test_Data));
ConfusionMatrix.ColumnSummary = 'column-normalized';
ConfusionMatrix.RowSummary = 'row-normalized';
ConfusionMatrix.Title = 'Classifier accuracy - Test';
cp = classperf(Test_GrndTrth,predict(Model,Test_Data)); % classifier performance
TestCorrectRate=cp.CorrectRate
end