##**~ Predicting stage progression for chronic kidney disease (CKD) ~**

These codes have been written for a data science task. We are provided with a set of longitudinal data of different lab measurements for patients diagnosed with chronic kidney disease (CKD). We are also given the information whether these patients progress in their CKD stage or not in the future. Using this dataset, we are required to come up with a solution to predict whether a patient will progress in CKD staging given the patient's past longitudinal information. All the data are provided in CSV files with filename starting with ‘T_’.

**MainPipeline.mlx** is the overarching script written to accomplish the task. In addition to the common libraries found in Octave/MATLAB, it directly or indirectly requires the following custom functions to run:
model_LDA.m
RegressClassi.m
Regress_costFunctionReg.m
Regress_predict.m
Regress_sigmoid.m
NeuNet.m
NeuNet_fmincg.m
NeuNet_nnCostFunction.m
NeuNet_predict.m
NeuNet_randInitializeWeights.m
NeuNet_sigmoid.m
NeuNet_sigmoidGradient.m




**MainPipeline.mlx**

This is the main pipeline to run for accomplishing the task. It uses the data from the CSV files to train 3 different models, namely, an LDA-based classifier, a logistic regression model and a neural network. Many versions of each model are trained with a range of model parameters. The training and test accuracies are printed with graphical representations as an attempt to determine the optimum parameters for each model. The resulting models are dissected in order to gain some insights about the medical/clinical factors required for the predictions.
MainPipeline.mlx fetches the ‘T_*.csv’ in order to build the starting dataset. It is therefore desirable that MainPipeline.mlx be in the same folder as the ‘T_*.csv’ files. Alternatively, the current folder can be changed to the folder containing the ‘T_*.csv’ files before running MainPipeline.mlx.
All the other functions must be in the same folder as MainPipeline.mlx for the latter to run properly. Alternatively, the folders in which they are stored may be added to the search path before running MainPipeline.mlx.



**model_LDA.m**

model_LDA.m is the function which is called by MainPipeline.mlx to build the linear discriminant classifier. As input, this section requires the training dataset, the training ground truth, the test dataset and the test ground truth. MainPipeline.mlx uses different numbers of principal components of the dataset and finds the optimal parameter for this model.



**RegressClassi.m, Regress_costFunctionReg.m, Regress_predict.m, Regress_sigmoid.m**

RegressClassi.m is the function called by MainPipeline.mlx in order to build the logistic regression model. It in turns directly/indirectly calls all the other ‘Regress_*.m’ functions listed above. The input for RegressClassi.m includes the training dataset, the training ground truth, the test dataset, the test ground truth as well as a list of regularization parameter (lambda). RegressClassi.m will build and assess the model accuracies using each of the regularization parameters. If the regularization parameter is set to 0 or is not provided, the model is trained without regularization. 



**NeuNet.m, NeuNet_fmincg.m, NeuNet_nnCostFunction.m, NeuNet_predict.m, NeuNet_randInitializeWeights.m, NeuNet_sigmoid.m, NeuNet_sigmoidGradient.m**

NeuNet.m is the function called by MainPipeline.mlx in order to build the neural networks. It in turns directly/indirectly calls all the other ‘NeuNet_*.m’ functions listed above. The input for NeuNet.m includes the training dataset, the training ground truth, the test dataset and the test ground truth. Additionally, a list of numbers can be provided for the number of neurons to be used in the hidden layer. A list of numbers can also be provided for the regularization parameter (lambda). Further, a list of numbers can also be provided to set the maximum number of iterations before triggering the stop condition during cost minimization. NeuNet is run with all possible combinations of parameters provided and a graphical representation of the model performance can be obtained as a function of the different sets of parameters.


-	Written in Octave/MATLAB
-	Written by D.S.Jokhun (d.s.jokhun@gmail.com) on 12.02.2020
