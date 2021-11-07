%% Initialization
clear all; 
clc;
data_ini;

y = X(:, end);
X(:, end) = [];
% The ring attribute goes from 3-17. We will make a binary problem dividing 
% the problem in if the abalone has 10 or more rings. '1' if it does it and
% '0' if it doesn't 
y = double(y>10 | y==10);

% include an additional attribute corresponding to the offset
[N, M] = size(X);
attributeNames=attributeNames(2:end-1);

% K-fold crossvalidation
K = 10; 
CV = cvpartition(y, 'Kfold', K);

% Parameters for neural network classifier
NHiddenUnits = 10;  % Number of hidden units
NTrain = 10; % Number of re-trains of neural network

% Variable for classification error
Error = nan(K,1);
bestnet = cell(K,1); 

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
  
    % Fit neural network to training set
    MSEBest = inf; 
    for t = 1:NTrain   
        netwrk = nc_main(X_train, y_train, X_test, y_test, NHiddenUnits);
        if netwrk.Etrain(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.Etrain(end); end
    end
    
    % Predict model on test data    
    y_test_est = bestnet{k}.t_est_test>.5;    
    
    % Compute error rate
    Error(k) = sum(y_test~=y_test_est); % Count the number of errors
end


% Print the error rate
fprintf('Error rate: %.1f%%\n', sum(Error)./sum(CV.TestSize)*100);

% Display the trained network 
mfig('Trained Network');
clf;
k=1; % cross-validation fold
displayNetworkClassification(bestnet{k});


% Display the decision boundary (use only for two class classification problems)
if size(X_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkClassification(X_train, y_train, X_test, y_test, bestnet{k});
end