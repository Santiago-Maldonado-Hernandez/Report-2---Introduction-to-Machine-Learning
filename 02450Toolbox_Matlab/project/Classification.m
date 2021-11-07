%% Initialization
clear all; 
clc;
data_ini;

%% Classification
% The classification problem will predict wheter an abalone has 10 or more
% rings. We set ring attribute as output 'y' while the rest of the data
% matrix 'X' is set as input. 
y = X(:, end);
X(:, end) = [];
% The ring attribute goes from 3-17. We will make a binary problem dividing 
% the problem in if the abalone has 10 or more rings. '1' if it does it and
% '0' if it doesn't 
y = double(y>10 | y==10);

% include an additional attribute corresponding to the offset
[N, M] = size(X);
attributeNames=attributeNames(2:end-1);

K = 10;
CV = cvpartition(N, 'kFold', K);

C = 2;

Error_test = zeros(1, K);
lambda_opt = zeros(1, K);
Error_test_no_features = zeros(1, K);

for kk=1:K
    fprintf('Crossvalidation fold %d/%d\n', kk, K);
    % Extract the training and test set
    X_train = X(CV.training(kk), :);
    y_train = y(CV.training(kk));
    X_test = X(CV.test(kk), :);
    y_test = y(CV.test(kk));
    
    [N_test, p] = size(X_test);
    [N_train, q] = size(X_train);

    % Standardize the data
    mu = mean(X_train,1);
    sigma = std(X_train,1);
    X_train_std = bsxfun(@times, X_train - mu, 1./ sigma);
    X_test_std = bsxfun(@times, X_test - mu, 1./ sigma);

    %% Fit model
    % Fit regularized logistic regression model to training data to predict 
    % the type of wine
    lambda = logspace(-8,0,50);
    test_error_rate = nan(length(lambda),1);
    train_error_rate = nan(length(lambda),1);
    coefficient_norm = nan(length(lambda),1);
    for k = 1:length(lambda)
        mdl = fitclinear(X_train_std, y_train, ...
                     'Lambda', lambda(k), ...
                     'Learner', 'logistic', ...
                     'Regularization', 'ridge');
        [y_train_est, p] = predict(mdl, X_train_std);
        train_error_rate(k) = sum( y_train ~= y_train_est ) / length(y_train);

        [y_test_est, p] = predict(mdl, X_test_std);
        test_error_rate(k) = sum( y_test ~= y_test_est ) / length(y_test);

        coefficient_norm(k) = norm(mdl.Beta,2);
    end
    [min_error,lambda_idx] = min(test_error_rate);
    Error_test(1, kk) = min_error;
    lambda_opt(1, kk) = lambda(lambda_idx);
    
    % Baseline no features
    y_test_est_no_features = ones(length(y_test), 1);
    y_test_est_no_features = y_test_est_no_features*mode(y_test);
    Error_test_no_features(1, kk) = sum( y_test ~= y_test_est_no_features ) / length(y_test);
    
    %%
    mfig("Logistic regression: optimal regularization strength. Fold K = "+kk); clf;
    subplot(3,1,1)
        semilogx(lambda, test_error_rate*100)
        hold on
        semilogx(lambda, train_error_rate*100)
        xlabel('Regularization strength (\lambda)')
        ylabel('Error rate (%)')
        ylim([10 35])
        xlim([min(lambda) max(lambda)])
        title('Error rate')
        legend(['Test error, n=', num2str(length(y_test))], ...
               ['Training Error, n=', num2str(length(y_train))])
        grid minor

    subplot(3,1,2)
        semilogx(lambda, test_error_rate*100)
        hold on
        semilogx(lambda, train_error_rate*100)
        title('Error rate (zoom)')
        xlabel('Regularization strength (\lambda)')
        ylabel('Error rate (%)')
        xlim([1e-6 1e0])
        ylim([10 35])
         text(1.5e-6, 2.25, ['Minimum test error: ', num2str(min_error*100,3), ' % at \lambda=1e', ...
            num2str(log10(lambda(lambda_idx)),2)], ...
            'FontSize',16)
        grid minor

    subplot(3,1,3)
        semilogx(lambda, coefficient_norm,'k')
        xlabel('Regularization strength (\lambda)')
        ylabel('L2 norm of parameter vector')
        title('Parameter vector norm')
        xlim([min(lambda) max(lambda)])
        grid minor
end

%% ANN

% Parameters for neural network classifier
NHiddenUnits = 10;  % Number of hidden units
NTrain = 10; % Number of re-trains of neural network

% Variable for classification error
Error = nan(K,1);
Error_test_ANN = nan(K,1);
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
    Error_test_ANN(k) = Error(k)/length(y_test); 
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

%% Table 
Outer_fold = 1:K;
varNames = {'Outer fold i', 'E test ANN', 'Lambda opt', 'E test linear regression', 'E test baseline'};
stats = table(Outer_fold', Error_test_ANN, lambda_opt', Error_test', Error_test_no_features', 'VariableNames', varNames);