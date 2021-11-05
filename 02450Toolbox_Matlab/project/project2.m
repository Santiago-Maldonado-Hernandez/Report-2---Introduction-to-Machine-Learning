%% Initialization
clear all; 
clc;
data_ini;

%% Regresion
% The regression problem will predict the length of the abalone based on 
% other attributes except the ring attribute. Remove the last column that
% corresponds with that attribute. 
X(:, end) = [];
% Set the lenght as the output 'y' that we want to predict and extract it
% from the dta matrix 'X'
y = X(:, 1);
X(:, 1) = [];

%% Part a: 1
% include an additional attribute corresponding to the offset
if (first_time==1)
[N, M] = size(X);
X=[ones(size(X,1),1) X];
M=M+1;
end
first_time=0;

attributeNames={'Offset', attributeNames{1:end}};

% Number of folds in the K-fold crossvalidation 
K = 10;
CV = cvpartition(N, 'Kfold', K);

% Values of lambda
lambda_tmp=10.^(-5:8);

% Initialize variables
T=length(lambda_tmp);
Error_train = nan(K,1);
Error_test = nan(K,1);
w = nan(M,T,K);
lambda_opt = nan(K,1);
w_rlr = nan(M,K);

for k=1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    Xty = X_train' * y_train;
    XtX = X_train' * X_train;
    for t=1:length(lambda_tmp)   
        % Learn parameter for current value of lambda for the given
        % inner CV_fold
        regularization = lambda_tmp(t) * eye(M);
        regularization(1,1) = 0; % Remove regularization of bias-term
        w(:,t,k)=(XtX+regularization)\Xty;
        % Evaluate training and test performance
        Error_train(t,k) = sum((y_train-X_train*w(:,t,k)).^2);
        Error_test(t,k) = sum((y_test-X_test*w(:,t,k)).^2);
    end   

    % Select optimal value of lambda
    [val,ind_opt]=min(sum(Error_test,2)/sum(CV.TestSize));
    lambda_opt(k)=lambda_tmp(ind_opt);    

    % Display result for last cross-validation fold (remove if statement to
    % show all folds)
    %if k == K
        mfig(sprintf('Regularized Solution for linear regression',k));    
        subplot(1,2,1); % Plot error criterion
        semilogx(lambda_tmp, mean(w(2:end,:,:),3),'.-');
        % For a more tidy plot, we omit the attribute names, but you can
        % inspect them using:
        %legend(attributeNames(2:end), 'location', 'best');
        xlabel('\lambda');
        ylabel('Coefficient Values');
        title('Values of w');
        subplot(1,2,2); % Plot error        
        loglog(lambda_tmp,[sum(Error_train,2)/sum(CV.TrainSize) sum(Error_test,2)/sum(CV.TestSize)],'.-');   
        legend({'Training Error as function of lambda','Test Error as function of lambda'},'Location','SouthEast');
        title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(k)))]);
        xlabel('\lambda');           
        drawnow;    
    %end
    
    mu(k,  :) = mean(X_train(:,2:end));
    sigma(k, :) = std(X_train(:,2:end));

    X_train_std = X_train;
    X_test_std = X_test;
    X_train_std(:,2:end) = (X_train(:,2:end) - mu(k , :)) ./ sigma(k, :);
    X_test_std(:,2:end) = (X_test(:,2:end) - mu(k, :)) ./ sigma(k, :);

    % Estimate w for the optimal value of lambda
    Xty=(X_train_std'*y_train);
    XtX=X_train_std'*X_train_std;

    regularization = lambda_opt(k) * eye(M);
    regularization(1,1) = 0; 
    w_rlr(:,k) = (XtX+regularization)\Xty;
end

disp('The program is in pause, click the space bar to continue');
pause;

%% Part b: 2-Level Cross-Validation
% To run this section firs run the Initializatin section
if (first_time==1)
X(:, end) = [];
y = X(:, 1);
X(:, 1) = [];
% include an additional attribute corresponding to the offset
[N, M] = size(X);
X=[ones(size(X,1),1) X];
M=M+1;
end
first_time=0;
attributeNames={'Offset', attributeNames{3:end-1}};
 
% Crossvalidation
% Create crossvalidation partition for evaluation of performance of optimal
% model
K = 10;
CV = cvpartition(N, 'Kfold', K);

% Values of lambda
lambda_tmp=10.^(-9:6);

% Initialize variables
T=length(lambda_tmp);
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_rlr = nan(K,1);
Error_test_rlr = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
Error_train2 = nan(T,K);
Error_test2 = nan(T,K);
w = nan(M,T,K);
lambda_opt = nan(K,1);
w_rlr = nan(M,K);
mu = nan(K, M-1);
sigma = nan(K, M-1);
w_noreg = nan(M,K);

% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Use 10-fold crossvalidation to estimate optimal value of lambda    
    KK = 10;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    for kk=1:KK
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
        % Standardize the training and test set based on training set in
        % the inner fold
        mu2 = mean(X_train2(:,2:end));
        sigma2 = std(X_train2(:,2:end));
        X_train2(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
        X_test2(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;
        
        Xty2 = X_train2' * y_train2;
        XtX2 = X_train2' * X_train2;
        for t=1:length(lambda_tmp)   
            % Learn parameter for current value of lambda for the given
            % inner CV_fold
            regularization = lambda_tmp(t) * eye(M);
            regularization(1,1) = 0; % Remove regularization of bias-term
            w(:,t,kk)=(XtX2+regularization)\Xty2;
            % Evaluate training and test performance
            Error_train2(t,kk) = sum((y_train2-X_train2*w(:,t,kk)).^2);
            Error_test2(t,kk) = sum((y_test2-X_test2*w(:,t,kk)).^2);
        end
    end    
    
    % Select optimal value of lambda
    [val,ind_opt]=min(sum(Error_test2,2)/sum(CV2.TestSize));
    lambda_opt(k)=lambda_tmp(ind_opt);    

    % Display result for last cross-validation fold (remove if statement to
    % show all folds)
    if k == K
        mfig(sprintf('(%d) Regularized Solution',k));    
        subplot(1,2,1); % Plot error criterion
        semilogx(lambda_tmp, mean(w(2:end,:,:),3),'.-');
        % For a more tidy plot, we omit the attribute names, but you can
        % inspect them using:
        legend(attributeNames(2:end), 'location', 'best');
        xlabel('\lambda');
        ylabel('Coefficient Values');
        title('Values of w');
        subplot(1,2,2); % Plot error        
        loglog(lambda_tmp,[sum(Error_train2,2)/sum(CV2.TrainSize) sum(Error_test2,2)/sum(CV2.TestSize)],'.-');   
        legend({'Training Error as function of lambda','Test Error as function of lambda'},'Location','SouthEast');
        title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(k)))]);
        xlabel('\lambda');           
        drawnow;    
    end
    
    % Standardize datasets in outer fold, and save the mean and standard
    % deviations since they're part of the model (they would be needed for
    % making new predictions)
    mu(k,  :) = mean(X_train(:,2:end));
    sigma(k, :) = std(X_train(:,2:end));

    X_train_std = X_train;
    X_test_std = X_test;
    X_train_std(:,2:end) = (X_train(:,2:end) - mu(k , :)) ./ sigma(k, :);
    X_test_std(:,2:end) = (X_test(:,2:end) - mu(k, :)) ./ sigma(k, :);
        
    % Estimate w for the optimal value of lambda
    Xty=(X_train_std'*y_train);
    XtX=X_train_std'*X_train_std;
    
    regularization = lambda_opt(k) * eye(M);
    regularization(1,1) = 0; 
    w_rlr(:,k) = (XtX+regularization)\Xty;
    
    % evaluate training and test error performance for optimal selected value of
    % lambda
    Error_train_rlr(k) = sum((y_train-X_train_std*w_rlr(:,k)).^2);
    Error_test_rlr(k) = sum((y_test-X_test_std*w_rlr(:,k)).^2);
    
    % Compute squared error without regularization
    w_noreg(:,k)=XtX\Xty;
    Error_train(k) = sum((y_train-X_train_std*w_noreg(:,k)).^2);
    Error_test(k) = sum((y_test-X_test_std*w_noreg(:,k)).^2);
    
    % Compute squared error without using the input data at all (Baseline)
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);
     
end

% Display results
fprintf('\n');
fprintf('Linear regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
fprintf('Regularized linear regression:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train_rlr)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test_rlr)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train_rlr))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test_rlr))/sum(Error_test_nofeatures));

fprintf('\n');
fprintf('Weight in last fold: \n');
for m = 1:M
    disp( sprintf(['\t', attributeNames{m},':\t ', num2str(w_rlr(m,end))]))
end
disp(w_rlr(:,end))

%% neural network

% predict length of abalone
if (first_time==1)
y=X(:,1);
X(:, end) = [];
X(:,1)=[];
[N, M] = size(X);
% X=[ones(size(X,1),1) X];
% M=M+1;
end
first_time=0;

attributeNames={'Offset', attributeNames{1:end}};

% K-fold crossvalidation
K = 10;
CV = cvpartition(N,'Kfold', K);

% Parameters for neural network classifier
% NHiddenUnits = 10;  % Number of hidden units
NHiddenUnits = 15:20;
T=length(NHiddenUnits);
NTrain = 1; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);
best_units = nan(K,1);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));

    % Fit neural network to training set
    MSEBest = inf;
    for t = 1:T
        netwrk = nr_main(X_train, y_train, X_test, y_test, NHiddenUnits(t));
        if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); end
        best_units(k)=bestnet{k}.Nh;
    end
    
    % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error_train(k) = sum((y_train-y_train_est).^2);
    Error_test(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);  
    
%     [val,ind_opt]=min(sum(Error_test,2)/sum(CV2.TestSize));
%     best_units(k)=NHiddenUnits(ind_opt);
end



% Print the least squares errors
% Display results
fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));

test_error_real=sum(Error_test)/(sum(CV.TestSize))

% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold
displayNetworkRegression(bestnet{k});

% Display how network predicts (only for when there are two attributes)
if size(X_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkRegression(X_train, y_train, X_test, y_test, bestnet{k});
end


%% neural network (attempt to 2 level cross validation)

% predict length of abalone
if (first_time==1)
X(:, end) = [];
y=X(:,1);
X(:,1)=[];
[N, M] = size(X);
% X=[ones(size(X,1),1) X];
% M=M+1;
end
first_time=0;

attributeNames={'Offset', attributeNames{1:end}};

% K-fold crossvalidation
K = 10;
CV = cvpartition(N,'Kfold', K);

% Parameters for neural network classifier
%  NHiddenUnits = 1;  % Number of hidden units
NHiddenUnits = 1:5;
T=length(NHiddenUnits);
NTrain = 1; % Number of re-trains of neural network

% Variable for regression error
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);
Error_train2 = nan(T,K);
Error_test2 = nan(T,K);
best_units = nan(K,1);
for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    
    KK = 5;
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    for kk=1:KK
     
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
        % Fit neural network to training set
        MSEBest = inf;
        for t = 1:T
            netwrk = nr_main(X_train2, y_train2, X_test2, y_test2, NHiddenUnits(t));
            if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); end
        % Evaluate training and test performance
%         Error_train2(t,kk) = sum((y_train2-X_train2*ones(M,1)).^2);
%         Error_test2(t,kk) = sum((y_test2-X_test2*ones(M,1)).^2);
  
        % Predict model on test and training data
        y_train_est2 = bestnet{k}.t_pred_train;
        y_test_est2 = bestnet{k}.t_pred_test;
        
        % Compute least squares error
        Error_train2(t,kk) = sum((y_train2-y_train_est2).^2);
        Error_test2(t,kk) = sum((y_test2-y_test_est2).^2);
        end
    end
    [val,ind_opt]=min(sum(Error_test2,2)/sum(CV2.TestSize));
    best_units(k)=NHiddenUnits(ind_opt);
    
    if k == K
         % Plot error
        loglog(NHiddenUnits,[sum(Error_train2,2)/sum(CV2.TrainSize) sum(Error_test2,2)/sum(CV2.TestSize)],'.-');
        legend({'Training Error as function of h','Test Error as function of h'},'Location','SouthEast');
        title(['Optimal value of h: 1e' num2str(best_units(k))]);
        xlabel('\h');
        drawnow;
    end
    
    % Predict model on test and training data    
%     y_train_est = bestnet{k}.t_pred_train;    
%     y_test_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
%     Error_train(k) = sum((y_train-y_train_est).^2);
%     Error_test(k) = sum((y_test-y_test_est).^2); 
        
    % Compute least squares error when predicting output to be mean of
    % training data
    Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
    Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);            
end

% Print the least squares errors
% Display results
% fprintf('\n');
% fprintf('Neural network regression without feature selection:\n');
% fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
% fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
% fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
% fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));

test_error_real=sum(Error_test2)/sum(CV2.TestSize)

% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold
displayNetworkRegression(bestnet{k});

% Display how network predicts (only for when there are two attributes)
if size(X_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkRegression(X_train, y_train, X_test, y_test, bestnet{k});
end