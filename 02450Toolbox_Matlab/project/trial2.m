load ionosphere;
rng(151515); % set the RNG seed for reproducibility
cvp = cvpartition(Y,'holdout',0.5);
itrain = training(cvp);
itest = test(cvp);
svm = fitcsvm(X(itrain,:),Y(itrain),'Standardize',true,...
    'KernelFunction','RBF','KernelScale','auto');
YhatSVM = predict(svm,X(itest,:));
bag = fitensemble(X(itrain,:),Y(itrain),'Bag',100,'Tree','type','classification');
YhatBag = predict(bag,X(itest,:));
[h,p] = testcholdout(YhatSVM,YhatBag,Y(itest))