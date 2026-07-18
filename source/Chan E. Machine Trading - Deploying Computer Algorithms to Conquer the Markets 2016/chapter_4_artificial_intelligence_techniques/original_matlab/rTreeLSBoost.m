% rTreeLSBoost.m
% Use boosting to improve performance of rTree.
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

allM=[5, 10, 20, 40, 80, 100, 160, 320, 640, 1280, 2560, 5120]; % number of iterations

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build model on training data
trainset=1:floor(length(tday)/2);

% Select best regression tree model based on default criteria
% Syntax: ens = fitensemble(X,Y,model,numberens,learners)
% X is the matrix of data. Each row contains one observation, and each column contains one predictor variable.
% Y is the responses, with the same number of observations as rows in X.
% model is a string naming the type of ensemble.
% numberens is the number of weak learners in ens from each element of learners. The number of elements in ens is numberens times the number of elements in learners.
% learners is a string naming a weak learner, a weak learner template, or a cell array of such strings and templates.

cagrTrain=NaN(size(allM));
cagrTest=cagrTrain;
sharpeTrain=NaN(size(allM));
sharpeTest=cagrTrain;

for i=1:length(allM)
    M=allM(i);
    
    rng(1); % set fixed random seed for picking first learners, in order to get reproducible results
    model=fitensemble([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset), 'LSBoost', M, 'Tree');
    
    % Make "predictions" on training set (in-sample)
    
    retPred1=predict(model, [ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)]);
    
    % Backtest trading model based on "prediction" on training set
    positions=zeros(length(trainset), 1);
    
    positions(retPred1 > 0)=1;
    positions(retPred1 < 0)=-1;
    
    dailyRet=backshift(1, positions).*ret1(trainset);
    dailyRet(~isfinite(dailyRet))=0;
    
    cumret=cumprod(1+dailyRet)-1;
    
    plot(cumret);
    
    cagr=(1+cumret(end))^(252/length(cumret))-1;
    sharpe=sqrt(252)*mean(dailyRet)/std(dailyRet);
    [maxDD, maxDDD]=calculateMaxDD(cumret);
    fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sharpe, maxDD, maxDDD, -cagr/maxDD);
    % In-sample: CAGR=1.401662 Sharpe ratio=3.914854 maxDD=-0.081498 maxDDD=80 Calmar ratio=17.198762
    cagrTrain(i)=cagr;
    sharpeTrain(i)=sharpe;

    
    % Make real predictions on test (out-of-sample) set
    testset=floor(length(tday)/2)+1:length(tday);
    
    retPred1=predict(model, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);
    
    positions=zeros(length(testset), 1);
    
    positions(retPred1 > 0)=1;
    positions(retPred1 < 0)=-1;
    
    dailyRet=backshift(1, positions).*ret1(testset);
    dailyRet(~isfinite(dailyRet))=0;
    
    cumret=cumprod(1+dailyRet)-1;
    
    plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
    title('Regression tree on SPY with boosting. M=5.');
    xlabel('Date');
    ylabel('Cumulative Returns');
    
    cagr=(1+cumret(end))^(252/length(cumret))-1;
    sharpe=sqrt(252)*mean(dailyRet)/std(dailyRet);
    [maxDD, maxDDD]=calculateMaxDD(cumret);
    fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sharpe, maxDD, maxDDD, -cagr/maxDD);
    % Out-of-sample: CAGR=-0.031724 Sharpe ratio=-0.114672 maxDD=-0.369351 maxDDD=1170 Calmar ratio=-0.085891
    cagrTest(i)=cagr;
    sharpeTest(i)=sharpe;

    
end

plot(allM', [cagrTrain', cagrTest'], '*-');
title('Effect of boosting on train vs test set');
xlabel('Number of boosters');
ylabel('CAGR');
legend('Train set', 'Test set');

plot(allM', [sharpeTrain', sharpeTest'], '*-');
title('Effect of boosting on train vs test set');
xlabel('Number of boosters');
ylabel('Sharpe Ratio');
legend('Train set', 'Test set');

% Use cross validation
model_cv=fitensemble([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset), 'LSBoost', 5120, 'Tree', 'CrossVal', 'On', 'KFold', 5);

L= kfoldLoss(model_cv,'mode','individual'); % Find the loss (mean squared error) between the observations in a fold when compared against predictions made with a tree trained on the out-of-fold data.
[~, minLidx]=min(L); % pick the tree with the minimum loss, i.e. with least overfitting error.

bestTree=model_cv.Trained{minLidx};

retPred1=predict(bestTree, [ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)]);

positions=zeros(length(testset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Regression tree on SPY with boosting and cross validation. M=5120. K=10');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=-0.027547 Sharpe ratio=-0.088287 maxDD=-0.379288 maxDDD=845 Calmar ratio=-0.072628

