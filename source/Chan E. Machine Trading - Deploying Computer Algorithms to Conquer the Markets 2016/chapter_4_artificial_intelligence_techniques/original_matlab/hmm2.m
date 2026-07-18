% hmm2.m using BNT
% Classification using HMM
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
% ret2=calculateReturns(cl, 2);
% ret5=calculateReturns(cl, 5);
% ret20=calculateReturns(cl, 20);


% retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable.
data=double(ret1 >= 0)+1; % 1 represents ret1 < 0, 2 represents ret1 >= 0.

ss = 2;%slice size(ss)

intra = zeros(ss);
intra(1,2) = 1; % node 1 in slice t connects to node 2 in slice t

inter = zeros(ss);
inter(1,1) = 1; % node 1 in slice t-1 connects to node 1 in slice t

Q=2; % 2 hidden states
O=2; % 2 observables

ns = [Q O]; % Node sizes
dnodes = 1:2;
onodes = [2]; 

eclass1 = [1 2];
eclass2 = [3 2];

bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
% for i=1:3
%   bnet.CPD{i} = tabular_CPD(bnet, i);
% end

% Build model on training data
trainset=1:floor(length(tday)/2);
ncases = length(trainset);%number of examples
T=ncases;
cases = cell(1, ncases);
for i=1:ncases
%   ev = sample_dbn(bnet, T);
  cases{i} = cell(ss,T);
  cases{i}(onodes,:) = num2cell(data(trainset)');
end

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

bestLoglik=-Inf;

% Find best error minima from different initial guesses
for trial=1:100
    
    % Random initial guesses
    prior1 = normalise(rand(Q,1));
    transmat1 = mk_stochastic(rand(Q,Q));
    obsmat1 = mk_stochastic(rand(Q,O));
    bnet.CPD{1} = tabular_CPD(bnet, 1, prior1);
    bnet.CPD{2} = tabular_CPD(bnet, 2, obsmat1);
    bnet.CPD{3} = tabular_CPD(bnet, 3, transmat1);
    
    engine = smoother_engine(hmm_2TBN_inf_engine(bnet));
    
    %     [~, prior2, transmat2, obsmat2] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', 1000, 'verbose', 1);
    %     loglik = dhmm_logprob(data, prior2, transmat2, obsmat2);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', 1000);
    loglik=LLtrace(end);
    
    fprintf(1, 'trial=%i loglik=%f5.4\n', trial, loglik);
    
    if (loglik > bestLoglik)
        bestNet=bnet2;
    end
    
end

[prior, transmat, obsmat] = dbn_to_hmm(bestNet);
obsmat=obsmat{1}.CPT;

% Make predictions on test set 
% Buy long if predicted value is True, otherwise short
testset=floor(length(tday)/2)+1:length(tday);

isRetPositiveOrZero=predict(bestTree, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);

positions=zeros(length(testset), 1);

positions(isRetPositiveOrZero)=1;
positions(~isRetPositiveOrZero)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated classification tree on SPY');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% Out-of-sample: CAGR=0.047740 Sharpe ratio=0.366381 maxDD=-0.272897 maxDDD=457 Calmar ratio=0.174938

% Try threshold for long or short only model, trade only when large positive or negative returns
threshold=0.001;

model_long=fitctree([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset) >= threshold, 'MinLeafSize', 100, 'CrossVal', 'On', 'KFold', 5); % Response: True if >=0, False if < 0.
L= kfoldLoss(model_long,'mode','individual'); % Find the loss (mean squared error) between the predicted responses and true responses in a fold when compared against predictions made with a tree trained on the out-of-fold data.
[~, minLidx]=min(L); % pick the tree with the minimum loss, i.e. with least overfitting error.

bestTree_long=model_long.Trained{minLidx};

model_short=fitctree([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset) <= -threshold, 'MinLeafSize', 100, 'CrossVal', 'On', 'KFold', 5); % Response: True if >=0, False if < 0.
L= kfoldLoss(model_short,'mode','individual'); % Find the loss (mean squared error) between the predicted responses and true responses in a fold when compared against predictions made with a tree trained on the out-of-fold data.
[~, minLidx]=min(L); % pick the tree with the minimum loss, i.e. with least overfitting error.

bestTree_short=model_short.Trained{minLidx};

% Make predictions on test set 
% Buy long if predicted value by model_long is True, short if predicted
% value by model_short is True. 
testset=floor(length(tday)/2)+1:length(tday);

toBuy=predict(bestTree_long, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);
toShort=predict(bestTree_short, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);

% Long only model
positions=zeros(length(testset), 1);

positions(toBuy)=1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated classification tree on SPY: long-only');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample long-only: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

positions=ones(length(testset), 1); % Buy and hold
dailyRet_Buyandhold=backshift(1, positions).*ret1(testset);
dailyRet_Buyandhold(~isfinite(dailyRet_Buyandhold))=0;

cumret=cumprod(1+dailyRet_Buyandhold)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated classification tree on SPY with 3 classes');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample buy-and-hold: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet_Buyandhold)/std(dailyRet_Buyandhold), maxDD, maxDDD, -cagr/maxDD);
fprintf(1, 'Out-of-sample long-only: IR=%f\n',  sqrt(252)*mean(dailyRet-dailyRet_Buyandhold)/std(dailyRet-dailyRet_Buyandhold));


% Short only model
positions=zeros(length(testset), 1);

positions(toShort)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated classification tree on SPY: short-only');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample long-only: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

positions=-ones(length(testset), 1); % Short and hold
dailyRet_Shortandhold=backshift(1, positions).*ret1(testset);
dailyRet_Shortandhold(~isfinite(dailyRet_Shortandhold))=0;

cumret=cumprod(1+dailyRet_Shortandhold)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated classification tree on SPY with 3 classes');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample buy-and-hold: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet_Shortandhold)/std(dailyRet_Shortandhold), maxDD, maxDDD, -cagr/maxDD);
fprintf(1, 'Out-of-sample long-only: IR=%f\n',  sqrt(252)*mean(dailyRet-dailyRet_Shortandhold)/std(dailyRet-dailyRet_Shortandhold));