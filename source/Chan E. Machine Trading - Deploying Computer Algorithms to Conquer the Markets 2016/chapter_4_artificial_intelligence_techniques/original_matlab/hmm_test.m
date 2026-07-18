% hmm_train.m Parameter estimation for HMM using BNT picking from 10 starting points using max
% likelihood
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
% ret2=calculateReturns(cl, 2);
% ret5=calculateReturns(cl, 5);
% ret20=calculateReturns(cl, 20);


% retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable.
data=double(ret1 >= 0)+1; % 1 represents ret1 < 0, 2 represents ret1 >= 0.

% prior =
% 
%    0.001229270954050
%    0.998770729045950
% 
% 
% transmat =
% 
%    0.595799945653891   0.404200054346109
%    0.748746357609356   0.251253642390644
% 
% 
% obsmat =
% 
%    0.190513239994937   0.809486760005063
%    0.973037767548789   0.026962232451211
% 

trainset=1:floor(length(tday)/2);

T=[0.595799945653891   0.404200054346109; 
    0.748746357609356   0.251253642390644];

E=[0.190513239994937   0.809486760005063;
    0.973037767548789   0.026962232451211];

pemis=NaN(2, size(data, 1));
for t=1:size(data, 1)-1
    [pstates]=hmmdecode(data(1:t)', T, E);
    pemis(:, t+1)=E'*T'*pstates(:, end);
end


% pemis=E'*pstates;
length(find(pemis(1, :) > pemis(2, :)))/length(pemis)
% ans =
% 
%               0.549002969876962

% Make "predictions" on training set (in-sample)
% Buy long if predicted value is True, otherwise short
isRetPositiveOrZero=pemis(1, trainset) <= pemis(2, trainset);

positions=zeros(length(trainset), 1);

positions(isRetPositiveOrZero)=1;
positions(~isRetPositiveOrZero)=-1;

dailyRet=backshift(1, positions).*ret1(trainset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(trainset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('HMM on SPY: train set');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=0.086644 Sharpe ratio=0.467205 maxDD=-0.310919 maxDDD=389 Calmar ratio=0.278670

% Test set
testset=floor(length(tday)/2)+1:length(tday);

isRetPositiveOrZero=pemis(1, testset) <= pemis(2, testset);

positions=zeros(length(testset), 1);

positions(isRetPositiveOrZero)=1;
positions(~isRetPositiveOrZero)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('HMM on SPY: test set');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=-0.010173 Sharpe ratio=0.019724 maxDD=-0.289298 maxDDD=723 Calmar ratio=-0.035165

% Finding the most probable hidden state sequence
states=hmmviterbi(data, T, E);

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), states);
title('Most probable state sequence');
xlabel('Date');
ylabel('State');