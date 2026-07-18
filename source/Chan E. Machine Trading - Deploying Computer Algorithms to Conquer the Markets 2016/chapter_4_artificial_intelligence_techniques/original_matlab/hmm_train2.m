% hmm_train2.m Parameter estimation for HMM using Statistics Toolbox picking from 10 starting points using max
% likelihood
assert(0, 'Does not work because hmmtrain always return zeros in second row of transmat');

clear;
numTrials=100; % number of starting points

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
% ret2=calculateReturns(cl, 2);
% ret5=calculateReturns(cl, 5);
% ret20=calculateReturns(cl, 20);

Q=2; % 2 hidden states
O=2; % 2 observables

% retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable.
data=double(ret1 >= 0)+1; % 1 represents ret1 < 0, 2 represents ret1 >= 0.

% Build model on training data
trainset=1:floor(length(tday)/2);

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

bestLoglik=-Inf;

% Find best error minima from different initial guesses
for trial=1:numTrials
    
    % Random initial guesses
    %     prior1 = normalise(rand(Q,1)); % Make the entries of a (multidimensional) array sum to 1
    transmat1 = mk_stochastic(rand(Q,Q)); % Ensure the argument is a stochastic matrix, i.e., the sum over the last dimension is 1.
    obsmat1 = mk_stochastic(rand(Q,O));

    [estTR,estE] = hmmtrain(data(trainset),transmat1,obsmat1,'verbose',true);

   
    %     fprintf(1, 'trial=%i loglik=%f5.4\n', trial, loglik);
    
    %     if (loglik > bestLoglik)
    %         bestPrior=prior2;
    %         bestTransmat=transmat2;
    %         bestObsmat=obsmat2;
    %     end
    
end


