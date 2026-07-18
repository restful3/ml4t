% hmm_train.m Parameter estimation for HMM using BNT picking from 10 starting points using max
% likelihood
clear;
numTrials=10; % number of starting points

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
% ret2=calculateReturns(cl, 2);
% ret5=calculateReturns(cl, 5);
% ret20=calculateReturns(cl, 20);


% retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable.
data=double(ret1 >= 0)+1; % 1 represents ret1 < 0, 2 represents ret1 >= 0.

ss = 2; %slice size(ss)

intra = zeros(ss);
intra(1,2) = 1; % node 1 in slice t connects to node 2 in slice t

inter = zeros(ss);
inter(1,1) = 1; % node 1 in slice t-1 connects to node 1 in slice t

Q=2; % 2 hidden states
O=2; % 2 observables

ns = [Q O]; % Node sizes
dnodes = 1:2;
onodes = [2]; 

% eclass1 = [1 2];
% eclass2 = [3 2];

% bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'observed', onodes);

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
for trial=1:numTrials
    
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
        bestLoglik=loglik;
        fprintf(1, ' ***bestLoglik=%f5.4\n', bestLoglik);

    end
    
end

[prior, transmat, obsmat] = dbn_to_hmm(bestNet);

prior
transmat
obsmat=obsmat{1}.CPT

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


