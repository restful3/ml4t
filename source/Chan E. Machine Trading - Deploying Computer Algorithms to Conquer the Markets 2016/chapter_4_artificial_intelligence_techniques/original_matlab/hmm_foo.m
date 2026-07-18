clear;

trans = [0.95,0.05;
      0.10,0.90];
emis = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;
   1/10, 1/10, 1/10, 1/10, 1/10, 1/2];

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

seqTrain = hmmgenerate(100,trans,emis);
seqTest = hmmgenerate(100,trans,emis);

% original guess
transGuess = trans;
emisGuess = emis;

[estTR,estE] = hmmtrain(seq,transGuess,emisGuess);

% estTR =
% 
%     0.9483    0.0517
%     0.1866    0.8134

% estE =
% 
%     0.1897    0.2111    0.2039    0.1916    0.0960    0.1077
%     0.0068    0.1139    0.0019    0.0000    0.2525    0.6250

% initial guess 1
transGuess = [0.5,0.5;
      0.5,0.5];
emisGuess = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;
   1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

[estTR,estE] = hmmtrain(seq,transGuess,emisGuess);

% estTR =
% 
%     0.5000    0.5000
%     0.5000    0.5000
% 
% estE =
% 
%     0.1600    0.2000    0.1300    0.1600    0.1300    0.2200
%     0.1600    0.2000    0.1300    0.1600    0.1300    0.2200


% initial guess 2
transGuess = [0.2,0.8;
      0.9,0.1];
emisGuess = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6;
   1/6, 1/6, 1/6, 1/6, 1/6, 1/6];

[estTR,estE] = hmmtrain(seq,transGuess,emisGuess)

% estTR =
% 
%     0.1165    0.8835
%     1.0000    0.0000
% 
% 
% estE =
% 
%     0.0525    0.2429    0.0424    0.1687    0.1032    0.3904
%     0.3435    0.0462    0.1855    0.1503    0.1388    0.1356
