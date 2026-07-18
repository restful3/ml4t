clear;

% Use spot price of 6M futures
load('C:/Projects/Futures_data/inputDataDaily_MP_20151026', 'tday', 'contracts', 'cl');

assert(strcmp(contracts(1), '0000$'));

cl=cl(:, 1);

cumret_mxn=cl./cl(1)-1;

[maxDD, maxDDD]=calculateMaxDD(cumret_mxn);
fprintf(1, 'maxDD MXN: %f\n', maxDD);

ret_mxn=calculateReturns(cl, 1);

[ret_min, idxMin]=min(ret_mxn);

fprintf(1, 'Worse daily returns for MXN=%f on %i\n', ret_min, tday(idxMin) );

[ret_max, idxMax]=max(ret_mxn);

fprintf(1, 'Best daily returns for MXN=%f on %i\n', ret_max, tday(idxMax));


fprintf(1, 'AnnAvgRet MXN: %f\n', 252*smartmean(ret_mxn));


fprintf(1, 'Volatility MXN: %f\n', sqrt(252)*smartstd(ret_mxn));


fprintf(1, 'Kurtosis MXN: %f\n', kurtosis(ret_mxn));



