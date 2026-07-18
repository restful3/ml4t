clear;

load('C:/Projects/prod_data/inputDataOHLCDaily_ETF_20151023', 'tday', 'stocks', 'cl');

cl=cl(:, strcmp(stocks, 'SPY'));

% ret=log(cl)-backshift(1, log(cl));

ret=calculateReturns(cl, 1);

cumret=cl./cl(1)-1;

% cumret=smartcumsum(ret);

% cl_es=es.cl(idx2);
% cumret_es=cl_es./cl_es(1)-1;
% 
% plot(cumret_mxn);
% 
% hold on;
% 
% plot(cumret_es, 'r');
% 
% hold off;

[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD: %f\n', maxDD);


% [maxDD, maxDDD]=calculateMaxDD(cumret_es);
% fprintf(1, 'maxDD ES: %f\n', maxDD);

% [ret_mxn_sort idx_mxn]=sort(ret_mxn);
[ret_min, idxMin]=min(ret);

fprintf(1, 'Worse daily returns=%f on %i\n', ret_min, tday(idxMin) );

% [num2cell(ret_mxn_sort(1:10)) num2cell(tday(idx_mxn(1:10)))]

[ret_max, idxMax]=max(ret);

fprintf(1, 'Best daily returns=%f on %i\n', ret_max, tday(idxMax));

% [num2cell(ret_mxn_sort(end:-1:end-9)) num2cell(tday(idx_mxn(end:-1:end-9)))]

% ret_es=calculateReturns(cl_es, 1);
% [ret_es_sort idx_es]=sort(ret_es);
% 
% fprintf(1, 'Worse daily returns for ES\n');
% 
% [num2cell(ret_es_sort(1:10)) num2cell(tday(idx_es(1:10)))]
% 
% fprintf(1, 'Best daily returns for ES\n');

% [num2cell(ret_es_sort(end:-1:end-9)) num2cell(tday(idx_es(end:-1:end-9)))]

fprintf(1, 'AnnAvgRet: %f\n', 252*smartmean(ret));

% fprintf(1, 'AnnAvgRet ES: %f\n', 252*smartmean(ret_es));

fprintf(1, 'Volatility: %f\n', sqrt(252)*smartstd(ret));
% 
% fprintf(1, 'Volatility ES: %f\n', sqrt(252)*smartstd(ret_es));

fprintf(1, 'Kurtosis: %f\n', kurtosis(ret));

fprintf(1, 'Period=%i to %i\n', tday(1), tday(end));
% fprintf(1, 'Kurtosis ES: %f\n', kurtosis(ret_es));

% maxDD: -0.551894
% Worse daily returns=-0.098480 on 20081015
% Best daily returns=0.145138 on 20081013
% AnnAvgRet: 0.093365
% Volatility: 0.206609
% Kurtosis: 17.778388
% Period=20051117 to 20151023
