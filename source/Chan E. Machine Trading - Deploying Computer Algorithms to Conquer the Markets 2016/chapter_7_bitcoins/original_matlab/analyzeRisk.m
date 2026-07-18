clear;
%% BTCUSD

load('Jonathan_BTCUSD_trades_daily',  'tday', 'cl');


idx=find(isfinite(cl));
tday=tday(idx);
cl=cl(idx);


cumret=cl./cl(1)-1;


[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD BTCUSD: %f\n', maxDD);

ret=calculateReturns(cl, 1);
[ret_min, idxMin]=min(ret);

fprintf(1, 'Worse daily returns for BTCUSD=%f on %i\n', ret_min, tday(idxMin) );

[ret_max, idxMax]=max(ret);

fprintf(1, 'Best daily returns for BTCUSD=%f on %i\n', ret_max, tday(idxMax));



fprintf(1, 'AnnAvgRet HYG: %f\n', 252*smartmean(ret));


fprintf(1, 'Volatility HYG: %f\n', sqrt(252)*smartstd(ret));

fprintf(1, 'Kurtosis HYG: %f\n', kurtosis(ret)*252/length(ret));

fprintf(1, 'Period analyzed=%i-%i\n', tday(1), tday(end));


%% MXNUSD
% usdmxn=load('C:/Projects/FX_data/inputData_USDMXN_20120730', 'tday', 'hhmm', 'op', 'hi', 'lo', 'cl');
% usdmxn=load('C:/Projects/FX_data/inputData_USDMXN_20150807', 'tday', 'hhmm', 'bid', 'ask', 'mid');
usdmxn=load('C:/Projects/FX_data/inputData_USDMXN_20160226', 'hhmm', 'tday', 'mid', 'bid', 'ask');

isClose=find(usdmxn.hhmm==1659);
tday=usdmxn.tday(isClose);

cl_mxn=1./usdmxn.mid(isClose);
cumret_mxn=cl_mxn./cl_mxn(1)-1;

[maxDD, maxDDD]=calculateMaxDD(cumret_mxn);
fprintf(1, 'maxDD MXN: %f\n', maxDD);

ret_mxn=calculateReturns(cl_mxn, 1);
[ret_min, idxMin]=min(ret_mxn);

fprintf(1, 'Worse daily returns for MXN=%f on %i\n', ret_min, tday(idxMin) );

[ret_max, idxMax]=max(ret_mxn);

fprintf(1, 'Best daily returns for MXN=%f on %i\n', ret_max, tday(idxMax));


fprintf(1, 'AnnAvgRet MXN: %f\n', 252*smartmean(ret_mxn));


fprintf(1, 'Volatility MXN: %f\n', sqrt(252)*smartstd(ret_mxn));
% 

fprintf(1, 'Kurtosis MXN: %f\n', kurtosis(ret_mxn)*252/length(ret_mxn));

fprintf(1, 'Period analyzed=%i-%i\n', tday(1), tday(end));

%% SPY

spy=load(['C:/Projects/prod_data/inputDataOHLCDaily_ETF_', num2str(20160520)], 'stocks', 'tday', 'op', 'hi', 'lo', 'cl');
tday=spy.tday;

cl=spy.cl(:, strcmp(spy.stocks, 'SPY'));
cumret=cl./cl(1)-1;

[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD SPY: %f\n', maxDD);

ret=calculateReturns(cl, 1);
[ret_min, idxMin]=min(ret);

fprintf(1, 'Worse daily returns for SPY=%f on %i\n', ret_min, tday(idxMin) );

[ret_max, idxMax]=max(ret);

fprintf(1, 'Best daily returns for SPY=%f on %i\n', ret_max, tday(idxMax));


fprintf(1, 'AnnAvgRet SPY: %f\n', 252*smartmean(ret));


fprintf(1, 'Volatility SPY: %f\n', sqrt(252)*smartstd(ret));

fprintf(1, 'Kurtosis SPY: %f\n', kurtosis(ret)*252/length(ret));

fprintf(1, 'Period analyzed=%i-%i\n', tday(1), tday(end));

%% HYG

spy=load(['C:/Projects/prod_data/inputDataOHLCDaily_ETF_', num2str(20160520)], 'stocks', 'tday', 'op', 'hi', 'lo', 'cl');
tday=spy.tday;

cl=spy.cl(:, strcmp(spy.stocks, 'HYG'));

idx=find(isfinite(cl));
tday=tday(idx);
cl=cl(idx);


cumret=cl./cl(1)-1;


[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD HYG: %f\n', maxDD);

ret=calculateReturns(cl, 1);
[ret_min, idxMin]=min(ret);

fprintf(1, 'Worse daily returns for HYG=%f on %i\n', ret_min, tday(idxMin) );

[ret_max, idxMax]=max(ret);

fprintf(1, 'Best daily returns for HYG=%f on %i\n', ret_max, tday(idxMax));


fprintf(1, 'AnnAvgRet HYG: %f\n', 252*smartmean(ret));


fprintf(1, 'Volatility HYG: %f\n', sqrt(252)*smartstd(ret));

fprintf(1, 'Kurtosis HYG: %f\n', kurtosis(ret)*252/length(ret));

fprintf(1, 'Period analyzed=%i-%i\n', tday(1), tday(end));

