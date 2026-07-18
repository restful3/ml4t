clear;

% usdmxn=load('C:/Projects/FX_data/inputData_USDMXN_20120730', 'tday', 'hhmm', 'op', 'hi', 'lo', 'cl');
usdmxn=load('C:/Projects/FX_data/inputData_USDMXN_20150807', 'tday', 'hhmm', 'bid', 'ask', 'mid');

isClose=find(usdmxn.hhmm==1659);
tday_mxn=usdmxn.tday(isClose);

cl_mxn=1./usdmxn.mid(isClose);
cumret_mxn=cl_mxn./cl_mxn(1)-1;

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

[maxDD, maxDDD]=calculateMaxDD(cumret_mxn);
fprintf(1, 'maxDD MXN: %f\n', maxDD);


% [maxDD, maxDDD]=calculateMaxDD(cumret_es);
% fprintf(1, 'maxDD ES: %f\n', maxDD);

ret_mxn=calculateReturns(cl_mxn, 1);
% [ret_mxn_sort idx_mxn]=sort(ret_mxn);
[ret_min, idxMin]=min(ret_mxn);

fprintf(1, 'Worse daily returns for MXN=%f on %i\n', ret_min, tday_mxn(idxMin) );

% [num2cell(ret_mxn_sort(1:10)) num2cell(tday(idx_mxn(1:10)))]

[ret_max, idxMax]=max(ret_mxn);

fprintf(1, 'Best daily returns for MXN=%f on %i\n', ret_max, tday_mxn(idxMax));

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

fprintf(1, 'AnnAvgRet MXN: %f\n', 252*smartmean(ret_mxn));

% fprintf(1, 'AnnAvgRet ES: %f\n', 252*smartmean(ret_es));

fprintf(1, 'Volatility MXN: %f\n', sqrt(252)*smartstd(ret_mxn));
% 
% fprintf(1, 'Volatility ES: %f\n', sqrt(252)*smartstd(ret_es));

fprintf(1, 'Kurtosis MXN: %f\n', kurtosis(ret_mxn));

% fprintf(1, 'Kurtosis ES: %f\n', kurtosis(ret_es));


if (0)

% VIX Index
[num txt]=xlsread('VIX.csv');
VIX=num(:, end);

tday_VIX=str2double(cellstr(datestr(datenum(txt(2:end, 1), 'yyyy-mm-dd'), 'yyyymmdd')));

[mytday idx1 idx2]=intersect(tday, tday_VIX);
ret_mxn=ret_mxn(idx1);
ret_es=ret_es(idx1);
VIX=VIX(idx2);

prevVIX=backshift(1, VIX);

if (0)
    prevVIX(1)=[];
    ret_mxn(1)=[];
    ret_es(1)=[];

    fprintf(1, 'Linear corr prevVIX-MXNret: %f\n', corr(prevVIX, ret_mxn));
    fprintf(1, 'Rank corr prevVIX-MXNret: %f\n', corr(prevVIX, ret_mxn, 'type', 'Spearman'));
end

if (1)
    ret_mxn(prevVIX >= 35)=0;
    [ret_mxn_sort idx_mxn]=sort(ret_mxn);
end

disp('*** With VIX>=35 filter ***');
fprintf(1, 'Worse daily returns for MXN\n');
[num2cell(ret_mxn_sort(1:10)) num2cell(tday(idx_mxn(1:10)))]

fprintf(1, 'AnnAvgRet MXN: %f\n', 252*smartmean(ret_mxn));
fprintf(1, 'Volatility MXN: %f\n', sqrt(252)*smartstd(ret_mxn));
fprintf(1, 'Kurtosis MXN: %f\n', kurtosis(ret_mxn));

ret_es(prevVIX >= 35)=0;
[ret_es_sort idx_es]=sort(ret_es);

fprintf(1, 'Worse daily returns for ES\n');
[num2cell(ret_es_sort(1:10)) num2cell(tday(idx_es(1:10)))]

fprintf(1, 'AnnAvgRet ES: %f\n', 252*smartmean(ret_es));
fprintf(1, 'Volatility ES: %f\n', sqrt(252)*smartstd(ret_es));
fprintf(1, 'Kurtosis ES: %f\n', kurtosis(ret_es));


if (0)
disp('*** With deltaVIX>=3 filter ***');
deltaVIX=VIX-prevVIX;

ret_mxn(deltaVIX >= 3)=0;
[ret_mxn_sort idx_mxn]=sort(ret_mxn);
end

fprintf(1, 'Worse daily returns for MXN\n');
[num2cell(ret_mxn_sort(1:10)) num2cell(tday(idx_mxn(1:10)))]
fprintf(1, 'Best daily returns for MXN\n');

[num2cell(ret_mxn_sort(end:-1:end-9)) num2cell(tday(idx_mxn(end:-1:end-9)))]
fprintf(1, 'AnnAvgRet MXN: %f\n', 252*smartmean(ret_mxn));
fprintf(1, 'Volatility MXN: %f\n', sqrt(252)*smartstd(ret_mxn));
fprintf(1, 'Kurtosis MXN: %f\n', kurtosis(ret_mxn));

ret_es(prevVIX >= 35)=0;
[ret_es_sort idx_es]=sort(ret_es);

fprintf(1, 'Worse daily returns for ES\n');
[num2cell(ret_es_sort(1:10)) num2cell(tday(idx_es(1:10)))]

fprintf(1, 'Best daily returns for ES\n');

[num2cell(ret_es_sort(end:-1:end-9)) num2cell(tday(idx_es(end:-1:end-9)))]

fprintf(1, 'AnnAvgRet ES: %f\n', 252*smartmean(ret_es));
fprintf(1, 'Volatility ES: %f\n', sqrt(252)*smartstd(ret_es));
fprintf(1, 'Kurtosis ES: %f\n', kurtosis(ret_es));



ret_mxn(1)=0;
cumret_mxn=cumprod(ret_mxn+1)-1;
[maxDD, maxDDD]=calculateMaxDD(cumret_mxn);
fprintf(1, 'maxDD MXN: %f\n', maxDD);
ret_es(1)=0;

cumret_es=cumprod(ret_es+1)-1;


[maxDD, maxDDD]=calculateMaxDD(cumret_es);
fprintf(1, 'maxDD ES: %f\n', maxDD);

end