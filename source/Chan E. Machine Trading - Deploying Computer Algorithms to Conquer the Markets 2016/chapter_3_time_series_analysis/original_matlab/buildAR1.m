% Estimate parameters of an AR(1) model using Econometrics toolbox's estimate function 
clear;
% load('inputData_btcchina_secbars', 'ttime', 'cl');
load('Jonathan_BTCUSD_trades_daily', 'tday', 'cl'); % daily bars

model_ar1=arima(1, 0, 0) % assumes an AR(1) with unknown parameters

% model_ar1 = 
% 
%     ARIMA(1,0,0) Model:
%     --------------------
%     Distribution: Name = 'Gaussian'
%                P: 1
%                D: 0
%                Q: 0
%         Constant: NaN
%               AR: {NaN} at Lags [1]
%              SAR: {}
%               MA: {}
%              SMA: {}
%         Variance: NaN

model_ar1_estimates=estimate(model_ar1, cl); 

%     ARIMA(1,0,0) Model:
%     --------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant         3.4365       4.63118       0.742037
%         AR{1}       0.989484    0.00845394        117.044
%      Variance        405.023       15.2071        26.6337


load('Jonathan_BTCUSD_BBO_1minute', 'tday', 'HHMM', 'bid', 'ask');
mid=(bid+ask)/2;

model_ar1_estimates=estimate(model_ar1, mid); 

%    ARIMA(1,0,0) Model:
%     --------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant      0.0126353     0.0065861        1.91848
%         AR{1}       0.999972    1.0839e-05        92256.6
%      Variance       0.967225    9.8892e-05        9780.62


load('inputData_AUDUSD_20150807', 'tday', 'hhmm', 'mid');

% model_ar1_estimates=estimate(model_ar1, mid); 
% 
%    ARIMA(1,0,0) Model:
%     --------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant     1.4832e-06   8.63459e-07        1.71775
%         AR{1}       0.999998   9.82585e-07    1.01772e+06
%      Variance    5.03604e-08   3.37677e-10        149.138
