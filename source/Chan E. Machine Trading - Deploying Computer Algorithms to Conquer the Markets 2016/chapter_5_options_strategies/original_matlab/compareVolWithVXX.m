% How often does increase/decrease in "volatility" coincide with
% increase/decrease in VXX?
clear;
load('inputDataOHLCDaily_ETF_20151125', 'tday', 'stocks', 'cl');
% load('C:/Projects/prod_data/inputDataOHLCDaily_ETF_20160311', 'stocks', 'tday',  'cl');

idxS=find(strcmp('SPY', stocks));
idxV=find(strcmp('VXX', stocks));
vxx=cl(:, idxV);
spy=cl(:, idxS);

ret=[NaN; price2ret(spy)]; % log returns

deltaV=ret.^2-backshift(1, ret.^2); %  change in realized volatility
deltaVXX=vxx-backshift(1, vxx); %      change in implied volatility

fprintf(1, 'Percent of days with same direction=%f\n', sum(sign(deltaV)==sign(deltaVXX))/length(find(isfinite(deltaV))));
fprintf(1, 'Percent of days with positive returns with same direction=%f\n', sum(sign(deltaV(ret > 0))==sign(deltaVXX(ret > 0)))/length(deltaV(ret > 0)));
fprintf(1, 'Percent of days with negative returns with same direction=%f\n', sum(sign(deltaV(ret < 0))==sign(deltaVXX(ret < 0)))/length(deltaV(ret < 0)));

% tday([1 end])
% 
% ans =
% 
%     20051221
%     20151125
% Percent of days with same direction=0.350681
% Percent of days with positive returns with same direction=0.288420
% Percent of days with negative returns with same direction=0.426273

% tday([1 end])
% 
% ans =
% 
%     20020418
%     20160311
% Percent of days with same direction=0.261864
% Percent of days with positive returns with same direction=0.215778
% Percent of days with negative returns with same direction=0.318471