% Find best AR(p) model (i.e. best p) 
clear;
load('inputData_AUDUSD_20150807', 'tday', 'hhmm', 'mid');

idx=find(isfinite(mid));
tday(1:idx-1)=[];
hhmm(1:idx-1)=[];
mid(1:idx-1)=[];

mid=log(mid); % need log prices for ARIMA

LOGL=-Inf(10, 20); % log likelihood for up to 10 p and 20 q (20 minutes)
PQ=zeros(size(LOGL)); % p and q values

trainset=1:(length(mid)-252*(24*60-15)); % Use all but 1 year of bars for in-sample fitting

for p=1:size(PQ, 1)
    for q=1:size(PQ, 2)
        
        
        model=arima(p, 1, q); % Fix r in ARIMA(p, r, q)
        try
            [~,~,logL] = estimate(model, mid(trainset),'print',false);
            LOGL(p, q) = logL;
            PQ(p, q) = p+q;
        catch
        end
    end
end

% Has p+q+1 parameters, including constant
LOGL_vector = reshape(LOGL, size(LOGL, 1)*size(LOGL, 2), 1);
PQ_vector = reshape(PQ, size(LOGL, 1)*size(LOGL, 2), 1);
[~, bic]=aicbic(LOGL_vector, PQ_vector+1, length(mid(trainset)));
[bicMin, pMin]=min(bic)
%

bic(:)=NaN;
bic(pMin)=bicMin;
bic=reshape(bic,size(LOGL))
%
% bicMin =
% 
%     -3.530843733576701e+07
% 
% 
% pMin =
% 
%     81
% 
% 
% bic =
% 
%    1.0e+07 *
% 
%   Columns 1 through 4
% 
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
% 
%   Columns 5 through 8
% 
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
% 
%   Columns 9 through 12
% 
%   -3.530843733576701                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
% 
%   Columns 13 through 16
% 
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
% 
%   Columns 17 through 20
% 
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
%                  NaN                 NaN                 NaN                 NaN
