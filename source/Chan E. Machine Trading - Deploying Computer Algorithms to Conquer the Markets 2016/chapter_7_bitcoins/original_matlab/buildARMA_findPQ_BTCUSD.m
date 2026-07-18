% Find best AR(p) model (i.e. best p) 
clear;
load('Jonathan_BTCUSD_BBO_1minute', 'tday', 'HHMM', 'bid', 'ask');

mid=(bid+ask)/2;

idx=find(isfinite(mid));
tday(1:idx-1)=[];
HHMM(1:idx-1)=[];
mid(1:idx-1)=[];

LOGL=-Inf(10, 9); % log likelihood for up to 10 p and 9 q (10 minutes)
PQ=zeros(size(LOGL)); % p and q values

trainset=1:(length(mid)-126*24*60); % Use all but 0.5 year of bars for in-sample fitting

for p=1:size(PQ, 1)
    for q=1:size(PQ, 2)
        
        
        model=arima(p, 0, q);
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
bic=reshape(bic,size(LOGL));

% These are the best (P, Q)
P=find(any(isfinite(bic), 2))
Q=find(isfinite(bic(P, :)))
% 
% 
% P =
% 
%      3
% 
% 
% Q =
% 
%      7
