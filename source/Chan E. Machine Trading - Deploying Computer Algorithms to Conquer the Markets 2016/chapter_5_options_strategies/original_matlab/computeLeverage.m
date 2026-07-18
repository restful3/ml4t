% Compute leverage of one option (AAPL)
% Leverage = delta*mktValOfUnderlying/mktValOfOption

clear;
lastDay=20130830;
minTime2Expire=28;

stk=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'cusip', 'bid', 'ask', 'cl_unadj'); % isfinite(cl) will determine if stock is in SPX at time t

cusip_AAPL='03783310';

cusip8=char(stk.cusip');
cusip8(:, end)=[];
cusip8=cellstr(cusip8);

c=strmatch(cusip_AAPL, cusip8, 'exact'); % 175

mktVal_stk=stk.cl_unadj(:, c); % Stock's market value (1 share)
lastDayIdx=find(stk.tday==lastDay);

opt=load(['C:/Projects/Ivy/SPX_200701_201308/cusip_', cusip8{c}], 'strikes', 'impVol', 'delta', 'optExpireDate', 'cpflag', 'bid', 'ask');
        
t=lastDayIdx;

optExpireDate_uniq=unique(opt.optExpireDate(t, :));
optExpireDate_uniq(~isfinite(optExpireDate_uniq))=[];

if (~isempty(optExpireDate_uniq))
    time2Expire=datenum(cellstr(num2str(optExpireDate_uniq')), 'yyyymmdd')-datenum(cellstr(num2str(stk.tday(t))), 'yyyymmdd'); % in calendar days
    
    optExpireDate1month=optExpireDate_uniq(time2Expire >= minTime2Expire); % Choose contract with at least 1 month to expiration
    if (~isempty(optExpireDate1month))
        optExpireDate1month=optExpireDate1month(1); % Choose the nearest expiration
               
        strikes_uniq=unique(opt.strikes(t, :));
        strikes_uniq(~isfinite(strikes_uniq))=[];
    end
    
    idxC=find( opt.optExpireDate(t, :)==optExpireDate1month  & isfinite(opt.impVol(t, :)) & ...
        opt.bid(t, :)>0 & opt.ask(t, :)>0 & strcmp('C', opt.cpflag(t, :)));  

    idxP=find( opt.optExpireDate(t, :)==optExpireDate1month  & isfinite(opt.impVol(t, :)) & ...
        opt.bid(t, :)>0 & opt.ask(t, :)>0 & strcmp('P', opt.cpflag(t, :)));  

end

[~, idxSortC]=sort(opt.strikes(t, idxC));
[~, idxSortP]=sort(opt.strikes(t, idxP));

assert(all(opt.strikes(t, idxC(idxSortC(2:end))) > opt.strikes(t, idxC(idxSortC(1:end-1)))));
assert(all(opt.strikes(t, idxP(idxSortP(2:end))) > opt.strikes(t, idxP(idxSortP(1:end-1)))));

leverageC=opt.delta(t, idxC(idxSortC)).*mktVal_stk(t)./(opt.bid(t, idxC(idxSortC))+opt.ask(t, idxC(idxSortC)))*2;
leverageP=opt.delta(t, idxP(idxSortP)).*mktVal_stk(t)./(opt.bid(t, idxP(idxSortP))+opt.ask(t, idxP(idxSortP)))*2;
plot(opt.strikes(t, idxC(idxSortC))/1000, leverageC);
hold on;
plot(opt.strikes(t, idxP(idxSortP))/1000, leverageP, 'r');
title('Leverage of put and call AAPL options on 2013-08-30 that expire on 2013-09-27');
xlabel('Strike price');
ylabel('Leverage');
legend('Call', 'Put');

plot(mktVal_stk(t), -30, 'k*');

hold off;