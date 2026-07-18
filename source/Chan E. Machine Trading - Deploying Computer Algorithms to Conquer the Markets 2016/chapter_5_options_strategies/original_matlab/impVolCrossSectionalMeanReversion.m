% Same as impVolCrossSectionalMeanReversion3, but adjust contract number so
% that the equivalent stock market value is $1 for all stocks.

% Every day short topN PUTS with the highest impVol and long CALLS
% with lowest. Do not care about net delta.


% Use options file with all options id


clear;

topN=50;
minTime2Expire=28;
lastDay=20130830;

justRunEval=true;

stk=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'cusip', 'bid', 'ask', 'cl_unadj'); % isfinite(cl) will determine if stock is in SPX at time t

if (~justRunEval)
    
    lastDayIdx=find(stk.tday==lastDay);
    cusip8=char(stk.cusip');
    cusip8(:, end)=[];
    cusip8=cellstr(cusip8);
    
    posC=zeros(length(stk.tday), length(cusip8)); % number of call contracts
    posP=zeros(length(stk.tday), length(cusip8)); % number of put  contracts
    totIVC=NaN(size(posC));  % call  IV
    totIVP=NaN(size(posP));  % put  IV
    
    ATMCidx=NaN(size(posC)); % indices for strikes array that gives minimum CALL IV
    ATMPidx=NaN(size(posP)); % indices for strikes array that gives minimum PUT  IV
    
    
    for c=1:length(cusip8)
        opt=load(['C:/Projects/Ivy/SPX_200701_201308/cusip_', cusip8{c}], 'strikes', 'impVol', 'delta', 'optExpireDate', 'cpflag', 'bid', 'ask');
        
        for t=1:lastDayIdx
            
            optExpireDate_uniq=unique(opt.optExpireDate(t, :));
            optExpireDate_uniq(~isfinite(optExpireDate_uniq))=[];
            if (~isempty(optExpireDate_uniq))
                time2Expire=datenum(cellstr(num2str(optExpireDate_uniq')), 'yyyymmdd')-datenum(cellstr(num2str(stk.tday(t))), 'yyyymmdd'); % in calendar days
                
                optExpireDate1month=optExpireDate_uniq(time2Expire >= minTime2Expire); % Choose contract with at least 1 month to expiration
                if (~isempty(optExpireDate1month))
                    optExpireDate1month=optExpireDate1month(1); % Choose the nearest expiration
                    
                    minIV=Inf;
                    minIVIdx=NaN;
                    maxIV=-Inf;
                    maxIVIdx=NaN;
                    
                    strikes_uniq=unique(opt.strikes(t, :));
                    strikes_uniq(~isfinite(strikes_uniq))=[];
                    for s=1:length(strikes_uniq)
                        
                        idxC=find(opt.strikes(t, :)==strikes_uniq(s) & opt.optExpireDate(t, :)==optExpireDate1month  & ...
                            opt.bid(t, :)>0 & opt.ask(t, :)>0 & opt.bid(t+1, :)>0 & opt.ask(t+1, :)>0 & strcmp('C', opt.cpflag(t, :)));  % Ensure we can exit position day after entry!
                        if (length(idxC) > 1)
                            if (t > 1) % if duplicate, use old option data
                                idx2=find(~isfinite(opt.strikes(t-1, idxC)));
                                idxC(idx2)=[];
                                if (length(idxC) > 1)
                                    idxC=[];
                                end
                            else
                                idxC=[];
                            end
                        end
                        idxP=find(opt.strikes(t, :)==strikes_uniq(s) & opt.optExpireDate(t, :)==optExpireDate1month  & ...
                            opt.bid(t, :)>0 & opt.ask(t, :)>0 & opt.bid(t+1, :)>0 & opt.ask(t+1, :)>0 & strcmp('P', opt.cpflag(t, :)));  % Ensure we can exit position day after entry!
                        if (length(idxP) > 1)
                            if (t > 1) % if duplicate, use old option data
                                idx2=find(~isfinite(opt.strikes(t-1, idxP)));
                                idxP(idx2)=[];
                                if (length(idxP) > 1)
                                    idxP=[];
                                end
                            else
                                idxP=[];
                            end
                        end
                        
                        if (~isempty(idxC))
                            [myMinIVC, minIdx]=min(opt.impVol(t, idxC));
                            if (~isfinite(totIVC(t, c)) || myMinIVC < totIVC(t, c))
                                totIVC(t, c)=myMinIVC;
                                ATMCidx(t, c)=idxC(minIdx);
                            end
                        end
                        
                        if (~isempty(idxP))
                            [myMaxIVP, maxIdx]=max(opt.impVol(t, idxP));
                            if (~isfinite(totIVP(t, c)) || myMaxIVP > totIVP(t, c))
                                totIVP(t, c)=myMaxIVP;
                                ATMPidx(t, c)=idxP(maxIdx);
                            end
                        end
                        

                        
                    end
                    
                end
            end
        end
    end
    
    for t=1:lastDayIdx-1
        candidatesCIdx=find(isfinite(totIVC(t, :)) ); % Longs
        candidatesPIdx=find(isfinite(totIVP(t, :)) ); % Shorts
        
        if (length(candidatesCIdx) >= topN && length(candidatesPIdx) >= topN)
            [foo, minCIdx]=sort(totIVC(t, candidatesCIdx));
            [foo, maxPIdx]=sort(totIVP(t, candidatesPIdx), 'descend');
            
            posC(t, candidatesCIdx(minCIdx(1:topN)))=1;
            posP(t, candidatesPIdx(maxPIdx(1:topN)))=-1;
        end
        
    end
    
    save('C:/Projects/Options_data/impVolCrossSectionalMeanReversion_output');
    
else
    load('C:/Projects/Options_data/impVolCrossSectionalMeanReversion_output');
    stk=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'cusip', 'bid', 'ask', 'cl_unadj'); % isfinite(cl) will determine if stock is in SPX at time t

end





pnl=zeros(size(posC, 1), 1);
mktVal=zeros(size(pnl, 1), 1); % gross market value of all stock options

for c=1:size(posC, 2)
    load(['C:/Projects/Ivy/SPX_200701_201308/cusip_', cusip8{c}], 'bid', 'ask', 'delta', 'cfadj');
    bid=bid.*cfadj;
    ask=ask.*cfadj;

    mid=(bid+ask)/2;
    
 
    if (posC(1, c)~=0)    
        posC(1, c)=posC(1, c)/stk.cl_unadj(1, c);
    end
    if (posP(1, c)~=0)
        posP(1, c)=posP(1, c)/stk.cl_unadj(1, c);
    end
    
    if (~isfinite(posC(1, c)))
        posC(1, c)=0;
    end
    if (~isfinite(posP(1, c)))
        posP(1, c)=0;
    end
 
    
    for t=2:lastDayIdx
      
        if (posC(t, c)~=0)
            if (sign(posC(t, c))~=sign(posC(t-1, c)) && isfinite(stk.cl_unadj(t, c)))
                % Adjust position so that equivalent stock market value is +$1.
                posC(t, c)=posC(t, c)/stk.cl_unadj(t, c);
                assert(isfinite(posC(t, c)));
            else
                posC(t, c)=posC(t-1, c);
            end
        end
            
        if (posP(t, c)~=0)
            if (sign(posP(t, c))~=sign(posP(t-1, c)) && isfinite(stk.cl_unadj(t, c)))
                % Adjust position so that equivalent stock market value is +$1.
                posP(t, c)=posP(t, c)/stk.cl_unadj(t, c);
                assert(isfinite(posP(t, c)));
            else
                posP(t, c)=posP(t-1, c);
            end
        end
        
        if (posC(t-1, c) > 0)
            assert(all(isfinite([bid(t-1, ATMCidx(t-1, c)) ask(t-1, ATMCidx(t-1, c))])));
            
            if (~all(isfinite([bid(t, ATMCidx(t-1, c)) ask(t, ATMCidx(t-1, c))]))) % today's prices are missing!
                fprintf(1, 'Missing price for %s on %i\n', cusip8{c}, stk.tday(t));
                assert(all(any(isfinite([bid(t:end, ATMCidx(t-1, c)) ask(t:end, ATMCidx(t-1, c))]), 1))); % This must be some good prices in the future!
            else
                pnl(t)=pnl(t)+100*sum(posC(t-1, c)*(mid(t, ATMCidx(t-1, c))-mid(t-1, ATMCidx(t-1, c))));
                mktVal(t-1)=mktVal(t-1)+100*sum(abs(posC(t-1, c)*mid(t-1, ATMCidx(t-1, c, :))));

            end
        end
        
        if (posP(t-1, c) < 0)
            
            assert(all(isfinite([bid(t-1, ATMPidx(t-1, c)) ask(t-1, ATMPidx(t-1, c))])));
            
            if (~all(isfinite([bid(t, ATMPidx(t-1, c)) ask(t, ATMPidx(t-1, c))]))) % today's prices are missing!
                fprintf(1, 'Missing price for %s on %i\n', cusip8{c}, stk.tday(t));
                assert(all(any(isfinite(mid(t:end, ATMPidx(t-1, c))), 1))); % This must be some good prices in the future!
            else
                pnl(t)=pnl(t)+100*sum(posP(t-1, c)*(mid(t, ATMPidx(t-1, c))-mid(t-1, ATMPidx(t-1, c))));
                mktVal(t-1)=mktVal(t-1)+100*sum(abs(posP(t-1, c)*mid(t-1, ATMPidx(t-1, c, :))));

            end
        end
        
      

        
    end
end

pnl(lastDayIdx+1:end)=[];
mktVal(lastDayIdx+1:end)=[];

cumPL=smartcumsum(pnl);

dailyret=pnl./backshift(1, mktVal);
dailyret(~isfinite(dailyret))=0;
cumret=cumprod(1+dailyret)-1;

fprintf(1, 'Mean dailyret=%f\n', mean(dailyret));

cagr= prod(1+dailyret).^(252/length(dailyret))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret));

[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);

plot(datetime(stk.tday(1:lastDayIdx), 'ConvertFrom', 'yyyyMMdd'), cumret); % Cumulative P&L
title('Implied Volatility Cross-section Mean Reversion');
xlabel('Date');
ylabel('Cumulative Return');


% Mean dailyret=0.011611

% CAGR=15.872430 Sharpe=6.500396
% maxDD=-0.302968 maxDDD=96 Calmar ratio=52.389757

% plot(cumPL);
% 
% fprintf(1, 'APR/maxDD=%f \n', cumPL(end)/length(cumPL)*252/abs(calculateMaxDD_simple(cumPL)));


% APR/maxDD=7.946486 



