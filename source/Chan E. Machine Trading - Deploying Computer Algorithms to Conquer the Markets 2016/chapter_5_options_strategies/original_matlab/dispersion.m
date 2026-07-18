% Update options every day: do not hold existing positions
% Allow as few as 1 stock straddle if cannot find topN that fit criteria.

% Always use midpoints

% Use options file with all options id and no survivorship-bias

% Select topN1 ATM puts and calls with least negative (highest) theta
% period, and buy the ATM straddle (with abs(netDelta) <= 0.25). Form portfolio equal-weighted by
% their underlying's market value, and sell SPX straddles with same total vega.

clear;

topN=50;
maxNetDelta=0.25;
minTime2Expire=28;
lastDay=20130830;

justRunEval=true;

stk=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'cusip', 'cl_unadj'); % isfinite(cl) will determine if stock is in SPX at time t
lastDayIdx=find(stk.tday==lastDay);
cusip8=char(stk.cusip');
cusip8(:, end)=[];
cusip8=cellstr(cusip8);

pos=zeros(length(stk.tday), length(cusip8)+1); % number of call or put stock contracts. The last column is for SPX index option.
totTheta=NaN(size(pos));  % call + put theta
totVega=NaN(size(pos));  % call + put vega
ATMidx=NaN([size(pos), 2]); % indices for strikes array that gives minimum IV with small delta constraint, put and call (in no fixed order)

if (~justRunEval) 
    
    for c=1:length(cusip8)+1 % Go through each symbol. The +1 is for index option
        if (c <= length(cusip8)) % if this is a stock option
            opt=load(['C:/Projects/Ivy/SPX_200701_201308/cusip_', cusip8{c}], 'strikes', 'theta', 'vega', 'delta', 'optExpireDate', 'cpflag', 'bid', 'ask'); % Each stock's options data stored in a separate file.
        end
        
        
        for tt=1:lastDayIdx-1 % Go through each trading day. tt refers to time index in output array. 
            
            if (c==length(cusip8)+1) % if this is an index option
                opt=load(['C:/Projects/Ivy/SPX_index_200701_201308/', num2str(stk.tday(tt))]); % Each day of the SPX options stored in a separate file.
                t=1; % Since we handle index options one day at a time, the data arrays has dimension [1 m] where m= number of strike prices. t=1 refers to the first index in input arrays. But output arrays all have dimension [T, N+1], so we need index tt.
            else
                t=tt;
            end
            
            %             if (strcmp(cusip8{c}, '00282410') && stk.tday(tt)==20070720)
            %                 keyboard; % DEBUG ONLY!!!
            %             end
           
            
            optExpireDate_uniq=unique(opt.optExpireDate(t, :));
            optExpireDate_uniq(~isfinite(optExpireDate_uniq))=[];
            
            if (~isempty(optExpireDate_uniq))
                time2Expire=datenum(cellstr(num2str(optExpireDate_uniq')), 'yyyymmdd')-datenum(cellstr(num2str(stk.tday(tt))), 'yyyymmdd'); % in calendar days
                
                optExpireDate1month=optExpireDate_uniq(time2Expire >= minTime2Expire); % Choose contract with at least 1 month to expiration
                
                if (~isempty(optExpireDate1month))
                    optExpireDate1month=optExpireDate1month(1); % Choose the nearest expiration
                    
                    minAbsDelta=Inf;
                    minAbsDeltaIdx=NaN;
                    strikes_uniq=unique(opt.strikes(t, :));
                    strikes_uniq(~isfinite(strikes_uniq))=[];
                    
                    for s=1:length(strikes_uniq) % Go through each strike
                        
                        if (c<=length(cusip8))
                            idxC=find(opt.strikes(t, :)==strikes_uniq(s) & opt.optExpireDate(t, :)==optExpireDate1month & isfinite(opt.delta(t, :)) & ...
                                opt.bid(t, :)>0 & opt.ask(t, :)>0 & strcmp('C', opt.cpflag(t, :)) & opt.bid(t+1, :)>0 & opt.ask(t+1, :)>0);  % Ensure we can exit position day after entry!
                        else
                            idxC=find(opt.strikes(t, :)==strikes_uniq(s) & opt.optExpireDate(t, :)==optExpireDate1month & isfinite(opt.delta(t, :)) & ...
                                opt.bid(t, :)>0 & opt.ask(t, :)>0 & strcmp('C', opt.cpflag(t, :)));
                        end
                        
                        if (length(idxC) > 1)
                            if (t > 1) % if duplicate, use old option data
                                assert(c~=length(cusip8)+1, 'We cannot deal with SPX for duplicates yet!');
                                idx2=find(~isfinite(opt.strikes(t-1, idxC)));
                                idxC(idx2)=[];
                                if (length(idxC) > 1)
                                    idxC=[];
                                end
                            else
                                idxC=[];
                            end
                        end
                        
                        if (c<=length(cusip8)) % stock options
                            idxP=find(opt.strikes(t, :)==strikes_uniq(s) & opt.optExpireDate(t, :)==optExpireDate1month & isfinite(opt.delta(t, :)) & ...
                                opt.bid(t, :)>0 & opt.ask(t, :)>0 & strcmp('P', opt.cpflag(t, :)) & opt.bid(t+1, :)>0 & opt.ask(t+1, :)>0);  % Ensure we can exit position day after entry!
                        else % index options
                            idxP=find(opt.strikes(t, :)==strikes_uniq(s) & opt.optExpireDate(t, :)==optExpireDate1month & isfinite(opt.delta(t, :)) & ...
                                opt.bid(t, :)>0 & opt.ask(t, :)>0 & strcmp('P', opt.cpflag(t, :)));
                        end
                        
                        if (length(idxP) > 1)
                            if (t > 1) % if duplicate, use old option data
                                assert(c~=length(cusip8)+1, 'We cannot deal with SPX for duplicates yet!');
                                idx2=find(~isfinite(opt.strikes(t-1, idxP)));
                                idxP(idx2)=[];
                                if (length(idxP) > 1)
                                    idxP=[];
                                end
                            else
                                idxP=[];
                            end
                        end
                            
                        if (~isempty(idxC) && ~isempty(idxP))
                            assert(length(idxC)==1 && length(idxP)==1);
                            totAbsDelta=abs(opt.delta(t, idxC)+opt.delta(t, idxP));
                            if (totAbsDelta < minAbsDelta)
                                minAbsDelta=totAbsDelta;
                                minAbsDeltaIdx=[idxC idxP];
                            end
                        end
                    end
                    
                    % finding the straddle with the smallest absolute delta
                    if (minAbsDelta <= maxNetDelta)
                        totTheta(tt, c)=sum(opt.theta(t, minAbsDeltaIdx));
                        totVega(tt, c) =sum(opt.vega(t, minAbsDeltaIdx));
                        ATMidx(tt, c, :)=minAbsDeltaIdx; % Pick the strike prices for call and put to use 
                        
                    end
                end
            end
        end
               
    end
    
    for t=1:lastDayIdx-1
        candidatesIdx=find(isfinite(totTheta(t, 1:end-1)) & isfinite(stk.cl_unadj(t, :)) & isfinite(totVega(t, 1:end-1))); % consider all stock straddles
        
        if (length(candidatesIdx) >= 2 && all(isfinite(ATMidx(t, end, :))) && all(isfinite(ATMidx(t+1, end, :))) && isfinite(totVega(t, end))) % Can have positions only if enough stock candidates AND SPX index options are available both on t and t+1.
            [foo, maxIdx]=sort(totTheta(t, candidatesIdx), 'descend');
            
            assert(all(isfinite(stk.cl_unadj(t, candidatesIdx(maxIdx(1:topN))))), 'Some candidates have no stock price!');
            %             pos(t, candidatesIdx(maxIdx(1:topN)))=1./stk.cl_unadj(t, candidatesIdx(maxIdx(1:topN))); % Number of straddles set equal to $1 of underlying
            pos(t, candidatesIdx(maxIdx(1:topN)))=1./stk.cl_unadj(t, candidatesIdx(maxIdx(1:topN)))/100; % Number of straddles set equal to $1 of underlying. Recall each option has rights to 100 shares.
            
            assert(all(isfinite(totVega(t, candidatesIdx(maxIdx(1:topN))))), 'Some candidates have no vega!');
            totVega_stk=sum( pos(t, candidatesIdx(maxIdx(1:topN))).*totVega(t, candidatesIdx(maxIdx(1:topN))));
            
            %             assert(isfinite(totVega(t, end)), 'Index option has no vega!');
            totVega_spx=totVega(t, end);
            pos(t, end)=-totVega_stk/totVega_spx; % Set number of index straddles such that their total vega is negative that of total stock straddles' vega
            
        end
        
    end
    
    save('dispersion_output');
else
    load('dispersion_output'); % can be fewer than 50 straddles

end

pnl=zeros(size(pos, 1), 1);
mktVal=zeros(size(pos, 1), 1); % gross market value of all stock and index options
for c=1:size(pos, 2)-1
    load(['C:/Projects/Ivy/SPX_200701_201308/cusip_', cusip8{c}], 'bid', 'ask');
    mid=(bid+ask)/2;
    
    for t=2:lastDayIdx-1
        
        if (pos(t-1, c) ~= 0)
            assert(all(isfinite(mid(t-1, ATMidx(t-1, c, :)))));
            
            if (~all(isfinite(mid(t, ATMidx(t-1, c, :))))) % today's prices are missing!
                fprintf(1, 'Missing price for %s on %i\n', cusip8{c}, stk.tday(t));
                assert(all(any(isfinite(mid(t:end, ATMidx(t-1, c, :))), 1))); % There must be some good prices in the future!
            else
                %                 pnl(t)=pnl(t)+100*sum(pos(t-1, c)*(mid(t, ATMidx(t-1, c, :)).*cfadj(t, ATMidx(t-1, c, :)) - mid(t-1, ATMidx(t-1, c, :)).*cfadj(t-1, ATMidx(t-1, c, :)) )./cfadj(t-1, ATMidx(t-1, c, :)));
                pnl(t)=pnl(t)+100*sum(pos(t-1, c)*(mid(t, ATMidx(t-1, c, :)) - mid(t-1, ATMidx(t-1, c, :)) ));
                mktVal(t-1)=mktVal(t-1)+100*sum(abs(pos(t-1, c)*mid(t-1, ATMidx(t-1, c, :))));
            end
            
        end
                
    end
end

% SPX index
for t=2:lastDayIdx-1
 
    if (pos(t-1, end) ~= 0)
        opt1=load(['C:/Projects/Ivy/SPX_index_200701_201308/', num2str(stk.tday(t-1))], 'strikes', 'optExpireDate', 'bid', 'ask', 'cpflag');
        opt2=load(['C:/Projects/Ivy/SPX_index_200701_201308/', num2str(stk.tday(t))],   'strikes', 'optExpireDate', 'bid', 'ask', 'cpflag');
        %         opt3=load(['C:/Projects/Ivy/SPX_index_200701_201308/', num2str(stk.tday(t+1))],   'strikes', 'optExpireDate', 'bid', 'ask', 'cpflag', 'cfadj');

        % prices at t-1
        %         bid1=opt1.bid(1, ATMidx(t-1, end, :)).*opt1.cfadj(1, ATMidx(t-1, end, :)); % 1x2 array, [idxC idxP]
        %         ask1=opt1.ask(1, ATMidx(t-1, end, :)).*opt1.cfadj(1, ATMidx(t-1, end, :));
        bid1=opt1.bid(1, ATMidx(t-1, end, :)); % 1x2 array, [idxC idxP]
        ask1=opt1.ask(1, ATMidx(t-1, end, :));

        % We need price at t corresponding to strikes and
        % optExpireDate at t-1
        
        idxC=find(opt2.strikes==opt1.strikes(ATMidx(t-1, end, 1)) & opt2.optExpireDate==opt1.optExpireDate(ATMidx(t-1, end, 1)) & strcmp('C', opt2.cpflag) );
        idxP=find(opt2.strikes==opt1.strikes(ATMidx(t-1, end, 2)) & opt2.optExpireDate==opt1.optExpireDate(ATMidx(t-1, end, 2)) & strcmp('P', opt2.cpflag) );
        
        assert(length(idxC)==1 && length(idxP)==1);
        
        %         bid2=opt2.bid(1, [idxC idxP]).*opt2.cfadj(1, [idxC idxP]); % 1x2 array, [idxC idxP]
        %         ask2=opt2.ask(1, [idxC idxP]).*opt2.cfadj(1, [idxC idxP]); % 1x2 array, [idxC idxP]
        bid2=opt2.bid(1, [idxC idxP]); % 1x2 array, [idxC idxP]
        ask2=opt2.ask(1, [idxC idxP]); % 1x2 array, [idxC idxP]
        
        mid1=(bid1+ask1)/2;
        mid2=(bid2+ask2)/2;

       
        pnl(t)=pnl(t)+100*pos(t-1, end)*smartsum(mid2-mid1);
        mktVal(t-1)=mktVal(t-1)+100*sum(abs(pos(t-1, end)*mid1));

    end
    
   
end

pnl(lastDayIdx+1:end)=[];
mktVal(lastDayIdx+1:end)=[];

cumPL=smartcumsum(pnl);

dailyret=pnl./backshift(1, mktVal);
dailyret(~isfinite(dailyret))=0;
cumret=cumprod(1+dailyret)-1;

cagr= prod(1+dailyret).^(252/length(dailyret))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret));

[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);

plot(datetime(stk.tday(1:lastDayIdx), 'ConvertFrom', 'yyyyMMdd'), cumret); % Cumulative P&L
title('Dispersion trading SPX');
xlabel('Date');
ylabel('Cumulative Returns');

% CAGR=0.191357 Sharpe=0.884693
% maxDD=-0.507633 maxDDD=757 Calmar ratio=0.376959
% fprintf(1, 'APR/maxDD=%f \n', cumPL(end)/length(cumPL)*252/abs(calculateMaxDD_simple(cumPL)));

% topN=50 minTime2Expire=28 maxNetDelta=0.25;
% APR/maxDD=0.858626 



% plot(datetime(stk.tday(1:lastDayIdx), 'ConvertFrom', 'yyyyMMdd'), cumPL); % Cumulative P&L
% title('Dispersion trading SPX');
% xlabel('Date');
% ylabel('Cumulative P&L');

