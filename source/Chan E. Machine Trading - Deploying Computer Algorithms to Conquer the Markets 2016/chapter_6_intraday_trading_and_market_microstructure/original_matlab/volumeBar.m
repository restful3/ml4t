% Use output from unionNanexData3.m, with occasional lastPrice=NaN and
% vol=0 if there are change of quotes in a 25ms bar.
% Include change of buyVol
% Use volume bars and BVC.
clear;

lookback=100; % Num bars used for stddev of volume-weighted price changes
entryThreshold=0.95; % Fraction of Buy volume vs Total volume
exitThreshold =0.5; % 0
multiplier=50; % ES

tcost=0.095; % $0.095 per contract: minimum exchange and clearing fee. 0.16 is more typical. IB charges 2.47.
useMidPrice=false; % true if use mid point as  execution price, otherwise buy at ask, sell at bid

load('inputData_ESZ12_volbar500_TAQ_20121001_v003', 'dn', 'lastPrice', 'bid', 'ask');

badData=~isfinite(lastPrice);
dn(badData)=[];
bid(badData)=[];
ask(badData)=[];
lastPrice(badData)=[];


prevTradePrice=backshift(1, fillMissingData(lastPrice));
deltaPrice=lastPrice-prevTradePrice;

midPrice=(bid+ask)/2;
roundPrice=inline('round(x*4)/4');

%% order flow
buyVol=NaN(size(lastPrice)); % Buy volume as fraction of total volume
for t=lookback+1:length(buyVol)
    myDeltaPrice=deltaPrice(t-lookback:t-1);
    myDeltaPrice(~isfinite(myDeltaPrice))=[];
    buyVol(t)=cdf('Normal', deltaPrice(t), 0, std(myDeltaPrice));
end


%% order flow per bar
pos=0;
cumPL=0;
dailyPL=0;
numTrade=0;
dailyNumTrade=0;
for t=lookback+1:length(buyVol)
    
    if (buyVol(t) > entryThreshold)
        if ( pos <= 0)
            if (pos < 0)
                %                 fprintf(1, '%s: OrderFlow=%i: BUY 2, pos=1\n', hhmmssfff{t}, ordflow_lookback);
                if (~useMidPrice)
                    dailyPL=dailyPL+(entryP-ask(t)); % Aggressive
                else
                    dailyPL=dailyPL+(entryP-roundPrice(midPrice(t)));
                end
                dailyNumTrade=dailyNumTrade+2;
                if (~useMidPrice)
                    entryP=ask(t);
                else
                    entryP=roundPrice(midPrice(t));
                end
            else
                %                 fprintf(1, '%s: OrderFlow=%i: BUY 1, pos=1\n', hhmmssfff{t}, ordflow_lookback);
                if (~useMidPrice)
                    entryP=ask(t);
                else
                    entryP=roundPrice(midPrice(t));
                end
                dailyNumTrade=dailyNumTrade+1;
            end
            pos=1;
            
        end
  
    elseif (1-buyVol(t) > entryThreshold)
        if (pos >= 0)
            if (pos > 0)
                %                 fprintf(1, '%s: OrderFlow=%i: SELL 2, pos=-1\n', hhmmssfff{t}, ordflow_lookback);
                if (~useMidPrice)
                    dailyPL=dailyPL+(bid(t)-entryP);
                else
                    dailyPL=dailyPL+(roundPrice(midPrice(t))-entryP);
                end
                dailyNumTrade=dailyNumTrade+2;
                if (~useMidPrice)
                    entryP=bid(t);
                else
                    entryP=roundPrice(midPrice(t));
                end
            else
                %                 fprintf(1, '%s: OrderFlow=%i: SELL 1, pos=-1\n', hhmmssfff{t}, ordflow_lookback);
                if (~useMidPrice)
                    entryP=bid(t);
                else
                    entryP=roundPrice(midPrice(t));
                end
                dailyNumTrade=dailyNumTrade+1;
            end
            pos=-1;
            
        end
       
    else
        if ( buyVol(t) <= exitThreshold && pos > 0)
            %             fprintf(1, '%s: OrderFlow=%i: SELL 1, pos=0\n', hhmmssfff{t}, ordflow_lookback);
            if (~useMidPrice)
                dailyPL=dailyPL+(bid(t)-entryP);
            else
                dailyPL=dailyPL+(roundPrice(midPrice(t))-entryP);
            end
            dailyNumTrade=dailyNumTrade+1;
            pos=0;
           

        elseif ( 1-buyVol(t) <= exitThreshold && pos < 0)
            %             fprintf(1, '%s: OrderFlow=%i: BUY 1, pos=1\n', hhmmssfff{t}, ordflow_lookback);
            if (~useMidPrice)
                dailyPL=dailyPL+(entryP-ask(t));
            else
                dailyPL=dailyPL+(entryP-roundPrice(midPrice(t)));
            end
            dailyNumTrade=dailyNumTrade+1;
            pos=0;
           

        end
    
    end
    
    
    if (t < length(dn) && floor(dn(t)) < floor(dn(t+1)))
        fprintf(1, '%s: dailyPL (no cost)=%f dailyPL/trade=%f dailyNumTrade=%i dailyTcost=%f\n', datestr(dn(t), 'yyyymmdd'), multiplier*dailyPL,  multiplier*dailyPL/dailyNumTrade, dailyNumTrade, dailyNumTrade*tcost);
        
        cumPL=cumPL+dailyPL;
        numTrade=numTrade+dailyNumTrade;
        dailyPL=0;
        dailyNumTrade=0;
    end
    
end

if (~(floor(dn(t-1)) < floor(dn(t))))
    fprintf(1, '%s: dailyPL (no cost)=%f dailyPL/trade=%f dailyNumTrade=%i dailyTcost=%f\n', datestr(dn(t), 'yyyymmdd'), multiplier*dailyPL,  multiplier*dailyPL/dailyNumTrade, dailyNumTrade, dailyNumTrade*tcost);
    
    cumPL=cumPL+dailyPL;
    numTrade=numTrade+dailyNumTrade;
    dailyPL=0;
    dailyNumTrade=0;
end

fprintf(1, 'TotPL (no cost)=%f TotPL/trade=%f numTrade=%i TotCost=%f\n', multiplier*cumPL,  multiplier*cumPL/numTrade, numTrade, tcost*numTrade);