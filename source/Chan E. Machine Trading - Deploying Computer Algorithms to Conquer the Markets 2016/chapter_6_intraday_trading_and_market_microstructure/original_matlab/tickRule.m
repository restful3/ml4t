% Use output from unionNanexData3.m, with occasional tradePrice=NaN and
% vol=0 if there are change of quotes in a 25ms bar.
% Use Tick Rule to classify. Use limit orders.
% Sum order flow over 1 minute.
clear;


entryThreshold=66; % Roughly 5th (-66) and 95th (66) percentile
exitThreshold=0;
lookback=60; % 1 min
multiplier=50;
tcost=0.095; % $0.095 per contract: minimum exchange and clearing fee. 0.16 is more typical. IB charges 2.47.
useMidPrice=true; % true if use mid point as  execution price, otherwise buy at ask, sell at bid

load('inputData_ESZ12_TAQ_20121001_v003', 'dn', 'tradePrice', 'vol', 'bid', 'ask'); % dn is datenum

prevTradePrice=backshift(1, fillMissingData(tradePrice));
midPrice=(bid+ask)/2;
roundPrice=inline('round(x*4)/4');

%% order flow
buy=tradePrice > prevTradePrice;
sell=tradePrice < prevTradePrice;

for t=2:length(tradePrice)
    if (tradePrice(t)==prevTradePrice(t))
        buy(t)=buy(t-1);
        sell(t)=sell(t-1);
    end
end

ordflow=zeros(size(tradePrice));
ordflow(buy)=vol(buy);
ordflow(sell)=-vol(sell);

cumOrdflow=smartcumsum(ordflow);

%% order flow per minute
pos=0;
cumPL=0;
dailyPL=0;
numTrade=0;
dailyNumTrade=0;
for t=1:length(cumOrdflow)
    idx=find( dn <= dn(t)-lookback/60/60/24);
    
    if (~isempty(idx))
        ordflow_lookback=cumOrdflow(t)-cumOrdflow(idx(end));
        
        if (ordflow_lookback > entryThreshold)
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
                        entryP=ask(t); % Aggressive
                    else
                        entryP=roundPrice(midPrice(t));
                    end
                else
                    %                 fprintf(1, '%s: OrderFlow=%i: BUY 1, pos=1\n', hhmmssfff{t}, ordflow_lookback);
                    if (~useMidPrice)
                        entryP=ask(t); % Aggressive
                    else
                        entryP=roundPrice(midPrice(t));
                    end
                    dailyNumTrade=dailyNumTrade+1;
                end
                pos=1;
                
            end
        elseif (ordflow_lookback < -entryThreshold)
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
            if (ordflow_lookback <= exitThreshold && pos > 0)
                %             fprintf(1, '%s: OrderFlow=%i: SELL 1, pos=0\n', hhmmssfff{t}, ordflow_lookback);
                if (~useMidPrice)
                    dailyPL=dailyPL+(bid(t)-entryP);
                else
                    dailyPL=dailyPL+(roundPrice(midPrice(t))-entryP);
                end
                dailyNumTrade=dailyNumTrade+1;
                pos=0;
            elseif (ordflow_lookback >= -exitThreshold && pos < 0)
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
end

if (~(floor(dn(t-1)) < floor(dn(t))))
    fprintf(1, '%s: dailyPL (no cost)=%f dailyPL/trade=%f dailyNumTrade=%i dailyTcost=%f\n', datestr(dn(t), 'yyyymmdd'), multiplier*dailyPL,  multiplier*dailyPL/dailyNumTrade, dailyNumTrade, dailyNumTrade*tcost);
    
    cumPL=cumPL+dailyPL;
    numTrade=numTrade+dailyNumTrade;
    dailyPL=0;
    dailyNumTrade=0;
end



fprintf(1, 'TotPL (no cost)=%f TotPL/trade=%f numTrade=%i TotCost=%f\n', multiplier*cumPL,  multiplier*cumPL/numTrade, numTrade, tcost*numTrade);

% With mid point

% 20121001: dailyPL (no cost)=800.000000 dailyPL/trade=0.684932 dailyNumTrade=1168 dailyTcost=110.960000
% TotPL (no cost)=800.000000 TotPL/trade=0.684932 numTrade=1168 TotCost=110.960000

% With bid-ask
% 20121001: dailyPL (no cost)=-6687.500000 dailyPL/trade=-5.619748 dailyNumTrade=1190 dailyTcost=113.050000
% TotPL (no cost)=-6687.500000 TotPL/trade=-5.619748 numTrade=1190 TotCost=113.050000
