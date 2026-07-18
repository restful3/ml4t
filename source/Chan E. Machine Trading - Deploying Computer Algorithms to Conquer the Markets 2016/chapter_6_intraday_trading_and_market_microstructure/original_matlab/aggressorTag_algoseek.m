% Use Algoseek TAQ 1ms data which has aggressor flag to compute order flow.
% Sum order flow over 1 minute.
clear;


entryThreshold=66; % Roughly 5th (-66) and 95th (66) percentile
exitThreshold=0;
lookback=60; % 1 min
multiplier=50;
tcost=0.095; % $0.095 per contract: minimum exchange and clearing fee. 0.16 is more typical. IB charges 2.47.
useMidPrice=false; % true if use mid point as  execution price, otherwise buy at ask, sell at bid

% fin=fopen('N:/Data/Futures/20121001/ES_1212_FUT_Q_T.csv', 'r');
% fid=fopen('N:/Data/Futures/20151209/ES_1512_FUT_Q_T.csv', 'r');
fid=fopen('N:/Data/Futures/20150828/ES_1509_FUT_Q_T.csv', 'r');
C=textscan(fid, '%s%s%s%s%s', 1, 'Delimiter', ','); % header
C=textscan(fid, '%s%s%f%u%s', 'Delimiter', ',');
fclose(fid);

dn=datenum(C{:, 1}, 'HH:MM:SS.FFF'); % in UTC
event=C{:, 2}; %  B, A, or T
price=C{:, 3};
quantity=C{:, 4};
aggressor=C{:, 5}; % B or S

bid=NaN(size(price));
ask=NaN(size(price));

bid(strcmp(event, 'B'))=price(strcmp(event, 'B'));
ask(strcmp(event, 'A'))=price(strcmp(event, 'A'));
bid=fillMissingData(bid);
ask=fillMissingData(ask);

midPrice=(bid+ask)/2;
roundPrice=inline('round(x*4)/4');

%% order flow
ordflow=zeros(size(price));
ordflow(strcmp(event, 'T') & strcmp(aggressor, 'B'))= quantity(strcmp(event, 'T') & strcmp(aggressor, 'B'));
ordflow(strcmp(event, 'T') & strcmp(aggressor, 'S'))=-quantity(strcmp(event, 'T') & strcmp(aggressor, 'S'));

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

% 20121001: dailyPL (no cost)=762.500000 dailyPL/trade=10.166667 dailyNumTrade=75 dailyTcost=7.125000
% TotPL (no cost)=762.500000 TotPL/trade=10.166667 numTrade=75 TotCost=7.125000

% With bid-ask
% 20121001: dailyPL (no cost)=300.000000 dailyPL/trade=4.000000 dailyNumTrade=75 dailyTcost=7.125000
% TotPL (no cost)=300.000000 TotPL/trade=4.000000 numTrade=75 TotCost=7.125000

% 20151209: dailyPL (no cost)=-887.500000 dailyPL/trade=-20.639535 dailyNumTrade=43 dailyTcost=4.085000
% TotPL (no cost)=-887.500000 TotPL/trade=-20.639535 numTrade=43 TotCost=4.085000

% 20150828: dailyPL (no cost)=-250.000000 dailyPL/trade=-14.705882 dailyNumTrade=17 dailyTcost=1.615000
% TotPL (no cost)=-250.000000 TotPL/trade=-14.705882 numTrade=17 TotCost=1.615000

