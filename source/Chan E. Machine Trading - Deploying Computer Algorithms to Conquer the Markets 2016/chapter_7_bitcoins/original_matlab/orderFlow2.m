% Adapted to bitstamp data
% Sum order flow over 1 minute.
clear;

entryThreshold=90;
exitThreshold=0;
lookback=60; % 1 min
%% 1 sec trade ticks

T=readtable('2014-12.csv'); % time is "yyyy-mm-dd HH:MM:SS.FFFFFF" Price is BTCUSD
% T=readtable('trades_bitstamp/2015-01.csv'); % time is "yyyy-mm-dd HH:MM:SS.FFFFFF" Price is BTCUSD

timestr=char(regexprep(table2cell(T(:, 1)), '(:\d\d)$', '$1.000000')); % Some times are "yyyy-mm-dd HH:MM:SS"
dotIdx=regexp(timestr(1, :), '\.');
timestr1=timestr(:, 1:(dotIdx-1));
timestr2=num2str(roundoff(str2double(cellstr(timestr(:, dotIdx:end))), 3), '%0.3f'); % Reduce to 0.FFF
timestr2=timestr2(:, 2:end); % Reduce to .FFF
timestr=cellstr([timestr1 timestr2]);

dn=datenum(timestr, 'yyyy-mm-dd HH:MM:SS.FFF');


side=table2array(T(:, 2)); % buy (1) or sell (-1)
tradeSize=table2array(T(:, 3)); 
tradePrice=table2array(T(:, 4));

%% order flow
buy=side==1;
sell=side==-1;

ordflow=zeros(size(tradePrice));
ordflow(buy)=tradeSize(buy);
ordflow(sell)=-tradeSize(sell);

cumOrdflow=smartcumsum(ordflow);

%% order flow per minute
pos=0;
cumPL=0;
dailyPL=0;
numTrade=0;
dailyNumTrade=0;
for t=1:length(cumOrdflow)
    idx=find( dn  <= dn(t)-lookback/60/60/24);
    
    if (~isempty(idx))
        ordflow_lookback=cumOrdflow(t)-cumOrdflow(idx(end));
        
        if (ordflow_lookback > entryThreshold)
            if ( pos <= 0)
                if (pos < 0)
                    dailyPL=dailyPL+(entryP-tradePrice(t));
                    dailyNumTrade=dailyNumTrade+2;
                    entryP=tradePrice(t);
                else
                    entryP=tradePrice(t);
                    dailyNumTrade=dailyNumTrade+1;
                end
                pos=1;
                
            end
        elseif (ordflow_lookback < -entryThreshold)
            if (pos >= 0)
                if (pos > 0)
                    dailyPL=dailyPL+(tradePrice(t)-entryP);
                    dailyNumTrade=dailyNumTrade+2;
                    entryP=tradePrice(t);
                else
                    entryP=tradePrice(t);
                    dailyNumTrade=dailyNumTrade+1;
                end
                pos=-1;
                
            end
        else
            if (ordflow_lookback <= exitThreshold && pos > 0)
                dailyPL=dailyPL+(tradePrice(t)-entryP);
                dailyNumTrade=dailyNumTrade+1;
                pos=0;
            elseif (ordflow_lookback >= -exitThreshold && pos < 0)
                dailyPL=dailyPL+(entryP-tradePrice(t));
                dailyNumTrade=dailyNumTrade+1;
                pos=0;
            end
        end
        
        
        if (t < length(dn) && floor(dn(t)) < floor(dn(t+1)))
            fprintf(1, '%s: dailyPL (no cost)=%f dailyPL/trade=%f dailyNumTrade=%i\n', datestr(dn(t), 'yyyymmdd'), dailyPL,  dailyPL/dailyNumTrade, dailyNumTrade);
            
            cumPL=cumPL+dailyPL;
            numTrade=numTrade+dailyNumTrade;
            dailyPL=0;
            dailyNumTrade=0;
        end
        
    end
end

if (~(floor(dn(t-1)) < floor(dn(t))))
    % P&L is in USD

    fprintf(1, '%s: dailyPL (no cost)=%f dailyPL/trade=%f dailyNumTrade=%i\n', datestr(dn(t), 'yyyymmdd'), dailyPL,  dailyPL/dailyNumTrade, dailyNumTrade);
    
    cumPL=cumPL+dailyPL;
    numTrade=numTrade+dailyNumTrade;
    dailyPL=0;
    dailyNumTrade=0;
end


% P&L is in USD
fprintf(1, 'TotPL (no cost)=%f TotPL/trade=%f numTrade=%i\n',cumPL, cumPL/numTrade, numTrade); % bid-ask spread is around 0.5

% lookback=1 min, entryThreshold=90 ===
% 20141201: dailyPL (no cost)=5.900000 dailyPL/trade=0.983333 dailyNumTrade=6
% 20141202: dailyPL (no cost)=-3.630000 dailyPL/trade=-0.453750 dailyNumTrade=8
% 20141203: dailyPL (no cost)=-2.640000 dailyPL/trade=-0.660000 dailyNumTrade=4
% 20141204: dailyPL (no cost)=4.490000 dailyPL/trade=0.374167 dailyNumTrade=12
% 20141205: dailyPL (no cost)=0.120000 dailyPL/trade=0.010000 dailyNumTrade=12
% 20141206: dailyPL (no cost)=0.000000 dailyPL/trade=NaN dailyNumTrade=0
% 20141207: dailyPL (no cost)=-1.250000 dailyPL/trade=-0.625000 dailyNumTrade=2
% 20141208: dailyPL (no cost)=1.680000 dailyPL/trade=0.084000 dailyNumTrade=20
% 20141209: dailyPL (no cost)=3.310000 dailyPL/trade=0.110333 dailyNumTrade=30
% 20141210: dailyPL (no cost)=1.460000 dailyPL/trade=0.208571 dailyNumTrade=7
% 20141211: dailyPL (no cost)=1.490000 dailyPL/trade=0.059600 dailyNumTrade=25
% 20141212: dailyPL (no cost)=-1.470000 dailyPL/trade=-0.183750 dailyNumTrade=8
% 20141213: dailyPL (no cost)=0.000000 dailyPL/trade=NaN dailyNumTrade=0
% 20141214: dailyPL (no cost)=2.280000 dailyPL/trade=0.380000 dailyNumTrade=6
% 20141215: dailyPL (no cost)=-0.610000 dailyPL/trade=-0.101667 dailyNumTrade=6
% 20141216: dailyPL (no cost)=-1.130000 dailyPL/trade=-0.037667 dailyNumTrade=30
% 20141217: dailyPL (no cost)=13.600000 dailyPL/trade=0.377778 dailyNumTrade=36
% 20141218: dailyPL (no cost)=5.130000 dailyPL/trade=0.197308 dailyNumTrade=26
% 20141219: dailyPL (no cost)=1.850000 dailyPL/trade=0.102778 dailyNumTrade=18
% 20141220: dailyPL (no cost)=3.730000 dailyPL/trade=0.310833 dailyNumTrade=12
% 20141221: dailyPL (no cost)=0.730000 dailyPL/trade=0.182500 dailyNumTrade=4
% 20141222: dailyPL (no cost)=2.220000 dailyPL/trade=0.370000 dailyNumTrade=6
% 20141223: dailyPL (no cost)=-0.020000 dailyPL/trade=-0.005000 dailyNumTrade=4
% 20141224: dailyPL (no cost)=-1.340000 dailyPL/trade=-0.223333 dailyNumTrade=6
% 20141225: dailyPL (no cost)=-2.250000 dailyPL/trade=-0.375000 dailyNumTrade=6
% 20141226: dailyPL (no cost)=-2.000000 dailyPL/trade=-0.500000 dailyNumTrade=4
% 20141227: dailyPL (no cost)=0.620000 dailyPL/trade=0.103333 dailyNumTrade=6
% 20141228: dailyPL (no cost)=-1.790000 dailyPL/trade=-0.447500 dailyNumTrade=4
% 20141229: dailyPL (no cost)=-0.380000 dailyPL/trade=-0.095000 dailyNumTrade=4
% 20141230: dailyPL (no cost)=1.780000 dailyPL/trade=0.089000 dailyNumTrade=20
% 20141231: dailyPL (no cost)=0.660000 dailyPL/trade=0.165000 dailyNumTrade=4
% TotPL (no cost)=32.540000 TotPL/trade=0.096845 numTrade=336

