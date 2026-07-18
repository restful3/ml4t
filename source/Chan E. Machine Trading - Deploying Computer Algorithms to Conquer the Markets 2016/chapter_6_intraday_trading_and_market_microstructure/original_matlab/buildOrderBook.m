clear;

[num, txt]=xlsread('COINSETTER_execution_reports_03_2014-10_2014.csv'); % time is UTC, ms since 1970 (i.e. Unix time in ms in UTC.)

% unixtime2datenum(num(1, 2)/1000000)


ns=num(:, 2); % nanoseconds
side=txt(2:end, 3); % BUY or SELL
price=num(:, 5);
orderSize=round(num(:, 7)*1000000); % turn order size into integers
action=txt(2:end, 10); % WORKING_CONFIRMED, CANCELED_CONFIRMED, FILL_CONFIRMED, PARTIAL_FILL_CONFIRMED

secs=ceil(ns/1000000000); % round up to nearest second, so time points to end of 1-sec bar

bid=NaN(size(secs));
ask=NaN(size(secs));
bidSize=zeros(size(bid));
askSize=zeros(size(ask));
buyOrderBook=BinarySearchTree;
sellOrderBook=BinarySearchTree;

assert(strcmp(action(1), 'WORKING_CONFIRMED'));
if (strcmp(side(1), 'BUY'))
    bid(1)=price(1);
    bidSize(1)=orderSize(1);
    buyOrderBook.Insert(price(1), bidSize(1));
elseif (strcmp(side(1), 'SELL'))
    ask(1)=price(1);
    askSize(1)=orderSize(1);
    sellOrderBook.Insert(price(1), askSize(1));
end


fprintf(1, 'Bid=%f Ask=%f\n', bid(1), ask(1));


for t=2:length(secs)
   
    if (strcmp(action(t), 'WORKING_CONFIRMED'))
        if (strcmp(side(t), 'BUY'))
            if (price(t) > bid(t-1) || isnan(bid(t-1)))
                bid(t)=price(t);
                bidSize(t)=orderSize(t);
                buyOrderBook.Insert(price(t), bidSize(t));
            elseif (price(t) == bid(t-1))
                bid(t)=bid(t-1);
                bidSize(t)=bidSize(t-1)+orderSize(t);
                buyOrderBook.Delete(buyOrderBook.Search(price(t)));
                buyOrderBook.Insert(price(t), bidSize(t));
            else
                bid(t)=bid(t-1);
                bidSize(t)=bidSize(t-1);
                buyOrderBook.Insert(price(t), orderSize(t));
            end
            ask(t)=ask(t-1);
            askSize(t)=askSize(t-1);
        elseif (strcmp(side(t), 'SELL'))
            if (price(t) < ask(t-1) || isnan(ask(t-1)))
                ask(t)=price(t);
                askSize(t)=orderSize(t);                
                sellOrderBook.Insert(price(t), askSize(t));
            elseif (price(t) == ask(t-1))
                ask(t)=ask(t-1);
                askSize(t)=askSize(t-1)+orderSize(t);
                sellOrderBook.Delete(sellOrderBook.Search(price(t)));
                sellOrderBook.Insert(price(t), askSize(t));
            else
                ask(t)=ask(t-1);
                askSize(t)=askSize(t-1);
                sellOrderBook.Insert(price(t), orderSize(t));

            end
            bid(t)=bid(t-1);
            bidSize(t)=bidSize(t-1);
        end
    elseif (strcmp(action(t), 'CANCEL_CONFIRMED') || strcmp(action(t), 'FILL_CONFIRMED') || strcmp(action(t), 'PARTIAL_FILL_CONFIRMED') )
        if (strcmp(side(t), 'BUY'))
            if (price(t) == bid(t-1))
                if (orderSize(t) < bidSize(t-1))
                    bid(t)=bid(t-1);
                    bidSize(t)=bidSize(t-1)-orderSize(t);
                    buyOrderBook.Delete(buyOrderBook.Search(price(t)));
                    buyOrderBook.Insert(price(t), bidSize(t));
                else
                    assert(orderSize(t)==bidSize(t-1));
                    buyOrderBook.Delete(buyOrderBook.Search(price(t)));
                    if (~buyOrderBook.IsEmpty)
                        T=buyOrderBook.Maximum();
                        bid(t)=T.key;
                        bidSize(t)=T.value;
                    end
                end
            elseif (price(t) < bid(t-1))
                bid(t)=bid(t-1);
                bidSize(t)=bidSize(t-1);
                
                T=buyOrderBook.Search(price(t));
                if (~isnan(T)) % Some trades are wrong
                    if (orderSize(t) == T.value)
                        buyOrderBook.Delete(T);
                    else
                        %                         assert(orderSize(t) < T.value);
                        if (orderSize(t) > T.value)
                            fprintf(1, 'Trade size %i > bid size %i!\n', orderSize(t), T.value);
                        end
                        buyOrderBook.Delete(T);
                        buyOrderBook.Insert(price(t), T.value - orderSize(t));
                    end
                end
            end
            ask(t)=ask(t-1);
            askSize(t)=askSize(t-1);

        elseif (strcmp(side(t), 'SELL'))
            if (price(t) == ask(t-1))
                if (orderSize(t) < askSize(t-1))
                    ask(t)=ask(t-1);
                    askSize(t)=askSize(t-1)-orderSize(t);
                    sellOrderBook.Delete(sellOrderBook.Search(price(t)));
                    sellOrderBook.Insert(price(t), askSize(t));
                else
                    assert(orderSize(t)==askSize(t-1));
                    sellOrderBook.Delete(sellOrderBook.Search(price(t)));
                    if (~sellOrderBook.IsEmpty)
                        T=sellOrderBook.Minimum();
                        ask(t)=T.key;
                        askSize(t)=T.value;
                    end
                end
            elseif (price(t) > ask(t-1))
                ask(t)=ask(t-1);
                askSize(t)=askSize(t-1);
                T=sellOrderBook.Search(price(t));
                if (~isnan(T)) % Some trades are wrong
                    if (orderSize(t) == T.value)
                        sellOrderBook.Delete(T);
                    else
                        %                         assert(orderSize(t) < T.value);
                        if (orderSize(t) > T.value)
                            fprintf(1, 'Trade size %i > ask size %i!\n', orderSize(t), T.value);
                        end
                        sellOrderBook.Delete(T);
                        sellOrderBook.Insert(price(t), T.value - orderSize(t));
                    end
                end
            end
            
            bid(t)=bid(t-1);
            bidSize(t)=bidSize(t-1);

        end
        
    end
    
    %     assert(isnan(bid(t)) || isnan(ask(t)) || (bid(t) <= ask(t)));
    if (~(isnan(bid(t)) || isnan(ask(t)) || (bid(t) <= ask(t))))
        %         fprintf(1, 'Bid %f > ask %f!\n', bid(t), ask(t));
        0;
    else
        1;
    end

    fprintf(1, '%i %s Bid=%f bidSize=%i Ask=%f askSize=%i\n', t, datestr(unixtime2datenum(secs(t))), bid(t), bidSize(t), ask(t), askSize(t));
end
    

