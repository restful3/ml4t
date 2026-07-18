clear;

load('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/fundamentals', 'tday', 'syms', 'indQ', 'indY', 'ARQ*', 'ARY*', 'ART*');

% [num, txt]=xlsread('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/SF1_20150630.csv');
fid=fopen('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/SF1_20150630.csv', 'r');

b=1;
while (1)
    fprintf(1, 'Start block %i ...\n', b);
    C=textscan(fid, '%s%s%f', 1000000, 'Delimiter', ',');
    if (size(C{:, 1}, 1))==0
        break;
    end
    
    num=cell2mat(C(:, 3));
    txt= C(:, [1 2]);
        
    sym_ind_freq=txt{:, 1};
    yyyy_mm_dd=txt{:, 2};
    value=num(:, 1);
    
    SHARESBAS=NaN(length(tday), length(syms));
    
    for s=1:length(syms)
        match=regexp(sym_ind_freq, [syms{s}, '_SHARESBAS']);
        idx=find(~cellfun('isempty', match));
        
        if (~isempty(idx))
            yyyymmdd=str2double(cellstr(datestr(datenum(yyyy_mm_dd(idx), 'yyyy-mm-dd'), 'yyyymmdd')));
            
            [~, idx1, idx2]=intersect(tday, yyyymmdd);
            
            SHARESBAS(idx1, s)=value(idx(idx2));
        end
    end
    b=b+1;

end

fclose(fid);

save('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/fundamentals1', 'tday', 'syms', 'indQ', 'indY', 'ARQ*', 'ARY*', 'ART*', 'SHARESBAS');