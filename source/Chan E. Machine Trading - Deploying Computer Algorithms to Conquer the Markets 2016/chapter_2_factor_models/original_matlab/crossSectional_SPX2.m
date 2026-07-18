% crossSectional_SPX2.m
% Use all data, not just size-independent ones
clear;

p=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'syms', 'bid', 'ask');
load('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/fundamentals', 'tday', 'syms', 'ARQ*', 'ART*');
load('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/indicators', 'indQ', 'indT');

assert(all(strcmp(syms, p.syms)));
assert(all(tday==p.tday));
mid=(p.bid+p.ask)/2;


holdingDays=252/4; % hold a quarter

retQ=calculateReturns(mid, holdingDays); % quarterly return

retFut=fwdshift(holdingDays+1, retQ); % shifted next quarter's return to today's row to use as response variable. Can enter only at next day's close.

trainset=1:floor(length(tday)/2);

% Combine different independent variables into one matrix X for training
X=NaN(length(trainset)*length(syms), length(indQ)+length(indT));

Y=reshape(retFut(trainset, :), [length(trainset)*length(syms) 1]); % dependent variable

for iQ=1:length(indQ)
    eval(['X(:, iQ)=reshape(ARQ_', indQ{iQ}, '(trainset, :), [length(trainset)*length(syms) 1]);']);
end

for iT=1:length(indT)
    eval(['X(:, iQ+iT)=reshape(ART_', indT{iT}, '(trainset, :), [length(trainset)*length(syms) 1]);']);
end

% Linear regression
model_train=fitlm(X, Y,  'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.
% model_train = 
% 
% 
% Linear regression model:
%     y ~ [Linear formula with 113 terms in 112 predictors]
% 
% Estimated Coefficients:
%                          Estimate                    SE                    tStat           pValue
%                    _____________________    ____________________    ___________________    ______
% 
%     (Intercept)        -1.96747568568733        4.04749524600285     -0.486097096131325    NaN   
%     x1              6.20102736028422e-09    2.12133652918672e-09       2.92317002746451    NaN   
%     x2              5.45947091259444e-09    3.35866062231307e-09       1.62549049353922    NaN   
%     x3             -4.41191218906903e-09    3.24019207485037e-09      -1.36162057283989    NaN   
%     x4                                 0                       0                    NaN    NaN   
%     x5               0.00205400332326225       0.125206865786812     0.0164048777225968    NaN   
%     x6             -1.57617743456487e-08    5.39689767636613e-09      -2.92052495541504    NaN   
%     x7             -1.06454027868767e-09    4.72814315819369e-10      -2.25149756060762    NaN   
%     x8                                 0                       0                    NaN    NaN   
%     x9                 0.293199115566096       0.141941205302789       2.06563777544824    NaN   
%     x10               -0.503730829749694       0.249653240686766      -2.01772197454353    NaN   
%     x11             -2.5022168268678e-09    6.00275895479309e-10      -4.16844461973577    NaN   
%     x12                                0                       0                    NaN    NaN   
%     x13                                0                       0                    NaN    NaN   
%     x14                -1.81963454927565       0.417838954565882      -4.35487052940331    NaN   
%     x15                                0                       0                    NaN    NaN   
%     x16            -3.74500932687209e-09    1.15754400343968e-08     -0.323530623090239    NaN   
%     x17                                0                       0                    NaN    NaN   
%     x18                0.448840732930722        6.17191230857567      0.072723122184849    NaN   
%     x19                0.307628825509979        6.25688249615037     0.0491664699951217    NaN   
%     x20             -5.6699972409644e-09    3.18461607560688e-09      -1.78043352992994    NaN   
%     x21             1.67283853016617e-08    5.62542081740467e-09       2.97371269539608    NaN   
%     x22              -0.0720740027172782      0.0934893043405003     -0.770933137493197    NaN   
%     x23             4.19391983862596e-09    1.41691013780157e-09        2.9599053085562    NaN   
%     x24            -2.75649357836558e-10    7.59620047638727e-10     -0.362877939692893    NaN   
%     x25            -1.23312799960965e-08    3.10388077844411e-08     -0.397285877786768    NaN   
%     x26            -2.20100048314831e-10      6.750637190973e-10     -0.326043367593729    NaN   
%     x27                                0                       0                    NaN    NaN   
%     x28                                0                       0                    NaN    NaN   
%     x29            -2.73938956987482e-09    3.29013408408725e-09     -0.832607273704707    NaN   
%     x30            -1.68993221191213e-08    5.54692033889438e-09      -3.04661345154449    NaN   
%     x31            -1.01759191038498e-08    3.19797502060394e-09      -3.18198830143708    NaN   
%     x32            -1.06981106073313e-08    3.22260265586575e-09      -3.31971134817341    NaN   
%     x33            -4.81753383904334e-09    1.15768229798056e-08     -0.416136089102074    NaN   
%     x34             2.79100226993003e-08    7.52967372888568e-09       3.70667092681992    NaN   
%     x35             1.65017038763245e-08    5.48270466978015e-09       3.00977434864938    NaN   
%     x36                                0                       0                    NaN    NaN   
%     x37              1.2097454671538e-08    4.64702346757396e-09       2.60326954575369    NaN   
%     x38             1.63144574087851e-09    1.14126275576929e-08      0.142950931556406    NaN   
%     x39                                0                       0                    NaN    NaN   
%     x40            -9.68817201220886e-09    6.94244442227367e-09      -1.39549867783255    NaN   
%     x41             6.57130455898228e-10    8.71828501409081e-10      0.753738212086608    NaN   
%     x42                0.145491789389937      0.0705199586730678       2.06312924918803    NaN   
%     x43                                0                       0                    NaN    NaN   
%     x44            -4.90877182900084e-10    8.26418777703994e-10     -0.593981158395103    NaN   
%     x45            -2.05459130174002e-10    2.60662977944748e-10     -0.788217535892467    NaN   
%     x46             -1.5891312004984e-11    3.08751289970472e-10     -0.051469621411149    NaN   
%     x47             1.12065534995383e-08    1.20165241834238e-08      0.932595260366318    NaN   
%     x48            -2.79878151578732e-09    2.77318527239462e-09      -1.00922990744524    NaN   
%     x49            -6.35236393921349e-08    6.56662452314211e-08     -0.967371275276435    NaN   
%     x50             4.84081026778344e-08    6.16439919388472e-08      0.785285007594199    NaN   
%     x51                                0                       0                    NaN    NaN   
%     x52            -4.79134635765361e-09    1.25340265522912e-08     -0.382267130013201    NaN   
%     x53               -0.257179064437542      0.0818552044867888      -3.14187797892619    NaN   
%     x54             4.77988780881408e-09    3.04954454652998e-09       1.56741039059522    NaN   
%     x55             5.52134272558547e-11    4.39823612504588e-10       0.12553538665521    NaN   
%     x56                0.367555519467998       0.804735332261083      0.456740874587013    NaN   
%     x57            -4.08248847361893e-09     5.9850593185104e-09     -0.682113285158719    NaN   
%     x58                                0                       0                    NaN    NaN   
%     x59            -3.75626339750487e-09    6.26394432482093e-09     -0.599664237534909    NaN   
%     x60                                0                       0                    NaN    NaN   
%     x61                 4.41297838835659        1.52035847543352       2.90259071111387    NaN   
%     x62                                0                       0                    NaN    NaN   
%     x63            -9.10009599183926e-10    3.88792016782702e-09     -0.234060772830255    NaN   
%     x64                                0                       0                    NaN    NaN   
%     x65                                0                       0                    NaN    NaN   
%     x66                -5.48549670582638         5.0552560182446      -1.08510759613935    NaN   
%     x67                 4.70312794961667        5.11326328035947      0.919789905534853    NaN   
%     x68                -2.73184034779926        1.25465973171253      -2.17735556402251    NaN   
%     x69                 2.80217356267604        1.31576590790375       2.12969005036799    NaN   
%     x70             7.77098295799513e-10    5.16311396400014e-10       1.50509615169806    NaN   
%     x71              -0.0445945501571995      0.0870318627784742     -0.512393377936857    NaN   
%     x72                -0.11384029592837      0.0894142784401963       -1.2731780417432    NaN   
%     x73            -5.54368514869751e-10    5.51769643314427e-09     -0.100471006621479    NaN   
%     x74                0.573986494390166       0.302869565040711       1.89516069174205    NaN   
%     x75            -1.10232739173295e-09    1.06959631329264e-09      -1.03060133812498    NaN   
%     x76                 8.59767269203555        4.40685368286094       1.95097757056774    NaN   
%     x77                0.367567379043865         2.9154119697849      0.126077337560971    NaN   
%     x78             1.43091491892053e-08    6.57593770425726e-09       2.17598612285234    NaN   
%     x79               -0.240840882977955       0.160154174019936       -1.5038064693086    NaN   
%     x80            -6.41050924588333e-10    5.82496190776665e-09     -0.110052380554385    NaN   
%     x81             1.81547650945125e-09    1.38429211503567e-09        1.3114836743865    NaN   
%     x82             1.84429379228141e-09    1.30020733173095e-09       1.41846130787935    NaN   
%     x83             5.53685147565261e-09    7.17485165465783e-09      0.771702572004837    NaN   
%     x84            -6.33494120101713e-10    6.26683850012312e-09     -0.101086715429042    NaN   
%     x85             1.32542400991135e-09    5.71152276551673e-09      0.232061407145846    NaN   
%     x86                                0                       0                    NaN    NaN   
%     x87               -0.394929431267936       0.174611870545665      -2.26175591632902    NaN   
%     x88            -2.70215911990509e-09    6.59381316125964e-09     -0.409802196971697    NaN   
%     x89              6.1126757245913e-09    4.16419415658727e-09       1.46791323716781    NaN   
%     x90                                0                       0                    NaN    NaN   
%     x91             2.91001150151577e-08    1.05946052751206e-08       2.74669176052209    NaN   
%     x92                -0.09269897937992       0.113143588290572     -0.819303866710089    NaN   
%     x93                                0                       0                    NaN    NaN   
%     x94                0.258665047616362           1.15811050281      0.223350921167493    NaN   
%     x95              -0.0334666294815009      0.0237785434025535      -1.40742975357805    NaN   
%     x96               0.0489738895312781      0.0288487511890551       1.69760864899616    NaN   
%     x97                                0                       0                    NaN    NaN   
%     x98                -1.51381058154123        1.93764843167584     -0.781261738091443    NaN   
%     x99                 1.59715903498645        1.99916388087126      0.798913510927576    NaN   
%     x100           -1.15933447201006e-10    1.81419056821884e-10     -0.639036765111335    NaN   
%     x101               0.428020335373225       0.788620849385323      0.542745396227906    NaN   
%     x102           -1.41171712555127e-08    5.29339295243237e-09      -2.66694186174592    NaN   
%     x103               -24.5763570872446        12.1587014089152      -2.02129785580756    NaN   
%     x104                3.39567239606998        2.05960531404371       1.64870054127172    NaN   
%     x105                 13.341327554349        15.0324641377659      0.887501039888178    NaN   
%     x106           -2.95245751680781e-11    9.22653481868117e-10    -0.0319996355601445    NaN   
%     x107            1.73390233796661e-07    6.92670231251287e-08       2.50321474741938    NaN   
%     x108           -1.54368654508719e-07    6.48052598779344e-08      -2.38203897028551    NaN   
%     x109                2.36389510287304        1.39900588193677       1.68969632893929    NaN   
%     x110              0.0743803859084243      0.0286799276109961        2.5934649109751    NaN   
%     x111                -1.2113023853639        2.79617244928639     -0.433200171782337    NaN   
%     x112                               0                       0                    NaN    NaN   
% 
% 
% Number of observations: 103, Error degrees of freedom: 12
% Root Mean Squared Error: 0.105
% R-squared: 0.961,  Adjusted R-Squared 0.671
% F-statistic vs. constant model: 3.31, p-value = 0.0125

retPred=reshape(predict(model_train, X), [length(trainset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

% Backtest trading model based on "prediction" on training set
longs=backshift(1, retPred>0); %1 day later
shorts=backshift(1, retPred<0);

longs(1, :)=false;
shorts(1, :)=false;

positions=zeros(size(retPred));

for h=0:holdingDays-1
    long_lag=backshift(h, longs);
    long_lag(isnan(long_lag))=false;
    long_lag=logical(long_lag);
    
    short_lag=backshift(h, shorts);
    short_lag(isnan(short_lag))=false;
    short_lag=logical(short_lag);
    
    positions(long_lag)=positions(long_lag)+1;
    positions(short_lag)=positions(short_lag)-1;
end

ret1=calculateReturns(mid, 1);

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=0.476424 Sharpe ratio=1.684890 maxDD=-0.199723 maxDDD=44 Calmar ratio=2.385424


% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

X=NaN(length(testset)*length(syms), length(indQ)+length(indT));

Y=reshape(retFut(testset, :), [length(testset)*length(syms) 1]); % dependent variable

for iQ=1:length(indQ)
    eval(['X(:, iQ)=reshape(ARQ_', indQ{iQ}, '(testset, :), [length(testset)*length(syms) 1]);']);
end

for iT=1:length(indT)
    eval(['X(:, iQ+iT)=reshape(ART_', indT{iT}, '(testset, :), [length(testset)*length(syms) 1]);']);
end

retPred=reshape(predict(model_train, X), [length(testset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

longs=backshift(1, retPred>0); %1 day later
shorts=backshift(1, retPred<0);

longs(1, :)=false;
shorts(1, :)=false;

positions=zeros(size(retPred));

for h=0:holdingDays-1
    long_lag=backshift(h, longs);
    long_lag(isnan(long_lag))=false;
    long_lag=logical(long_lag);
    
    short_lag=backshift(h, shorts);
    short_lag(isnan(short_lag))=false;
    short_lag=logical(short_lag);
    
    positions(long_lag)=positions(long_lag)+1;
    positions(short_lag)=positions(short_lag)-1;
end

dailyRet=smartsum(backshift(1, positions).*ret1(testset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Stepwise regression on SPX fundamental facotrs');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=-0.040973 Sharpe ratio=-0.371196 maxDD=-0.230558 maxDDD=839 Calmar ratio=-0.177712


