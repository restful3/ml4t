#region imports
from AlgorithmImports import *

import plotly.graph_objects as go
#endregion


def rough_daily_backtest(qb, portfolio_weights):
    history = qb.history(
        list(portfolio_weights.columns), 
        portfolio_weights.index[0]-timedelta(1), portfolio_weights.index[-1], 
        Resolution.DAILY
    )['close'].unstack(0)
    # Calculate the daily returns of each asset.
    # `shift` so that it's the return from today to tomorrow.
    asset_daily_returns = history.pct_change(1).shift(-1).dropna() 
    asset_equity_curves = (asset_daily_returns + 1).cumprod() - 1
    strategy_daily_returns = (
        asset_daily_returns * portfolio_weights
    ).sum(axis=1)
    strategy_equity_curve = (strategy_daily_returns + 1).cumprod()
    # Plot the results.
    go.Figure(
        [
            go.Scatter(
                x=asset_equity_curves.index, y=asset_equity_curves[symbol] + 1,
                name=f"Buy-and-hold {str(symbol).split(' ')[0]}"
            )
            for symbol in asset_equity_curves.columns
        ] + [
            go.Scatter(
                x=strategy_equity_curve.index, y=strategy_equity_curve,
                name='Strategy'
            )
        ],
        dict(
            title="Rough Backtest Results<br><sup>The equity curves of "
                + "buy-and-hold for each asset and for the portfolio given "
                + "the asset weights</sup>",
            xaxis_title="Date", yaxis_title="Equity"
        )
    ).show()

