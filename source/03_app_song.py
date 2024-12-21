import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output
from openbb import obb
from sklearn.decomposition import PCA

obb.user.preferences.output_type = "dataframe"

pio.templates.default = "plotly"

# Dash 앱을 Bootstrap 스타일로 초기화
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 티커 입력을 위한 UI 컴포넌트
ticker_field = [
    html.Label("티커 심볼 입력:"),
    dcc.Input(
        id="ticker-input",
        type="text",
        placeholder="콤마로 구분된 티커 입력 (예: AAPL,MSFT)",
        style={"width": "50%"},
    ),
]

# PCA 컴포넌트 수 선택을 위한 UI 컴포넌트
components_field = [
    html.Label("컴포넌트 수 선택:"),
    dcc.Dropdown(
        id="component-dropdown",
        options=[{"label": i, "value": i} for i in range(1, 6)],
        value=3,
        style={"width": "50%"},
    ),
]

# 날짜 범위 선택을 위한 UI 컴포넌트
date_picker_field = [
    html.Label("날짜 범위 선택:"),  # 날짜 선택기 레이블
    dcc.DatePickerRange(
        id="date-picker",
        start_date=datetime.datetime.now() - datetime.timedelta(365 * 3),
        end_date=datetime.datetime.now(),  # 오늘 날짜를 기본값으로 설정
        display_format="YYYY-MM-DD",
    ),
]

# 업데이트 트리거를 위한 제출 버튼
submit = [
    html.Button("제출", id="submit-button"),
]

# 앱 레이아웃 정의
app.layout = dbc.Container(
    [
        html.H1("주식 수익률의 PCA 분석"),
        # 티커 입력
        dbc.Row([dbc.Col(ticker_field)]),
        dbc.Row([dbc.Col(components_field)]),
        dbc.Row([dbc.Col(date_picker_field)]),
        dbc.Row([dbc.Col(submit)]),
        # 차트
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="bar-chart")], width=4),
                dbc.Col([dcc.Graph(id="line-chart")], width=4),
                dbc.Col([dcc.Graph(id="scatter-plot")], width=4),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("bar-chart", "figure"),
        Output("line-chart", "figure"),
        Output("scatter-plot", "figure"),
    ],
    [Input("submit-button", "n_clicks")],
    [
        dash.dependencies.State("ticker-input", "value"),
        dash.dependencies.State("component-dropdown", "value"),
        dash.dependencies.State("date-picker", "start_date"),
        dash.dependencies.State("date-picker", "end_date"),
    ],
)
def update_graphs(n_clicks, tickers, n_components, start_date, end_date):
    """
    사용자 입력에 기반하여 그래프를 업데이트합니다.

    매개변수
    ----------
    n_clicks : int
        제출 버튼이 클릭된 횟수
    tickers : str
        콤마로 구분된 티커 심볼 목록
    n_components : int
        계산할 주성분 수
    start_date : str
        YYYY-MM-DD 형식의 시작 날짜
    end_date : str
        YYYY-MM-DD 형식의 종료 날짜

    반환값
    -------
    tuple
        막대 차트, 선 차트, 산점도를 위한 세 개의 Plotly 그림을 포함하는 튜플
    """
    if not tickers:
        return {}, {}, {}

    # 사용자 입력 파싱
    tickers = tickers.split(",")

    # 날짜 문자열을 datetime 객체로 변환
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%f").date()
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S.%f").date()

    # 주식 히스토리 데이터 다운로드
    data = obb.equity.price.historical(
        tickers, start_date=start_date, end_date=end_date, provider="yfinance"
    ).pivot(columns="symbol", values="close")
    # 일간 수익률 계산
    daily_returns = data.pct_change().dropna()

    # 일간 수익률에 PCA 적용
    pca = PCA(n_components=n_components)
    pca.fit(daily_returns)

    explained_var_ratio = pca.explained_variance_ratio_

    # 개별 설명 분산을 위한 막대 차트 생성
    bar_chart = go.Figure(
        data=[
            go.Bar(
                x=["PC" + str(i + 1) for i in range(n_components)],
                y=explained_var_ratio,
            )
        ],
        layout=go.Layout(
            title="컴포넌트별 설명 분산",
            xaxis=dict(title="주성분"),
            yaxis=dict(title="설명 분산"),
        ),
    )

    # 누적 설명 분산을 위한 선 차트 생성
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    line_chart = go.Figure(
        data=[
            go.Scatter(
                x=["PC" + str(i + 1) for i in range(n_components)],
                y=cumulative_var_ratio,
                mode="lines+markers",
            )
        ],
        layout=go.Layout(
            title="누적 설명 분산",
            xaxis=dict(title="주성분"),
            yaxis=dict(title="누적 설명 분산"),
        ),
    )

    # 요인 노출도 계산
    X = np.asarray(daily_returns)

    factor_returns = pd.DataFrame(
        columns=["f" + str(i + 1) for i in range(n_components)],
        index=daily_returns.index,
        data=X.dot(pca.components_.T),
    )

    # 각 주식에 대한 요인 노출도 계산
    factor_exposures = pd.DataFrame(
        index=["f" + str(i + 1) for i in range(n_components)],
        columns=daily_returns.columns,
        data=pca.components_,
    ).T

    labels = factor_exposures.index
    data = factor_exposures.values

    # 첫 두 요인에 대한 산점도 생성
    scatter_plot = go.Figure(
        data=[
            go.Scatter(
                x=factor_exposures["f1"],
                y=factor_exposures["f2"],
                mode="markers+text",
                text=labels,
                textposition="top center",
            )
        ],
        layout=go.Layout(
            title="첫 두 요인의 산점도",
            xaxis=dict(title="요인 1"),
            yaxis=dict(title="요인 2"),
        ),
    )

    return bar_chart, line_chart, scatter_plot


if __name__ == "__main__":
    # Dash 앱 실행
    app.run_server(debug=True)
