from typing import List
import pandas as pd
from pyfolio import timeseries
import pyfolio
from copy import deepcopy

from utils.pull_data import Pull_data
from utils import config

def get_daily_return(
    df: pd.DataFrame,
    value_col_name: str = "account_value"
) -> pd.Series:
    """获取每天的涨跌值"""
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")

    return pd.Series(df["daily_return"], index = df.index)

def backtest_stats(
    account_value: pd.DataFrame, 
    value_col_name: str = "account_value"
) -> pd.Series:
    """对回测数据进行分析"""
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB"
    )
    print(perf_stats_all)

    return perf_stats_all

def backtest_plot(
    account_value: pd.DataFrame,
    baseline_start: str = config.End_Trade_Date,
    baseline_end: str = config.End_Test_Date,
    baseline_ticker: List = config.SSE_50_INDEX,
    value_col_name: str = "account_value"
) -> None:
    """对回测数据进行分析并画图"""
    df = deepcopy(account_value)
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker,
        start=baseline_start,
        end=baseline_end
    )

    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns,
            benchmark_rets=baseline_returns,
            set_context=False
        )

def get_baseline(
    ticker: List, start: str, end: str
    ) -> pd.DataFrame:
    """获取指数的行情数据"""
    baselines = Pull_data(
        ticker_list=ticker,
        start_date=start,
        end_date=end,
        pull_index=True
    ).pull_data()

    return baselines