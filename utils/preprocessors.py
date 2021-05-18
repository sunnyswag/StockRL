from typing import List
import itertools
import pandas as pd
from stockstats import StockDataFrame as Sdf

from utils import config

class FeatureEngineer():
    """进行数据预处理的类

    Attributes
        return_full_table: 是否对当前时间段未上市的公司的所有行置零
        use_technical_indicator: 是否使用技术指标
        tech_indicator_list: 技术指标列表
    """

    def __init__(
        self, 
        return_full_table: bool = True,
        use_technical_indicator: bool = True,
        tech_indicator_list: List = config.TECHNICAL_INDICATORS_LIST
    ):
        self.return_full_table = return_full_table
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """对数据进行预处理"""
        if self.use_technical_indicator :
            df = self.add_technical_indicator(df)
            print("成功添加技术指标")

        if self.return_full_table:
            df = self.full_table(df)
            print("对当前时间段未上市的公司的所有行置零")
        
        return df

    def add_technical_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """对数据添加技术指标"""
        df = data.copy()
        df = df.sort_values(by=['tic', 'date'])
        # 获取 Sdf 的对象
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        # 添加技术指标
        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for ticker in unique_ticker:
                tmp_df = pd.DataFrame(stock[stock.tic == ticker][indicator])
                tmp_df['tic'] = ticker
                tmp_df['date'] = df[df.tic == ticker]['date'].to_list()
                indicator_df = indicator_df.append(tmp_df, ignore_index = True)
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        df = df.sort_values(by=['date', 'tic'])

        return df

    def full_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """对当前时间段未上市的公司的所有行置零"""
        
        # 获取 tic 和 date 的所有组合
        ticker_list = df['tic'].unique().tolist()
        date_list = list(pd.date_range(df['date'].min(), df['date'].max()).astype(str))
        combination = list(itertools.product(date_list, ticker_list))

        # 将 combination 和 df 按照 columns=["date", "tic"] 进行合并，将两表相交为空的行置零
        df_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(df, on=["date", "tic"], how="left")
        df_full = df_full[df_full["date"].isin(df["date"])].fillna(0)
        df_full = df_full.sort_values(['date', 'tic'], ignore_index=True)
        
        return df_full

def split_data(
    df: pd.DataFrame, start: str, end: str
    ) -> pd.DataFrame:
    """将数据集按照起止时间进行拆分"""
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(['date', 'tic'], ignore_index = True)
    data.index = data.date.factorize()[0]

    return data

if __name__ == "__main__":
    from pull_data import Pull_data

    df = Pull_data(config.SSE_50[:2], save_data=False).pull_data()
    df = FeatureEngineer().preprocess_data(df)
    df = split_data(df, '2009-01-01','2019-01-01')
    print(df.head())
