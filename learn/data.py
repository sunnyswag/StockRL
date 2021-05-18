import sys
import codecs
import os
from typing import List
import pandas as pd

sys.path.append("..")
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils.pull_data import Pull_data
from utils.preprocessors import FeatureEngineer, split_data
from utils import config

class Data(object):
    """用来获取数据的类

    Attributes:
        stock_list: 股票代码
    """

    def __init__(self, 
                stock_list: List = config.SSE_50) -> None:
        self.stock_list = stock_list
        self.data_dir = "data_file"
        self.create_data_dir()

    def create_data_dir(self) -> None:
        """创建存储数据的文件夹"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print("{} 文件夹创建成功!".format(self.data_dir))
        else:
            print("{} 文件夹已存在!".format(self.data_dir))
    
    def pull_data(self) -> pd.DataFrame:
        """使用Tushare API下载股票数据并对数据进行预处理"""
        data = Pull_data(self.stock_list).pull_data()

        data.sort_values(['date', 'tic'], ignore_index=True).head()
        print("数据下载的时间区间为：{} 至 {}".format(config.Start_Date, config.End_Date))
        print("下载的股票列表为: ")
        print(self.stock_list)

        processed_df = FeatureEngineer(use_technical_indicator=True).preprocess_data(data)
        processed_df['amount'] = processed_df.volume * processed_df.close
        processed_df['change'] = (processed_df.close-processed_df.open)/processed_df.close
        processed_df['daily_variance'] = (processed_df.high-processed_df.low)/processed_df.close
        processed_df = processed_df.fillna(0)

        print("技术指标列表: ")
        print(config.TECHNICAL_INDICATORS_LIST)
        print("技术指标数: {}个".format(len(config.TECHNICAL_INDICATORS_LIST)))
        print(processed_df.head())

        processed_df.to_csv(os.path.join(self.data_dir, "data.csv"), index = False)
        self.data_split(processed_df)

    def data_split(self, data: pd.DataFrame) -> pd.DataFrame:
        """将数据分为训练数据集和交易数据集"""
        train_data = split_data(data, config.Start_Trade_Date, config.End_Trade_Date)
        trade_data = split_data(data, config.End_Trade_Date, config.End_Test_Date)

        self.print_data_information(train_data, trade_data)
        self.save_data(train_data, trade_data)
    
    def print_data_information(self,
                                train_data: pd.DataFrame,
                                trade_data: pd.DataFrame) -> None:
        print("训练数据的范围：{} 至 {}".format(config.Start_Trade_Date, config.End_Trade_Date))
        print("测试数据的范围：{} 至 {}".format(config.End_Trade_Date, config.End_Test_Date))
        print("训练数据的长度: {},测试数据的长度:{}".format(len(train_data), len(trade_data)))
        print("训练集数据 : 测试集数据: {} : {}".format(round(len(train_data)/len(trade_data),1), 1))
        print("train_data.head():")
        print(train_data.head())
        print("trade_data.head():")
        print(trade_data.head())

    def save_data(self, 
                    train_data: pd.DataFrame,
                    trade_data: pd.DataFrame) -> None:
        train_data.to_csv(os.path.join(self.data_dir, "train.csv"), index = False)
        trade_data.to_csv(os.path.join(self.data_dir, "trade.csv"), index = False)

if __name__ == "__main__":
    Data().pull_data()