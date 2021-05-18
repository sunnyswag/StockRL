from typing import List
import tushare as ts
import pandas as pd
from utils import config
import time
from datetime import datetime

class Pull_data():
    """从 Tushare API 拉取数据
    
    Attributes
        ticker_list: 股票列表
        start_date: 日期的开始时间
        end_date: 日期的结束时间
        tushare_tocken: 使用 Tushare API 下载文件时所需要用到的 tocken
        pull_index: 拉取的是不是指数
    """

    def __init__(self, ticker_list: List, 
                start_date: str = config.Start_Date, 
                end_date: str = config.End_Date, 
                tushare_tocken: str = config.Tushare_Tocken,
                pull_index: bool = False) -> None:
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.tushare_tocken = tushare_tocken
        self.pull_index = pull_index

        self.ticker_len = len(self.ticker_list)
        self.date_time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

        self.init_tushare()

    def init_tushare(self) -> None:
        """初始化 Tushare API 的参数"""
        ts.set_token(self.tushare_tocken)
        self.pro = ts.pro_api()
    
    def pull_data(self) -> pd.DataFrame:
        """从 Tushare API 拉取数据"""
        data_df = pd.DataFrame()
        stock_num = 0

        print("   --- 开始下载 ----")
        for ticker in self.ticker_list:

            stock_num += 1
            if stock_num % 10 == 0:
                print("   下载进度 : {}%".format(stock_num / len(self.ticker_list) * 100))
            
            try:
                if not self.pull_index:
                    data_tmp = ts.pro_bar(ts_code=ticker, adj='qfq', 
                                            start_date=self.start_date, end_date=self.end_date)
                else:
                    data_tmp = self.pro.index_daily(ts_code=ticker, adj='qfq', 
                                                    start_date=self.start_date, end_date=self.end_date)
                data_tmp = data_tmp.set_index("trade_date", drop=True) # 将 trade_date 列设为索引
                data_df = data_df.append(data_tmp)
            except:
                print("休息 3s")
                time.sleep(3)
        print("   --- 下载完成 ----")

        # 删除一些列并更改列名
        data_df = data_df.reset_index()
        data_df = data_df.drop(["pre_close", "change", "pct_chg", "amount"], axis = 1)
        data_df.columns = ["date", "tic", "open", "high", "low", "close", "volume"]

        # 更改 date 列数据格式, 添加 day 列(星期一为 0), 再将格式改回成 str
        data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # 删除为空的数据行
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

        print("DataFrame 的大小: ", data_df.shape)
        return data_df

if __name__ == "__main__":
    df = Pull_data(ticker_list=config.SSE_50[:10]).pull_data()
    print(df.head())