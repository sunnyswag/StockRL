import tushare as ts
import pandas as pd
import config
import time
import os
from datetime import datetime

class Pull_data():
    """使用 Tushare API 拉取数据
    
    Attributes
    ----------
        ticker_list : list
            股票的列表
        start_date : str
            日期的开始时间
        end_date : str
            日期的结束时间
        tushare_tocken : str
            使用 Tushare API 下载文件时所需要用到的 tocken
        data_dir : str
            数据存储的目录
        save_data : boolean
            是否保存数据
    
    Methods
    -------
        init_tushare()
            初始化 Tushare API 的参数
        init_dir()
            初始化存储数据的文件目录
        pull_data()
            从 Tushare API 拉取数据
    """

    def __init__(self, ticker_list, start_date = config.Start_Date, end_date = config.End_Date, 
                    tushare_tocken = config.Tushare_Tocken,
                    data_dir = config.Dir_Data, save_data = config.Save_Data):
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.tushare_tocken = tushare_tocken
        self.data_dir = data_dir
        self.save_data = save_data

        self.ticker_len = len(self.ticker_list)
        self.date_time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.result_name = self.date_time + "_len{}".format(self.ticker_len) +  ".csv"

        self.init_tushare()
        self.init_dir()

    def init_tushare(self):
        """初始化 Tushare API 的参数
        Parameters
        ----------

        Returns
        -------
        """
        self.pro = ts.pro_api()
        ts.set_token(self.tushare_tocken)

    def init_dir(self):
        """初始化存储数据的文件目录
        Parameters
        ----------

        Returns
        -------
        """
        if self.save_data:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                print("成功创建 {} 目录!".format(self.data_dir))
            else:
                print(" {} 目录已存在!".format(self.data_dir))
        os.chdir(self.data_dir)
    
    def pull_data(self):
        """从 Tushare API 拉取数据
        Parameters
        ----------

        Returns
        -------
            `pd.DataFrame`
                    7 列: date, tick, open, high, low, close, volume
        """
        data_df = pd.DataFrame()
        stock_num = 0

        print("--- 开始下载 ----")
        for ticker in self.ticker_list:

            stock_num += 1
            if stock_num % 10 == 0:
                print("下载进度 : {}%".format(stock_num / len(self.ticker_list) * 100))
            
            try:
                data_tmp = ts.pro_bar(ts_code=ticker, adj='qfq', start_date=self.start_date, end_date=self.end_date)
                data_tmp = data_tmp.set_index("trade_date", drop=True) # 将 trade_date 列设为索引
                data_df = data_df.append(data_tmp)
            except:
                print("休息 3s")
                time.sleep(3)
        
        # 删除一些列并更改列名
        data_df = data_df.reset_index()
        data_df = data_df.drop(["pre_close", "change", "pct_chg", "amount"], axis = 1)
        data_df.columns = [
                "date",
                "tic",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]

        # 更改 date 列数据格式, 添加 day 列(星期一为 0), 再将格式改回成 str
        data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # 删除为空的数据行
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

        print("DataFrame 的大小: ", data_df.shape)

        if self.save_data:
            print("文件保存在 : {}".format(os.path.join(self.data_dir, self.result_name)))
            data_df.to_csv(self.result_name)
        return data_df

if __name__ == "__main__":
    df = Pull_data(ticker_list=config.SSE_50[:10]).pull_data()
    print(df.head())