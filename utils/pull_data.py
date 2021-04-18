import tushare as ts
import pandas as pd
from config import Index_Dic
import time
import os

class Pull_data():
    """使用 Tushare API 拉取数据
    
    Attributes
    ----------
        start_date : str
            日期的开始时间
        end_date : str
            日期的结束时间
        ticker_list : list
            股票的列表
    
    Methods
    -------
        pull_data()
            从 Tushare API 拉取数据
    """

    def __init__(self, ticker_list = Index_Dic, start_date = '20090101', end_date = '20210101', 
                    data_dir = 'datasets', ticker_name = 'SSE_50'):
        self.pro = ts.pro_api()
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_name = ticker_name
        self.ticker_list = ticker_list[self.ticker_name]
        self.data_dir = data_dir
        os.chdir(self.data_dir)
        ts.set_token("c576df5b626df4f37c30bae84520d70c7945a394d7ee274ef2685444")
    
    def pull_data(self):
        data_df = pd.DataFrame()
        stock_num = 0

        print("------ Start Download -------")
        for ticker in self.ticker_list:

            stock_num += 1
            if stock_num % (len(self.ticker_list) / 10) == 0:
                print("download process finish {}%".format(stock_num / len(self.ticker_list) * 100))
            
            try:
                data_tmp = self.pro.daily(ts_code=ticker, start_date=self.start_date, end_date=self.end_date)
                data_df = data_df.append(data_tmp)
            except:
                print("time to sleep")
                time.sleep(3)
        
        print("save file in {}".format(os.path.join(self.data_dir, self.ticker_name + ".csv")))
        data_df.to_csv(self.ticker_name + ".csv")

Pull_data().pull_data()