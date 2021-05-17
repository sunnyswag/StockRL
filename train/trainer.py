import pandas as pd
import os
import codecs
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
sys.path.append("..")
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils import config
from utils.pull_data import Pull_data
from utils.preprocessors import FeatureEngineer, split_data
from utils.env_retreat_penalty import StockTradingEnvRetreatpenalty
from utils.models import DRL_Agent


class Trainer(object):
    def __init__(self, model_name='a2c' , 
                        total_timesteps= 200000,
                        sub_result_dir="result",
                        data_filename="data.csv",
                        stock_list=config.SSE_50) -> None:
        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.sub_result_dir = sub_result_dir
        self.data_filename = data_filename
        self.stock_list = stock_list
        self.result_dir = self.get_result_dir()
        self.data = self.get_data()
    
    def get_result_dir(self) -> str:
        """获取存储训练结果的路径名"""
        result_dir = os.path.join(os.getcwd(), self.sub_result_dir)
        result_dir = os.path.join(result_dir, self.model_name)
        self.create_result_dir(result_dir)

        return result_dir
    
    def create_result_dir(self, result_dir: str) -> None:
        """创建存储训练结果的文件夹"""
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print("{} 文件夹创建成功!".format(self.sub_result_dir))
        else:
            print("{} 文件夹已存在!".format(self.sub_result_dir))

    def get_data(self) -> pd.DataFrame:
        """获取训练数据"""
        data_dir = os.path.join(os.getcwd(), self.data_filename)

        if os.path.exists(data_dir):
            print("{} 已存在，直接读取！".format(self.data_filename))
            data = pd.read_csv(data_dir)
        else:
            print("未找到 {}，需要下载！".format(self.data_filename))
            data = self.pull_data(data_dir)
        
        return data
    
    def pull_data(self, data_dir: str) -> pd.DataFrame:
        """使用Tushare API下载股票数据并对其进行处理"""
        data = Pull_data(self.stock_list, save_data=False).pull_data()

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

        processed_df.to_csv(data_dir, index = False)
        return processed_df
    
    def train(self) -> None:
        """开始训练"""
        train_data, trade_data = self.data_split()
        env_train, env_trade = self.get_env(train_data, trade_data)

        agent = DRL_Agent(env = env_train)

        model = agent.get_model(self.model_name,  
                                model_kwargs = config.__dict__["{}_PARAMS".format(self.model_name.upper())], 
                                verbose = 0)
        model.learn(total_timesteps = self.total_timesteps, 
                    eval_env = env_trade, 
                    eval_freq = 500,
                    log_interval = 1, 
                    tb_log_name = 'env_cashpenalty_highlr',
                    n_eval_episodes = 1)
        self.save_model(model)

        self.trade(model, env_trade)

    def data_split(self) -> pd.DataFrame:
        """将数据分为训练集和测试集"""
        train_data = split_data(self.data, config.Start_Trade_Date, config.End_Trade_Date)
        trade_data = split_data(self.data, config.End_Trade_Date, config.End_Test_Date)

        self.print_data_information(train_data, trade_data)
        return train_data, trade_data
    
    def print_data_information(self,
                                train_data : pd.DataFrame,
                                trade_data : pd.DataFrame) -> None:
        """打印数据的信息"""
        print("训练数据的范围：{} 至 {}".format(config.Start_Trade_Date, config.End_Trade_Date))
        print("测试数据的范围：{} 至 {}".format(config.End_Trade_Date, config.End_Test_Date))
        print("训练数据的长度: {},测试数据的长度:{}".format(len(train_data), len(trade_data)))
        print("训练集数据 : 测试集数据: {} : {}".format(round(len(train_data)/len(trade_data),1), 1))
        print("train_data.head():")
        print(train_data.head())
        print("trade_data.head():")
        print(trade_data.head())

    def get_env(self, 
                train_data : pd.DataFrame, 
                trade_data : pd.DataFrame) -> DummyVecEnv:
        """分别返回训练环境和测试环境"""
        e_train_gym = StockTradingEnvRetreatpenalty(df = train_data,
                                                    random_start = True,
                                                    **config.ENV_PARAMS)
        env_train, _ = e_train_gym.get_sb_env()

        e_trade_gym = StockTradingEnvRetreatpenalty(df = trade_data,
                                                    random_start = False,
                                                    **config.ENV_PARAMS)
        env_trade, _ = e_trade_gym.get_sb_env()

        return env_train, env_trade

    def save_model(self, model) -> None:
        model_path = os.path.join(self.result_dir, "scaling_reward_{}.model".format(self.model_name))
        model.save(model_path)

    def trade(self, 
                model, 
                e_trade_gym : DummyVecEnv) -> None:
        """使用训练好的模型进行交易"""
        df_account_value, df_actions = DRL_Agent.DRL_prediction(model = model, 
                                                                environment = e_trade_gym)
        self.save_trade_result(df_account_value, df_actions)
        self.print_trade_result(df_account_value, df_actions)
    
    def save_trade_result(self, 
                    df_account_value : pd.DataFrame, 
                    df_actions : pd.DataFrame) -> None:
        """保存交易后的数据"""
        account_value_path = os.path.join(self.result_dir, "df_account_value_{}.csv".format(self.model_name))
        df_account_value.to_csv(account_value_path, index=False)

        actions_path = os.path.join(self.result_dir, "df_actions_{}.csv".format(self.model_name))
        df_actions.to_csv(actions_path, index=False)
    
    def print_trade_result(self, 
                    df_account_value : pd.DataFrame, 
                    df_actions : pd.DataFrame) -> None:
        """打印交易后的数据"""
        print("回测的时间窗口：{} 至 {}".format(config.End_Trade_Date, config.End_Test_Date))

        print("查看日账户净值")
        print("开始: ")
        print(df_account_value.head())
        print("")
        print("结束: ")
        print(df_account_value.tail())

        print("查看每日所作的交易")
        print(df_actions.tail())


if __name__ == "__main__":
    Trainer().train()