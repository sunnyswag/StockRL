import pandas as pd
import os
import codecs
from stable_baselines3.common.vec_env import DummyVecEnv
from data import Data
import sys
from argparse import ArgumentParser

sys.path.append("..")
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils import config
from utils.env_retreat_penalty import StockTradingEnvRetreatpenalty
from utils.models import DRL_Agent


class Trader(object):
    def __init__(self, model_name='a2c') -> None:
        self.model_name = model_name
        self.train_dir = "train"
        self.data_dir = "data_file"
        self.trade_dir = "result"
        self.create_trade_dir()
    
    def create_trade_dir(self) -> None:
        """创建存储训练结果的文件夹"""
        if not os.path.exists(self.trade_dir):
            os.makedirs(self.trade_dir)
            print("{} 文件夹创建成功!".format(self.trade_dir))
        else:
            print("{} 文件夹已存在!".format(self.trade_dir))
    
    def get_trade_data(self):
        trade_data_path = os.path.join(self.data_dir, "trade.csv")
        if not os.path.exists(trade_data_path):
            Data.pull_data()

        return pd.read_csv(trade_data_path)

    def get_env(self, trade_data : pd.DataFrame) -> DummyVecEnv:
        """分别返回训练环境和测试环境"""
        e_trade_gym = StockTradingEnvRetreatpenalty(df = trade_data,
                                                    random_start = False,
                                                    **config.ENV_PARAMS)
        env_trade, _ = e_trade_gym.get_sb_env()

        return env_trade
    
    def get_model(self, agent):
        model = agent.get_model(self.model_name,  
                                model_kwargs = config.__dict__["{}_PARAMS".format(self.model_name.upper())], 
                                verbose = 0)
        model_dir = os.path.join(self.train_dir, "{}.model".format(self.model_name))
        
        if os.path.exists(model_dir):
            model.load(model_dir)
            return model
        else:
            return None

    def trade(self) -> None:
        """使用训练好的模型进行交易"""
        trade_data = self.get_trade_data()
        e_trade_gym = self.get_env(trade_data)
        agent = DRL_Agent(env = e_trade_gym)

        model = self.get_model(agent)

        if model is not None:
            df_account_value, df_actions = DRL_Agent.DRL_prediction(model = model, 
                                                                    environment = e_trade_gym)
            self.save_trade_result(df_account_value, df_actions)
            self.print_trade_result(df_account_value, df_actions)
        else:
            print("{} 文件夹中未找到 {} model，请先运行 trainer.py 或者将训练好的 {} model 放入 {} 中"
            .format(self.train_dir, self.model_name, self.model_name, self.train_dir))
    
    def save_trade_result(self, 
                    df_account_value : pd.DataFrame, 
                    df_actions : pd.DataFrame) -> None:
        """保存交易后的数据"""
        account_value_path = os.path.join(self.trade_dir, "account_value_{}.csv".format(self.model_name))
        df_account_value.to_csv(account_value_path, index=False)

        actions_path = os.path.join(self.trade_dir, "actions_{}.csv".format(self.model_name))
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


def start_trade():
    parser = ArgumentParser(description="set parameters for train mode")
    parser.add_argument(
        '--model', '-m',
        dest='model',
        default='a2c',
        help='choose the model you want to train',
        metavar="MODEL",
        type=str
    )

    options = parser.parse_args()
    Trader(model_name = options.model).trade()

if __name__ == "__main__":
    start_trade()