import numpy as np
import pandas as pd
import random
from copy import deepcopy
import gym
import time
from gym.utils import seeding
from gym import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger

class StockTradingEnvCashpenalty(gym.Env):
    """
        actions : list
            action 为买的金额
        state : list
            由三部分构成
            1、当前现金
            2、每个股票的持仓市值
            3、股票数 * 技术指标数
    """


    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        df,
        buy_cost_pct = 3e-3,
        sell_cost_pct = 3e-3,
        date_col_name = "date",
        hmax = 10,
        discrete_actions = False,
        shares_increment = 1,
        turbulence_threshold = None,
        print_verbosity = 10,
        initial_amount = 1e6,
        daily_information_cols = ["open", "close", "high", "low", "volume"],
        cache_indicator_data = True,
        cash_penalty_proportion = 0.2,
        random_start = True,
        patient = False,
        currency = "￥"
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
        self.episode = -1
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.cash_penalty_proportion = cash_penalty_proportion
        if self.cache_indicator_data:
            # cashing data 的结构:
            # [[date1], [date2], [date3], ...]
            # date1 : [stock1 * cols, stock2 * cols, ...]
            print("caching data")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("data cached!")
        
    def seed(self, seed=None):
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)
    
    @property
    def current_step(self):
        return self.date_index - self.starting_point
    
    @property
    def cash_on_hand(self):
        return self.state_memory[-1][0]
    
    @property
    def holdings(self):
        return self.state_memory[-1][1: len(self.assets) + 1]

    @property
    def closings(self):
        return np.array(self.get_date_vector(self.date_index,cols=["close"]))

    def get_date_vector(self, date, cols=None):
        if(cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            res = []
            for asset in self.assets:
                tmp_res = trunc_df[trunc_df[self.stock_col] == asset]
                res += tmp_res.loc[date, cols].tolist()
            assert len(res) == len(self.assets) * len(cols)
            return res
    
    def reset(self):
        self.seed()
        self.sum_trades = 0
        if self.random_start:
            self.starting_point = random.choice(range(int(len(self.dates) * 0.5)))
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": []
        }
        init_state = np.array(
            [self.initial_amount] 
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(init_state)
        return init_state

    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"]
        cash_pct = self.account_information["cash"][-1] / self.account_information["total_assets"][-1]
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            # TODO
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}", # 也没有这行
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%", # 没有这行吗
            f"{cash_pct*100:0.2f}%"
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.record("environment/GainLoss_pct", (gl_pct - 1) * 100)
        logger.record(
            "environment/total_assets",
            int(self.account_information["total_assets"][-1])
        )
        reward_pct = gl_pct
        logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
        logger.record("environment/total_trades", self.sum_trades)
        logger.record(
            "environment/avg_daily_trades",
            self.sum_trades / (self.current_step)
        )
        logger.record(
            "environment/avg_daily_trades_per_asset",
            self.sum_trades / (self.current_step) / len(self.assets)
        )
        logger.record("environment/completed_steps", self.current_step)
        logger.record(
            "environment/sum_rewards", np.sum(self.account_information["reward"])
        )
        logger.record(
            "environment/cash_proportion",
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )

        return state, reward, True, {}

    def log_header(self):
        if not self.printed_header:
            self.template = "{0:4}|{1:4}{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
            # 0, 1, 2, ... 是序号
            # 4, 4, 15, ... 是占位格的大小
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD_unsc",
                    "GAINLOSS_PCT",
                    "CASH_PROPORTION"
                )
            )
            self.printed_header = True

    def get_reward(self):
        pass

    def get_transactions(self, actions):
        pass

    def step(self):
        pass
    
