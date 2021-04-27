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

