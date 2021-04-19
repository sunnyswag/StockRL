import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

class Stock_Trading_Env(gym.Env):
    """构建股票交易环境的类，继承自 gym.Env
    Attributes
    ----------
        df : pd.DataFrame
            用于训练的数据集
        stock_dim : int
            交易时所涉及到的股票数量
        hmax : int

        initial_amount : int
            初始资金量
        buy_cost_pct : float
            买股票时的手续费
        sell_cost_pct : float
            卖股票时的手续费
        reward_scaling : float

        state_space : int
            状态空间 # TODO
            由四部分组成 : 
                1、今天的资金量
                2、今天的收盘价
                3、昨天的收盘价
                4、股票数*技术指标数
        action_space : int
            交易时所涉及到的股票数量
        tech_indicator_list : List
            技术指标列表
        turbulence_threshold : None
        make_plots : boolean
            是否作图
        print_verbosity = 10,
        day : int
            天数，由于日期相同时使用的是相同的 index, 
            所以通过 day 这个索引可以取到天数相同的所有股票行
        initial = True,
        previous_state : list
            上一个状态
        model_name = '',
        mode = '',
        iteration = ''
    Methods
    -------


    """

    def __init__(self,
                df,
                stock_dim,
                hmax,
                initial_amount,
                buy_cost_pct,
                sell_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold = None,
                make_plots = False,
                print_verbosity = 10,
                day = 0,
                initial = True,
                previous_state = [],
                model_name = '',
                mode = '',
                iteration = ''
    ):
        self.df = df,
        self.stock_dim = stock_dim,
        self.hmax = hmax,
        self.initial_amount = initial_amount,
        self.buy_cost_pct = buy_cost_pct,
        self.sell_cost_pct = sell_cost_pct,
        self.reward_scaling = reward_scaling,
        self.state_space = state_space,
        self.action_space = action_space,
        self.tech_indicator_list = tech_indicator_list,
        self.turbulence_threshold = turbulence_threshold,
        self.make_plots = make_plots,
        self.print_verbosity = print_verbosity,
        self.day = day,
        self.initial = initial
        self.previous_state = previous_state,
        self.model_name = model_name, 
        self.mode = mode,
        self.iteration = iteration
        self.terminal = False
        
        # 初始化 Gym enviroment
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.state = self._initiate_state()

        # 初始化变量
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # 初始化存储数据的 List
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self._seed()
        self.data = self.df.loc[self.day, :]
    
    def _sell_stock(self, index, action):
        pass

    def _buy_stock(self, index, action):
        pass

    def _make_plot(self):
        pass

    def step(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def _initiate_state(self):
        pass

    def _update_state(self):
        pass

    def _get_date(self):
        pass

    def save_asset_memory(self):
        pass

    def save_action_memory(self):
        pass

    def _seed(self, seed=None):
        pass

    def get_sb_env(self):
        pass