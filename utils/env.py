import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger

# TODO 
# add Readme and description
class Stock_Trading_Env(gym.Env):
    """构建股票交易环境的类，继承自 gym.Env
    Attributes
    ----------
        df : pd.DataFrame
            用于训练的数据集
        stock_dim : int
            交易时所涉及到的股票数量
        hmax : int
            可进行交易的最大数量
        initial_amount : int
            初始资金量
        buy_cost_pct : float
            买股票时的手续费
        sell_cost_pct : float
            卖股票时的手续费
        reward_scaling : float
            reward * reward_scaling 再输入到 model 中
        state_space : int
            状态空间 # TODO
            由四部分组成 : 
                1、今天的资金量
                2、每只股票今天的收盘价
                3、每只股票当前的持仓量
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

    Description
    -----------
    # TODO
    交易环境为：每次从 day 0 训练到 ady -1 ？？？
    为啥 self.cost 一直为 0
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
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.day = day
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name,
        self.mode = mode
        self.iteration = iteration
        self.terminal = False

        # 初始化相关的变量
        self.data = self.df.loc[self.day, :]
        self.multi_stocks = len(self.df.tic.unique()) > 1
        
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
    
    def _sell_stock(self, index, action):

        def _do_sell_normal():
            # 当前股价 > 0 并且 当前有持仓的情况下才可以卖出
            if self.state[index + 1] > 0 and self.state[index + 1 + self.stock_dim] > 0:
                sell_num_shares = min(abs(action), self.state[index + 1 + self.stock_dim])
                # 股价 * 股票数 * (1- 手续费)
                sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct)
                # 更新数据
                self.state[0] += sell_amount
                self.state[index + 1 + self.stock_dim] -= sell_num_shares
                self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                self.trades += 1
            else :
                sell_num_shares = 0
            
            return sell_num_shares
        
        if self.turbulence_threshold is None:
            sell_num_shares = _do_sell_normal()
        else:
            # TODO
            pass
        
        return sell_num_shares

    def _buy_stock(self, index, action):
        
        def _do_buy():
            # 股价 > 0 时执行买操作
            if self.state[index + 1] > 0:
                buy_num_shares = min(self.state[0] // self.state[index + 1], action)
                buy_amount = self.state[index + 1] * buy_num_shares * (1 - self.buy_cost_pct)
                # 更新数据
                self.state[0] -= buy_amount
                self.state[index + 1 + self.stock_dim] += buy_num_shares
                self.cost = self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                self.trades += 1
            else :
                buy_num_shares = 0
            
            return buy_num_shares
        
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass
        
        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, 'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            if self.make_plots:
                self._make_plot()
            
            # 计算最终的资产
            end_total_asset = self.state[0] + \
                sum(np.array(self.state[1: (self.stock_dim + 1)]) * \
                    np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]))

            total_reward = end_total_asset - self.initial_amount

            # 使用 pandas 来存储最终的数据
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            # pct_change；计算改变的百分比
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)

            if df_total_value['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() /\
                    df_total_value['daily_return'].std()
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]

            if self.episode % self.print_verbosity == 0:
                print("天数: {}天, episode: {}".format(self.day, self.episode))
                print("开始时的总资产: {}".format(round(self.asset_memory[0], 2)))
                print("结束时的总资产: {}".format(round(end_total_asset, 2)))
                print("总奖励值: {}".format(round(total_reward, 2)))
                print("总的手续费: {}".format(round(self.cost, 2)))
                print("总的交易次数: {}".format(self.trades))
                if df_total_value['daily_return'].std() != 0:
                    print("Sharpe: {}".format(round(sharpe, 3)))
                print("=============================")
        
            if self.model_name != '' and self.mode != '':
                pass
            
            logger.record("environment/portfolio_value", end_total_asset)
            logger.record("environment/total_reward", total_reward)
            logger.record("environment/total_reward_pct", (total_reward / (end_total_asset - total_reward)) * 100)
            logger.record("environment/total_cost", self.cost)
            logger.record("environment/total_trades", self.trades)
        
            return self.state, self.reward, self.terminal, {}

        else:
            # actions∈[-1, 1], 乘上 hmax 得到实际交易的股票数
            actions = (actions * self.hmax).astype(int)
            
            if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
                actions = np.array([self.hmax * (-1)] * self.stock_dim)
            # 计算初始资产，计算方式：现金 + 持仓的股票市值
            # np.array([]) * np.array([]) 对应的列相乘
            # 如：np.array([1, 2]) * np.array([1, 2]) = np.array([1, 4])
            begin_total_asset = self.state[0] + \
                sum(np.array(self.state[1: (self.stock_dim + 1)]) * \
                    np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]))

            # np.argsort : 按照 index 进行排序
            # actions : np.array([1,2,5,3,10,8])
            # arg_actions : array([0, 1, 3, 2, 5, 4], dtype=int64)
            argsort_actions = np.argsort(actions)
            mid_index = np.where(actions < 0)[0].shape[0]
            sell_index = argsort_actions[: mid_index]
            buy_index = argsort_actions[mid_index:]

            # 执行卖操作
            for index in sell_index:
                # 返回值为实际进行买卖的成交量
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            # 执行买操作
            for index in buy_index:
                # 返回值为实际进行买卖的成交量
                actions[index] = self._buy_stock(index, actions[index])

            # 计算 reward
            end_total_asset = self.state[0] + \
                sum(np.array(self.state[1: (self.stock_dim + 1)]) * \
                    np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]))
            self.reward = end_total_asset - begin_total_asset

            # 添加新数据
            self.actions_memory.append(actions)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.rewards_memory.append(self.reward)
            
            # 更新 day 和 state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                self.turbulence = self.data['turbulence'].value[0]
            self.state = self._update_state()

            self.reward = self.reward * self.reward_scaling
        
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0] +\
                sum(np.array(self.state[1:(self.stock_dim + 1)]) * \
                    np.array(self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]))
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state

    def render(self):
        return self.state

    def _initiate_state(self):
        # 初始化数据
        if self.initial:
            # 多只股票
            if self.multi_stocks:
                state = [self.initial_amount] + \
                        self.data.close.values.tolist() + \
                        [0] * self.stock_dim + \
                        sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
                        # sum([[1, 2], [3, 4]] ,[]) = [1, 2, 3, 4]
            # 单只股票
            else:
                state = [self.initial_amount] + \
                        [self.data.close] + \
                        [0] * self.stock_dim + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
        # 使用过去的数据
        else:
            # 多只股票
            if self.multi_stocks:
                state = [self.previous_state[0]] + \
                        self.data.close.values.tolist() + \
                        self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)] + \
                        sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
            # 单只股票
            else:
                state = [self.previous_state[0]] + \
                        [self.data.close] + \
                        self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)] + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list], [])

        return state

    def _update_state(self):
        if self.multi_stocks:
            # 持仓量和现金数在进行交易的时候已经更新了
            state = [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]) + \
                    sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
        else:
            state = [self.state[0]] + \
                    [self.data.close] + \
                    list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]) + \
                    sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
        
        return state

    def _get_date(self):
        if self.multi_stocks:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        
        return date

    def save_asset_memory(self):
        return pd.DataFrame({
            'date': self.date_memory,
            'account_value': self.asset_memory
        })

    def save_action_memory(self):
        if self.multi_stocks:
            # 让 date 和 close_price 有相同的长度
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({
                'date': date_list, 'actions':action_list
            })
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

if __name__ == "__main__":
    from pull_data import Pull_data
    from preprocessors import FeatureEngineer, split_data
    import config
    import time

    # 拉取数据
    df = Pull_data(config.SSE_50[:2], save_data=False).pull_data()
    df = FeatureEngineer().preprocess_data(df)
    df = split_data(df, '2009-01-01','2019-01-01')
    print(df.head())

    # 处理超参数
    stock_dimension = len(df.tic.unique()) # 2
    state_space = 1 + 2*stock_dimension + \
        len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension # 23
    print("stock_dimension: {}, state_space: {}".format(stock_dimension, state_space))
    env_kwargs = {
        "stock_dim": stock_dimension, 
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "state_space": state_space, 
        "action_space": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST
    }

    # 测试环境
    e_train_gym = Stock_Trading_Env(df = df, **env_kwargs)

    ### 测试一次
    # observation = e_train_gym.reset()
    # print("reset_observation: ", observation)
    # action = e_train_gym.action_space.sample()
    # print("action: ", action)
    # observation_later, reward, done, _ = e_train_gym.step(action)
    # print("observation_later: ", observation_later)
    # print("reward: {}, done: {}".format(reward, done))

    ### 多次测试
    observation = e_train_gym.reset()       #初始化环境，observation为环境状态
    count = 0
    for t in range(10):
        action = e_train_gym.action_space.sample()  #随机采样动作
        observation, reward, done, info = e_train_gym.step(action)  #与环境交互，获得下一个state的值
        if done:             
            break
        count+=1
        time.sleep(0.2)      #每次等待 0.2s
    print("observation: ", observation)
    print("reward: {}, done: {}".format(reward, done))