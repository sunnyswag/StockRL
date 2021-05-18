from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import gym
import time
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger

class StockLearningEnv(gym.Env):
    """构建强化学习交易环境

        Attributes
            df: 构建环境时所需要用到的行情数据
            buy_cost_pct: 买股票时的手续费
            sell_cost_pct: 卖股票时的手续费
            date_col_name: 日期列的名称
            hmax: 最大可交易的数量
            print_verbosity: 打印的频率
            initial_amount: 初始资金量
            daily_information_cols: 构建状态时所考虑的列
            cache_indicator_data: 是否把数据放到内存中
            random_start: 是否随机位置开始交易（训练和回测环境分别为True和False）
            patient: 是否在资金不够时不执行交易操作，等到有足够资金时再执行
            currency: 货币单位
    """

    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        df: pd.DataFrame,
        buy_cost_pct: float = 3e-3,
        sell_cost_pct: float = 3e-3,
        date_col_name: str = "date",
        hmax: int = 10,
        print_verbosity: int = 10,
        initial_amount: int = 1e6,
        daily_information_cols: List = ["open", "close", "high", "low", "volume"],
        cache_indicator_data: bool = True,
        random_start: bool = True,
        patient: bool = False,
        currency: str = "￥"
    ) -> None:
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
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
        self.max_total_assets = 0
        if self.cache_indicator_data:
            """cashing data 的结构:
               [[date1], [date2], [date3], ...]
               date1 : [stock1 * cols, stock2 * cols, ...]
            """
            print("加载数据缓存")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("数据缓存成功!")
        
    def seed(self, seed: Any = None) -> None:
        """设置随机种子"""
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)
    
    @property
    def current_step(self) -> int:
        """当前回合的运行步数"""
        return self.date_index - self.starting_point
    
    @property
    def cash_on_hand(self) -> float:
        """当前拥有的现金"""
        return self.state_memory[-1][0]
    
    @property
    def holdings(self) -> List:
        """当前的持仓数据"""
        return self.state_memory[-1][1: len(self.assets) + 1]

    @property
    def closings(self) -> List:
        """每支股票当前的收盘价"""
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def get_date_vector(self, date: int, cols: List = None) -> List:
        """获取 date 那天的行情数据"""
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
    
    def reset(self) -> np.ndarray:
        self.seed()
        self.sum_trades = 0
        self.max_total_assets = self.initial_amount
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

    def log_step(
        self, reason: str, terminal_reward: float=None
        ) -> None:
        """打印"""
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        
        assets = self.account_information["total_assets"][-1]
        tmp_retreat_ptc = assets / self.max_total_assets - 1
        retreat_pct = tmp_retreat_ptc if assets < self.max_total_assets else 0
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount

        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(assets))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{retreat_pct*100:0.2f}%"
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def return_terminal(
        self, reason: str = "Last Date", reward: int = 0
    ) -> Tuple[List, float, bool, dict]:
        """terminal 的时候执行的操作"""
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
            "environment/retreat_proportion",
            self.account_information["total_assets"][-1] / self.max_total_assets
        )

        return state, reward, True, {}

    def log_header(self) -> None:
        """Log 的列名"""
        if not self.printed_header:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
            # 0, 1, 2, ... 是序号
            # 4, 4, 15, ... 是占位格的大小
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD",
                    "GAINLOSS_PCT",
                    "RETREAT_PROPORTION"
                )
            )
            self.printed_header = True

    def get_reward(self) -> float:
        """获取奖励值"""
        if self.current_step == 0:
            return 0
        else:
            assets = self.account_information["total_assets"][-1]
            retreat = 0
            if assets >= self.max_total_assets:
                self.max_total_assets = assets
            else:
                retreat = assets / self.max_total_assets - 1
            reward = assets / self.initial_amount - 1
            reward += retreat
            return reward

    def get_transactions(self, actions: np.ndarray) -> np.ndarray:
        """获取实际交易的股数"""
        self.actions_memory.append(actions)
        actions = actions * self.hmax

        # 收盘价为 0 的不进行交易
        actions = np.where(self.closings > 0, actions, 0)

        # 去除被除数为 0 的警告
        out = np.zeros_like(actions)
        zero_or_not = self.closings != 0
        actions = np.divide(actions, self.closings, out=out, where = zero_or_not)
        
        # 不能卖的比持仓的多
        actions = np.maximum(actions, -np.array(self.holdings))

        # 将 -0 的值全部置为 0
        actions[actions == -0] = 0

        return actions

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List, float, bool, dict]:
        self.sum_trades += np.sum(np.abs(actions))
        self.log_header()
        if(self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        if self.date_index == len(self.dates) - 1:
            return self.return_terminal(reward=self.get_reward())
        else:
            begin_cash = self.cash_on_hand
            assert min(self.holdings) >= 0
            assert_value = np.dot(self.holdings, self.closings)
            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(assert_value)
            self.account_information["total_assets"].append(begin_cash + assert_value)
            reward = self.get_reward()
            self.account_information["reward"].append(reward)

            transactions = self.get_transactions(actions)
            sells = -np.clip(transactions, -np.inf, 0)
            proceeds = np.dot(sells, self.closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds # 计算现金的数量

            buys = np.clip(transactions, 0, np.inf)
            spend = np.dot(buys, self.closings)
            costs += spend * self.buy_cost_pct

            if (spend + costs) > coh: # 如果买不起
                if self.patient:
#                     self.log_step(reason="CASH SHORTAGE")
                    transactions = np.where(transactions > 0, 0, transactions)
                    spend = 0
                    costs = 0
                else:
                    return self.return_terminal(
                        reason="CASH SHORTAGE", reward=self.get_reward()
                    )
            self.transaction_memory.append(transactions)
            assert (spend + costs) <= coh
            coh = coh - spend - costs
            holdings_updated = self.holdings + transactions
            self.date_index += 1
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)
            return state, reward, False, {}

    def get_sb_env(self) -> Tuple[Any, Any]:
        def get_self():
            return deepcopy(self)
        
        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(
        self, n: int = 10
    ) -> Tuple[Any, Any]:
        def get_self():
            return deepcopy(self)
        
        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs
    
    def save_asset_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]):
            ]
            return pd.DataFrame(self.account_information)
    
    def save_action_memory(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]):],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory
                }
            )