from utils.pull_data import Pull_data
from utils.preprocessors import FeatureEngineer, split_data
from utils.models import DRL_Agent
from utils.env import Stock_Trading_Env
from utils import config
import time
from utils.backtest import backtest_stats, backtest_plot, get_baseline

# 拉取数据
# df = Pull_data(config.SSE_50[:2], save_data=False).pull_data()
# df = FeatureEngineer().preprocess_data(df)
# df = split_data(df, '2009-01-01','2019-01-01')
# print(df.head())

# # 处理超参数
# stock_dimension = len(df.tic.unique()) # 2
# state_space = 1 + 2*stock_dimension + \
#     len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension # 23
# print("stock_dimension: {}, state_space: {}".format(stock_dimension, state_space))
# env_kwargs = {
#     "stock_dim": stock_dimension, 
#     "hmax": 100, 
#     "initial_amount": 1e6, 
#     "buy_cost_pct": 0.001,
#     "sell_cost_pct": 0.001,
#     "reward_scaling": 1e-4,
#     "state_space": state_space, 
#     "action_space": stock_dimension, 
#     "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST
# }

# # 测试环境
# e_train_gym = Stock_Trading_Env(df = df, **env_kwargs)

### 测试一次
# observation = e_train_gym.reset()
# print("reset_observation: ", observation)
# action = e_train_gym.action_space.sample()
# print("action: ", action)
# observation_later, reward, done, _ = e_train_gym.step(action)
# print("observation_later: ", observation_later)
# print("reward: {}, done: {}".format(reward, done))

### 多次测试
# observation = e_train_gym.reset()       #初始化环境，observation为环境状态
# count = 0
# for t in range(10):
#     action = e_train_gym.action_space.sample()  #随机采样动作
#     observation, reward, done, info = e_train_gym.step(action)  #与环境交互，获得下一个state的值
#     if done:             
#         break
#     count+=1
#     time.sleep(0.2)      #每次等待 0.2s
# print("observation: ", observation)
# print("reward: {}, done: {}".format(reward, done))

# 测试 model
# env_train, _ = e_train_gym.get_sb_env()
# print(type(env_train))

# agent = DRL_Agent(env= env_train)
# SAC_PARAMS = {
#     "batch_size": 128,
#     "buffer_size": 1000000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1"
# }
# model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

baseline_df = get_baseline(config.SSE_50_INDEX, 
              start="20190101",
              end="20210101")

baseline_stats = backtest_stats(baseline_df, value_col_name='close')
print(baseline_df.head())