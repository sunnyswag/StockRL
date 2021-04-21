import pandas as pd
import numpy as np
import time
import gym

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import config
from preprocessors import split_data
from env import Stock_Trading_Env

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
MODEL_KWARGS = {x: config.__dict__["{}_PARAMS".format(x.upper())] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise
}

class DRL_Agent():
    """

    """

    @staticmethod
    def DRL_prediction(model, environment):
        test_env, test_obs = environment.get_sb_env()

        account_memory = []
        actions_memory = []
        test_env.reset()

        len_environment = len(environment.df.index.unique())
        for i in range(len_environment):
            action, _states = model.predict(test_obs)
            test_obs, _, dones, _ = test_env.step(action)
            if i == (len_environment - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("完成!")
                break
        return account_memory[0], actions_memory[0]

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy = "MlpPolicy",
        policy_kwargs = None,
        model_kwargs = None,
        verbose = 1
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        
        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]
        
        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)

        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log="{}/{}".format(config.TENSORBOARD_LOG_DIR, model_name),
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs
        )
        
        return model

    def train_model(self, model, tb_log_name, total_timesteps = 5000):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        return model
    
# TODO
class DRLEnsembleAgent:

    @staticmethod
    def get_model():
        pass

    @staticmethod
    def train_model():
        pass

    @staticmethod
    def get_validation_sharpe():
        pass

    def __init__(self):
        pass

    def DRL_validation(self):
        pass

    def DRL_prediction(self):
        pass

    def run_ensemble_strategy(self):
        pass

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
        "initial_amount": 1e6, 
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

    # 测试 model
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    agent = DRL_Agent(env= env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1"
    }
    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    trained_sac = agent.train_model(
        model=model_sac,
        tb_log_name='sac', 
        total_timesteps= 20000
    )