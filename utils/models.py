from typing import Any
import pandas as pd
import numpy as np
import time

from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from utils import config
from utils.preprocessors import split_data
from utils.env import StockLearningEnv

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
MODEL_KWARGS = {x: config.__dict__["{}_PARAMS".format(x.upper())] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise
}

class DRL_Agent():
    """强化学习交易智能体

    Attributes:
        env: 强化学习环境
    """

    @staticmethod
    def DRL_prediction(
        model: Any, environment: Any
        ) -> pd.DataFrame:
        """回测函数"""
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
                print("回测完成!")
                break
        return account_memory[0], actions_memory[0]

    def __init__(self, env: Any) -> None:
        self.env = env

    def get_model(
        self,
        model_name: str,
        policy: str = "MlpPolicy",
        policy_kwargs: dict = None,
        model_kwargs: dict = None,
        verbose: int = 1
    ) -> Any:
        """根据超参数生成模型"""
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

    def train_model(
        self, model: Any, tb_log_name: str, total_timesteps: int = 5000
        ) -> Any:
        """训练模型"""
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        return model

if __name__ == "__main__":
    from pull_data import Pull_data
    from preprocessors import FeatureEngineer, split_data
    from utils import config
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
    e_train_gym = StockLearningEnv(df = df, **env_kwargs)

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
        total_timesteps= 50000
    )