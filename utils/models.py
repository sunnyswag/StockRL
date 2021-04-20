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
            test_obs, rewards, dones, _ = test_env.step(action)
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
    pass
    # TODO