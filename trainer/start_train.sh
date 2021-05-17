mkdir nohup
nohup python -u RL_in_Finance_DDPG_Docker.py >./nohup/DDPG.log 2>&1 &
nohup python -u RL_in_Finance_A2C_Docker.py >./nohup/A2C.log 2>&1 &
#nohup python -u RL_in_Finance_PPO_Docker.py >./nohup/PPO.log 2>&1 &
#nohup python -u RL_in_Finance_TD3_Docker.py >./nohup/TD3.log 2>&1 &
#nohup python -u RL_in_Finance_SAC_Docker.py >./nohup/SAC.log 2>&1 &
