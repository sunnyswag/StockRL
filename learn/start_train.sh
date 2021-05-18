mkdir nohup
nohup python -u ./trainer.py -m 'a2c' -tts 200000 >./nohup/A2C.log 2>&1 &
# nohup python -u ./trainer.py -m 'ddpg' -tts 200000 >./nohup/DDPG.log 2>&1 &
# nohup python -u ./trainer.py -m 'ppo' -tts 200000 >./nohup/PPO.log 2>&1 &
nohup python -u ./trainer.py -m 'td3' -tts 200000 >./nohup/TD3.log 2>&1 &
# nohup python -u ./trainer.py -m 'sac' -tts 200000 >./nohup/SAC.log 2>&1 &
