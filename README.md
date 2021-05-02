# RL_in_Finance
使用强化学习，构建自动交易的金融交易系统，目的是减少回撤

### 快速开始

方式一：

```shell
# excute in shell
git clone https://github.com/sunnyswag/RL_in_Finance.git
git pip install -r requirements.txt
```

方式二：

​	打开 [RL_in_Finance_Test.ipynb](./RL_in_Finance_Test.ipynb) 并运行(推荐使用 colab)

### 环境描述

**state_space 五部分组成 :** 

1. 当前的资金量

2. 每只股票当前的收盘价

3. 每只股票当前的持仓量

4. 股票数 * 技术指标数

5. 成交量

**reward 的计算方式：**

* reward 交易前的总资产-当天交易后的总资产 = 当天交易的手续费

* TODO：待改进

**action_space 的空间：**

* actions ∈[-100, 100]

* 正数表示买入，负数表示卖出，0表示不进行买入卖出操作

* 绝对值表示买入卖出的数量

### TODO

- [x] 改进 reward 的计算机制
- [x] reset 在任意位置
- [x] 解决回测中的 0 值问题
- [x] 更换技术指标，在 state 中增加成交量这个状态
- [x] 解决为啥 finrl 在构建 Dockerfile 的时候使用的路径为 docker/ 而不是 .
- [x] 用 Docker 运行
- [x] 解决训练不充分时，会进行超买超卖导致预测时会 terminal 的情况
- [x] 重新设计 reward function，重新设计的 reward function 有两个重点：强调绝对收益和控制收益的回撤
- [ ] 进行模拟交易
- [ ] 对多个强化学习算法进行测试

### LOGO
1. n_cores = 24 and n_cores = 44 都未能运行成功
2. 得训练大概 100w num_timesteps 才差不多能学会, 训练 5w 大概 1个小时，训练 100 得 20 h 
3. 是通过 logger 传数据到 model 的 replay_buffer 中吗？
4. 目测得跑 2 天了

### Reference

[FinRL](https://github.com/AI4Finance-LLC/FinRL)