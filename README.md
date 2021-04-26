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
- [ ] 用 Docker 运行
- [ ] 进行模拟交易
- [ ] 对多个强化学习算法进行测试

### Reference

[FinRL](https://github.com/AI4Finance-LLC/FinRL)