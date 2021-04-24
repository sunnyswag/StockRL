"""
    Save_Data : 是否保存数据
    Dir_Data : 存放数据的目录
    Tushare_Tocken : 使用 Tushare API 下载文件时所需要用到的 tocken
    Start_Date : 数据开始下载的日期
    End_Date : 数据截止下载的日期
    TECHNICAL_INDICATORS_LIST : 技术指标列表

    SSE_50 : 上证 50 成分股
    CSI_300 : 沪深 300 成分股
"""

Save_Data = False # 是否保存数据
Dir_Data = "datasets" # 存放数据的目录

# 使用 Tushare API 下载文件时所需要用到的 tocken
Tushare_Tocken = "c576df5b626df4f37c30bae84520d70c7945a394d7ee274ef2685444"

# 数据开始下载和截止下载的日期
Start_Date = '20090101'
End_Date = '20210101'

Start_Trade_Date = "2009-01-01"
End_Trade_Date = "2019-01-01"
End_Test_Date = "2021-01-01"

# 技术指标列表
TECHNICAL_INDICATORS_LIST = [
    "boll_ub","boll_lb","rsi_30", "cci_30", "dx_30", \
    "macd","volume_20_sma","volume_60_sma","volume_120_sma","close_20_sma","close_60_sma","close_120_sma"
]

# 模型的超参数
A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.01, 
    "learning_rate": 0.0007
    }
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64
    }
DDPG_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 50000, 
    "learning_rate": 0.001
    }
TD3_PARAMS = {
    "batch_size": 100, 
    "buffer_size": 1000000, 
    "learning_rate": 0.001
    }
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 2000,
    "ent_coef": "auto_0.1"
}

TENSORBOARD_LOG_DIR = f"tensorboard_log"

# 上证 50 指数和成分股
SSE_50_INDEX = ["000016.SH"]
SSE_50 = [
    "600000.SH",
    "600009.SH",
    "600016.SH",
    "600028.SH",
    "600030.SH",
    "600031.SH",
    "600036.SH",
    "600048.SH",
    "600050.SH",
    "600104.SH",
    "600196.SH",
    "600276.SH",
    "600309.SH",
    "600519.SH",
    "600547.SH",
    "600570.SH",
    "600585.SH",
    "600588.SH",
    "600690.SH",
    "600703.SH",
    "600745.SH",
    "600837.SH",
    "600887.SH",
    "600918.SH",
    "601012.SH",
    "601066.SH",
    "601088.SH",
    "601138.SH",
    "601166.SH",
    "601186.SH",
    "601211.SH",
    "601236.SH",
    "601288.SH",
    "601318.SH",
    "601319.SH",
    "601336.SH",
    "601398.SH",
    "601601.SH",
    "601628.SH",
    "601668.SH",
    "601688.SH",
    "601816.SH",
    "601818.SH",
    "601857.SH",
    "601888.SH",
    "603160.SH",
    "603259.SH",
    "603288.SH",
    "603501.SH",
    "603986.SH"]

# 沪深 300 成分股
CSI_300 = [
    "600000.SH",
    "600004.SH",
    "600009.SH",
    "600010.SH",
    "600011.SH",
    "600015.SH",
    "600016.SH",
    "600018.SH",
    "600019.SH",
    "600025.SH",
    "600027.SH",
    "600028.SH",
    "600029.SH",
    "600030.SH",
    "600031.SH",
    "600036.SH",
    "600048.SH",
    "600050.SH",
    "600061.SH",
    "600066.SH",
    "600068.SH",
    "600085.SH",
    "600104.SH",
    "600109.SH",
    "600111.SH",
    "600115.SH",
    "600118.SH",
    "600150.SH",
    "600161.SH",
    "600176.SH",
    "600177.SH",
    "600183.SH",
    "600196.SH",
    "600208.SH",
    "600233.SH",
    "600271.SH",
    "600276.SH",
    "600297.SH",
    "600299.SH",
    "600309.SH",
    "600332.SH",
    "600340.SH",
    "600346.SH",
    "600352.SH",
    "600362.SH",
    "600369.SH",
    "600383.SH",
    "600390.SH",
    "600406.SH",
    "600436.SH",
    "600438.SH",
    "600482.SH",
    "600487.SH",
    "600489.SH",
    "600498.SH",
    "600519.SH",
    "600522.SH",
    "600547.SH",
    "600570.SH",
    "600584.SH",
    "600585.SH",
    "600588.SH",
    "600600.SH",
    "600606.SH",
    "600637.SH",
    "600655.SH",
    "600660.SH",
    "600690.SH",
    "600703.SH",
    "600705.SH",
    "600741.SH",
    "600745.SH",
    "600760.SH",
    "600763.SH",
    "600795.SH",
    "600809.SH",
    "600837.SH",
    "600845.SH",
    "600848.SH",
    "600872.SH",
    "600886.SH",
    "600887.SH",
    "600893.SH",
    "600900.SH",
    "600918.SH",
    "600919.SH",
    "600926.SH",
    "600958.SH",
    "600989.SH",
    "600998.SH",
    "600999.SH",
    "601006.SH",
    "601009.SH",
    "601012.SH",
    "601021.SH",
    "601066.SH",
    "601077.SH",
    "601088.SH",
    "601100.SH",
    "601108.SH",
    "601111.SH",
    "601117.SH",
    "601138.SH",
    "601155.SH",
    "601162.SH",
    "601166.SH",
    "601169.SH",
    "601186.SH",
    "601198.SH",
    "601211.SH",
    "601216.SH",
    "601225.SH",
    "601229.SH",
    "601231.SH",
    "601236.SH",
    "601238.SH",
    "601288.SH",
    "601318.SH",
    "601319.SH",
    "601328.SH",
    "601336.SH",
    "601360.SH",
    "601377.SH",
    "601390.SH",
    "601398.SH",
    "601555.SH",
    "601577.SH",
    "601600.SH",
    "601601.SH",
    "601607.SH",
    "601618.SH",
    "601628.SH",
    "601633.SH",
    "601658.SH",
    "601668.SH",
    "601669.SH",
    "601688.SH",
    "601696.SH",
    "601698.SH",
    "601727.SH",
    "601766.SH",
    "601788.SH",
    "601800.SH",
    "601808.SH",
    "601816.SH",
    "601818.SH",
    "601838.SH",
    "601857.SH",
    "601872.SH",
    "601877.SH",
    "601878.SH",
    "601881.SH",
    "601888.SH",
    "601899.SH",
    "601901.SH",
    "601916.SH",
    "601919.SH",
    "601933.SH",
    "601939.SH",
    "601985.SH",
    "601988.SH",
    "601989.SH",
    "601990.SH",
    "601998.SH",
    "603019.SH",
    "603087.SH",
    "603156.SH",
    "603160.SH",
    "603195.SH",
    "603259.SH",
    "603288.SH",
    "603369.SH",
    "603392.SH",
    "603501.SH",
    "603658.SH",
    "603799.SH",
    "603833.SH",
    "603899.SH",
    "603986.SH",
    "603993.SH",
    "688008.SH",
    "688009.SH",
    "688012.SH",
    "688036.SH",
    "000001.SZ",
    "000002.SZ",
    "000063.SZ",
    "000066.SZ",
    "000069.SZ",
    "000100.SZ",
    "000157.SZ",
    "000166.SZ",
    "000333.SZ",
    "000338.SZ",
    "000425.SZ",
    "000538.SZ",
    "000568.SZ",
    "000596.SZ",
    "000625.SZ",
    "000627.SZ",
    "000651.SZ",
    "000656.SZ",
    "000661.SZ",
    "000671.SZ",
    "000703.SZ",
    "000708.SZ",
    "000723.SZ",
    "000725.SZ",
    "000728.SZ",
    "000768.SZ",
    "000776.SZ",
    "000783.SZ",
    "000786.SZ",
    "000858.SZ",
    "000860.SZ",
    "000876.SZ",
    "000895.SZ",
    "000938.SZ",
    "000961.SZ",
    "000963.SZ",
    "000977.SZ",
    "001979.SZ",
    "002001.SZ",
    "002007.SZ",
    "002008.SZ",
    "002024.SZ",
    "002027.SZ",
    "002032.SZ",
    "002044.SZ",
    "002049.SZ",
    "002050.SZ",
    "002120.SZ",
    "002129.SZ",
    "002142.SZ",
    "002146.SZ",
    "002153.SZ",
    "002157.SZ",
    "002179.SZ",
    "002202.SZ",
    "002230.SZ",
    "002236.SZ",
    "002241.SZ",
    "002252.SZ",
    "002271.SZ",
    "002304.SZ",
    "002311.SZ",
    "002352.SZ",
    "002371.SZ",
    "002384.SZ",
    "002410.SZ",
    "002414.SZ",
    "002415.SZ",
    "002422.SZ",
    "002456.SZ",
    "002460.SZ",
    "002463.SZ",
    "002475.SZ",
    "002493.SZ",
    "002508.SZ",
    "002555.SZ",
    "002558.SZ",
    "002594.SZ",
    "002600.SZ",
    "002601.SZ",
    "002602.SZ",
    "002607.SZ",
    "002624.SZ",
    "002673.SZ",
    "002714.SZ",
    "002736.SZ",
    "002739.SZ",
    "002773.SZ",
    "002812.SZ",
    "002821.SZ",
    "002841.SZ",
    "002916.SZ",
    "002938.SZ",
    "002939.SZ",
    "002945.SZ",
    "002958.SZ",
    "003816.SZ",
    "300003.SZ",
    "300014.SZ",
    "300015.SZ",
    "300033.SZ",
    "300059.SZ",
    "300122.SZ",
    "300124.SZ",
    "300136.SZ",
    "300142.SZ",
    "300144.SZ",
    "300347.SZ",
    "300408.SZ",
    "300413.SZ",
    "300433.SZ",
    "300498.SZ",
    "300529.SZ",
    "300601.SZ",
    "300628.SZ",
    "300676.SZ"]