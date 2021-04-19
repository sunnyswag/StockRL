import re
import os
import sys

# 定义目录名
path = "test"
txt_path = "Index_TXT"
txt_transfer_path = txt_path + "_Transfer"
os.chdir(path)

# 创建转换后的目录
if not os.path.exists(txt_transfer_path):
    os.makedirs(txt_transfer_path)
    print("创建 {} 目录成功!".format(txt_transfer_path))
else:
    print(" {} 目录已存在!".format(txt_transfer_path))

def transfer(txt_dir, transfer_dir, file_name):
    """使用 transfer 函数将原始股票代码批量修改成可以被 Tushare API 识别的股票代码
    Attributes
    ----------
        txt_dir : str
            原始文件存放的目录
        transfer_dir : str
            修改后的文件存放的目录
        file_name : str
            原始文件的文件名
    
    Return
    ------
        None
    """
    test = []
    file_name_sub = re.findall(r"(.+?).txt", file_name)[0]
    test.append("    \"" + str(file_name_sub) + "\":[") # 第一行添加属于某个指数的标识, 生成为 dic 数据类型

    cur_dir = os.path.join(txt_dir, file_name)
    new_dir = os.path.join(transfer_dir, file_name_sub + "_Transferred.txt")
    with open(cur_dir, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line[0] == '6':
                test.append("    \"" + str(line) + ".SH" + "\"")
            else:
                test.append("    \"" + str(line) + ".SZ" + "\"")

    with open(new_dir, "w") as f:
        # 第一行和最后一行单独处理
        f.write(test[0])
        f.write('\n')

        for i in range(1, len(test) - 1):
            f.write(test[i] + ',')
            f.write('\n')
        
        f.write(test[-1] + "]")

txt_list = os.listdir(txt_path)
for txt in txt_list:
    transfer(txt_dir=txt_path, transfer_dir=txt_transfer_path, file_name=txt)

# cwd = os.getcwd()
# files = os.listdir(cwd)
# print(cwd, files)