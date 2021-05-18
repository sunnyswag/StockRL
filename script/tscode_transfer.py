import re
import os

def creat_dir(txt_transfer_dir: str) -> None:
    """创建转换后的目录"""
    if not os.path.exists(txt_transfer_dir):
        os.makedirs(txt_transfer_dir)
        print("创建 {} 目录成功!".format(txt_transfer_dir))
    else:
        print(" {} 目录已存在!".format(txt_transfer_dir))

def transfer(
        txt_dir: str, transfer_dir: str, file_name: str
    ) -> None:
    """将原始股票代码批量修改成可以被 Tushare API 识别的股票代码

    Attributes
        txt_dir: 原始文件存放的目录
        transfer_dir: 修改后的文件存放的目录
        file_name: 原始文件的文件名
    """

    transferd = []
    file_name_sub = re.findall(r"(.+?).txt", file_name)[0]
    # 第一行添加该指数的标识
    transferd.append(str(file_name_sub) + " = [") 

    read_dir = os.path.join(txt_dir, file_name)
    write_dir = os.path.join(transfer_dir, file_name_sub + "_transferred.txt")
    
    with open(read_dir, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line[0] == '6':
                transferd.append("    \"" + str(line) + ".SH" + "\"")
            else:
                transferd.append("    \"" + str(line) + ".SZ" + "\"")

    with open(write_dir, "w") as f:
        # 第一行和最后一行单独处理
        f.write(transferd[0])
        f.write('\n')

        for i in range(1, len(transferd) - 1):
            f.write(transferd[i] + ',')
            f.write('\n')
        
        f.write(transferd[-1] + "]")

if __name__ == "__main__":
    txt_dir = "index_txt"
    txt_transfer_dir = txt_dir + "_transfer"

    creat_dir(txt_transfer_dir)

    txt_list = os.listdir(txt_dir)
    for txt in txt_list:
        transfer(txt_dir=txt_dir, transfer_dir=txt_transfer_dir, file_name=txt)