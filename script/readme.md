### Script 文件夹的作用

将股票代码修改成可以被 Tushare API 识别的股票代码

### 运行结果

from

> 600000
> 600004
> 600009
> ...

to

> CSI_300 = [
>     "600000.SH",
>     "600004.SH",
>     "600009.SH",
> ...

### 使用方法

1. 在 [./index_txt](./index_txt) 中放入需要转换的文件

2. 运行 [./tscode_transfer.py](./tscode_transfer.py)

   ```shell
   python ./tscode_transfer.py
   ```

3. 在 [./index_txt_transfer](./index_txt_transfer) 中查看转换结果

​	