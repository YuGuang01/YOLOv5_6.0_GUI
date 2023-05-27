"""
作者：jason
编写时间：2023/4/11 13:12
其他：pytorch 测试
"""
import torch

# 返回当前设备索引
print("返回当前设备索引")
print(torch.cuda.current_device())

# 返回GPU的数量
print("返回GPU的数量")
print(torch.cuda.device_count())

# 返回gpu名字，设备索引默认从0开始
print("返回gpu名字，设备索引默认从0开始")
print(torch.cuda.get_device_name(0))

# cuda是否可用
print("cuda是否可用")
print(torch.cuda.is_available())

# 查看pytorch 版本
print("查看pytorch 版本")
print(torch.__version__)

# 查看pytorch对应的cuda 版本
print("查看pytorch对应的cuda 版本")
print(torch.version.cuda)

# 判断pytorch是否支持GPU加速
print("判断pytorch是否支持GPU加速")
print(torch.cuda.is_available())

# 使用的是GPU的，会输出：cuda:0
# 使用的不是GPU的，会输出：cpu
print("使用的是GPU的，会输出：cuda:0", "# 使用的不是GPU的，会输出：cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
