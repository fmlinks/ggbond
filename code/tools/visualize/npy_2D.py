import numpy as np
import matplotlib.pyplot as plt

# 创建一个2D的NumPy数组数据（示例数据）
# data = np.random.rand(10, 10)  # 这里使用随机数据，您可以替换为您的实际数据
data = np.load(r"C:\lfm\code\monai\MSCDA-main\data\dataset_2\DYN\Subject_001\1.npz")['arr_0']

# 使用Matplotlib绘制热图
plt.imshow(data, cmap='viridis')  # 使用'viridis'颜色映射，您可以选择其他颜色映射
plt.colorbar()  # 添加颜色条

# 显示图形
plt.show()
