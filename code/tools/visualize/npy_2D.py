import numpy as np
import matplotlib.pyplot as plt

# 创建一个2D的NumPy数组数据（示例数据）
# data = np.random.rand(10, 10)  # 这里使用随机数据，您可以替换为您的实际数据
data = np.load(r"C:\lfm\code\monai\MSCDA-main\test\data\dataset_3\DYN\Subject_017\111.npz")['arr_1']

# org_data = np.load(r"C:\lfm\code\monai\MSCDA-main\test\data\dataset_2\DYN\Subject_007\1.npz")
# print(org_data.shape, 'aaaaaa')

print(np.max(data), np.min(data), data.dtype, data.shape)

# 使用Matplotlib绘制热图
plt.imshow(data, cmap='viridis')  # 使用'viridis'颜色映射，您可以选择其他颜色映射
plt.colorbar()  # 添加颜色条

# 显示图形
plt.show()
