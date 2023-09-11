from mayavi import mlab
import numpy as np

# 加载3D数据文件
data = np.load(r"C:\lfm\code\monai\MSCDA-main\data\dataset_1\DYN\Subject_001\1.npz")['arr_0']

# 创建一个Mayavi场景
mlab.figure()

# 可视化3D数据并启用交互模式
mlab.contour3d(data, colormap="jet")  # 您可以根据需要选择不同的颜色映射

# 显示Mayavi可视化窗口
mlab.show()
