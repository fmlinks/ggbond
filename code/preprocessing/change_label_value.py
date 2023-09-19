import os
import cv2

# 指定标签图像文件夹路径
label_folder = r'D:\B\data\DomainAdaptation2D\data\MRA\train\label'

# 遍历标签图像文件夹
for filename in os.listdir(label_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        # 读取标签图像
        label_img = cv2.imread(os.path.join(label_folder, filename), cv2.IMREAD_GRAYSCALE)

        # 将所有11的像素值替换为1
        label_img[label_img == 11] = 1

        # 保存替换后的标签图像
        cv2.imwrite(os.path.join(label_folder, filename), label_img)

print("标签值替换完成")
