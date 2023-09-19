import os
import cv2

# 指定图像文件夹路径
image_folder = r'D:\B\data\DomainAdaptation2D\data\MRA\val\image'

# 遍历文件夹中的图像文件
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        # 读取图像
        img = cv2.imread(os.path.join(image_folder, filename))

        # 进行0轴（垂直翻转）
        img_flip_0 = cv2.flip(img, 0)
        new_filename_0 = os.path.splitext(filename)[0] + '_flip_0' + os.path.splitext(filename)[1]
        cv2.imwrite(os.path.join(image_folder, new_filename_0), img_flip_0)

        # 进行1轴（水平翻转）
        img_flip_1 = cv2.flip(img, 1)
        new_filename_1 = os.path.splitext(filename)[0] + '_flip_1' + os.path.splitext(filename)[1]
        cv2.imwrite(os.path.join(image_folder, new_filename_1), img_flip_1)

        # 进行0轴和1轴的翻转
        img_flip_both = cv2.flip(img, -1)
        new_filename_both = os.path.splitext(filename)[0] + '_flip_2' + os.path.splitext(filename)[1]
        cv2.imwrite(os.path.join(image_folder, new_filename_both), img_flip_both)

print("数据增强完成")
