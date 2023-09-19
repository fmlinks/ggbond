from PIL import Image

# 指定输入图像文件夹和输出图像文件夹的路径
input_folder = "D:/B/data/DomainAdaptation2D/data/3DRA/val/image"
output_folder = "D:/B/data/DomainAdaptation2D/data/3DRA/val/image"

# 为新的大小创建一个矩形框
new_width = 320
new_height = 384

# 遍历输入文件夹中的图像文件
import os
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 打开图像文件
        img = Image.open(os.path.join(input_folder, filename))

        # 获取图像的宽度和高度
        width, height = img.size

        # 计算剪裁框的左上角坐标
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        # 执行中心剪裁
        img_cropped = img.crop((left, top, right, bottom))

        # 保存剪裁后的图像到输出文件夹
        img_cropped.save(os.path.join(output_folder, filename))

print("中心剪裁完成")
