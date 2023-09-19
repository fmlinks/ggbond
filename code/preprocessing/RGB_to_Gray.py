from PIL import Image
import os

# 指定输入和输出文件夹
input_folder = 'D:/B/data/DomainAdaptation2D/data/MRA/val/image/'
output_folder = 'D:/B/data/DomainAdaptation2D/data/MRA/val/image/'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有文件
image_files = os.listdir(input_folder)

# 遍历每个图像文件并转换为灰度图像
for image_file in image_files:
    # 构建输入和输出文件的完整路径
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)

    # 打开图像文件并转换为灰度图像
    img = Image.open(input_path).convert('L')

    # 保存灰度图像
    img.save(output_path)

print("灰度图像转换完成。")
