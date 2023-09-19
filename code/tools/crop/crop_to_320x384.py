from PIL import Image
import os

# 指定输入和输出文件夹
input_folder = 'D:/B/data/domainAdaptation2D/data/MRA/train/labela/'
output_folder = 'D:/B/data/domainAdaptation2D/data/MRA/train/cropped_image/'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有文件
image_files = os.listdir(input_folder)

# 定义目标剪裁尺寸
target_width = 320
target_height = 384

# 遍历每个图像文件并进行中心剪裁
for image_file in image_files:
    # 构建输入和输出文件的完整路径
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)

    # 打开图像文件
    img = Image.open(input_path)

    # 获取图像的原始尺寸
    width, height = img.size

    # 计算剪裁框的位置
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2

    # 进行中心剪裁
    cropped_img = img.crop((left, top, right, bottom))

    # 保存剪裁后的图像
    cropped_img.save(output_path)

print("中心剪裁完成。")
