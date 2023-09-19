from PIL import Image

# 指定图像文件路径
image_path = "D:/B/data/DomainAdaptation2D/data/3DRA/train/image/01001.png"

# 打开图像
img = Image.open(image_path)

# 获取图像的形状
img_shape = img.size  # 形状以 (宽度, 高度) 的形式返回

print("图像形状:", img_shape)
