import os

# 指定标签文件夹路径
label_folder = r'D:\B\data\DomainAdaptation2D\data\3DRA\val\label'

# 遍历标签文件夹中的所有文件
for filename in os.listdir(label_folder):
    if filename.endswith('.png'):
        # 构建原始文件路径
        old_filepath = os.path.join(label_folder, filename)

        # 从文件名中移除"_gtFine_labelTrainIds"部分
        new_filename = filename.replace("_labelTrainIds", "")

        # 构建新的文件路径
        new_filepath = os.path.join(label_folder, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)

print("文件重命名完成")
