import os
import numpy as np
import pydicom
import nibabel as nib

# 定义图像和标签文件的路径
image_file_path = "D:/B/Paper/domain/table/SOTA/MRA-SMILE/image/sub020.nii.gz"
label_file_path = "D:/B/Paper/domain/table/SOTA/MRA-SMILE/label/sub020.nii.gz"

# 加载图像和标签
image_data = nib.load(image_file_path).get_fdata()
label_data = nib.load(label_file_path).get_fdata()

# 归一化图像数据到0到1之间，并将数据类型更改为float32
image_data = image_data.astype(np.float32)
label_data = label_data.astype(np.float32)
image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))


# 创建目标文件夹
output_folder = "C:/lfm/code/monai/MSCDA-main/test/data/dataset_3/DYN/Subject_020"
os.makedirs(output_folder, exist_ok=True)

# 逐个切片保存为.npz文件
for i in range(image_data.shape[2]):

    # 获取当前切片
    image_slice = image_data[:, :, i]
    label_slice = label_data[:, :, i]

    # 创建包含切片的字典
    data_dict = {'arr_0': image_slice, 'arr_1': label_slice}

    # 生成带有零填充的文件名
    file_name = f"{i + 1:03d}.npz"

    # 保存为.npz文件
    npz_file_path = os.path.join(output_folder, file_name)
    np.savez(npz_file_path, **data_dict)

print("切片提取并保存完成。")
