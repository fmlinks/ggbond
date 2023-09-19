import os
import nibabel as nib
import numpy as np

# 输入目录路径
input_dir = "D:/B/data/CROWN/data"

# 输出目录路径
output_dir = "D:/B/data/CROWN/data_padded_cropped"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历子文件夹
subfolders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

for subfolder in subfolders:
    # 构建NIfTI文件的完整路径
    nii_file_path = os.path.join(input_dir, subfolder, "3D_TOF_MRA.nii.gz")

    # 使用nibabel库加载NIfTI文件
    nii_data = nib.load(nii_file_path)

    # 获取MRI数据
    mri_data = nii_data.get_fdata()

    # 创建 [640, 480] 的零矩阵
    padded_cropped_data = np.zeros((640, 480))

    # 计算裁剪的起始和结束位置
    h, w = mri_data.shape
    top = (h - 640) // 2
    left = (w - 480) // 2
    bottom = top + 640
    right = left + 480

    # 复制数据到中心区域
    padded_cropped_data = mri_data[top:bottom, left:right]

    # 创建新的NIfTI图像
    nii_image = nib.Nifti1Image(padded_cropped_data, nii_data.affine)

    # 构建输出文件的完整路径
    output_subfolder = os.path.join(output_dir, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)
    output_file_path = os.path.join(output_subfolder, "3D_TOF_MRA_padded_cropped.nii.gz")

    # 保存新的NIfTI文件
    nib.save(nii_image, output_file_path)

print("已将所有数据进行补零和裁剪，保存到目录：", output_dir)
