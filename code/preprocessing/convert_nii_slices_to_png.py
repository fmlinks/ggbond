import os
import nibabel as nib
import numpy as np

# 输入和输出目录路径
input_dir = "D:/B/data/SMILE-UHURA/train/label"
output_dir = "D:/B/data/SMILE-UHURA/data2d/train/label"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取输入目录下所有.nii.gz文件的列表
# nii_files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]
nii_files = [f for f in os.listdir(input_dir) if f.endswith(".nii")]

# 遍历每个.nii.gz文件
for nii_file in nii_files:
    # 构建输入文件的完整路径
    input_path = os.path.join(input_dir, nii_file)

    # 使用nibabel库加载.nii.gz文件
    nii_data = nib.load(input_path)

    # 获取MRI数据的切片
    mri_slices = nii_data.get_fdata()
    mri_slices = np.transpose(mri_slices, (2, 1, 0))  # 转置切片以匹配目标大小

    # 去除文件名中的扩展名部分
    # file_name_without_extension = os.path.splitext(os.path.splitext(nii_file)[0])[0]
    file_name_without_extension = os.path.splitext(nii_file)[0]

    # 遍历每个切片并保存为NIfTI文件
    for i, slice in enumerate(mri_slices):
        # 将切片大小调整为 [640, 480] 并进行中心剪裁
        h, w = slice.shape
        top = (h - 640) // 2
        left = (w - 480) // 2
        resized_slice = slice[top:top+640, left:left+480]  # 调整大小为 [640, 480]

        save_slice = np.transpose(resized_slice, (1, 0))

        # 创建NIfTI图像对象
        nii_image = nib.Nifti1Image(save_slice, nii_data.affine)

        # 生成NIfTI文件名，格式为文件名（去除扩展名部分）+切片序号
        filename = f"{file_name_without_extension}_{i:04d}.nii.gz"

        # 构建输出文件的完整路径
        output_path = os.path.join(output_dir, filename)

        # 保存切片为NIfTI文件
        nib.save(nii_image, output_path)

print(f"已将所有.nii.gz文件的切片保存为NIfTI格式到目录：{output_dir}")
