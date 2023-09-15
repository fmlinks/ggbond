import nibabel as nib

# 读取原始的big_label文件
data_path = r"D:\B\Paper\domain\table\SOTA\MRA-SMI\GT_V3\sub020_Segmentation_7.nii.gz"
# data_path = r"D:\B\Paper\domain\table\SOTA\MRA-SMI\FullySupervised_fullsize\sub020.nii.gz"
big_label_nii = nib.load(data_path)
big_label_data = big_label_nii.get_fdata()

# Crop
normalized_data = big_label_data[48:-48, 192:512, :]

# 创建一个新的Nifti1Image对象来保存归一化后的数据
normalized_nii = nib.Nifti1Image(normalized_data, big_label_nii.affine)

# 使用nibabel的save方法保存到新的文件路径
# save_path = r"D:\B\Paper\domain\table\SOTA\MRA-SMI\FullySupervised\sub020.nii.gz"
save_path = r"D:\B\Paper\domain\table\SOTA\MRA-SMI\GT_V3_crop\sub020_Segmentation_7.nii.gz"
nib.save(normalized_nii, save_path)

print(f"Normalized data saved to {save_path}")
