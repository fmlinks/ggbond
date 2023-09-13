import nibabel as nib
import numpy as np
from tqdm import tqdm

def find_coordinates(big_label_path, small_label_path):
    # 读取两个label文件
    big_label_nii = nib.load(big_label_path)
    small_label_nii = nib.load(small_label_path)

    # 获取数据数组
    big_label_data = big_label_nii.get_fdata()
    small_label_data = small_label_nii.get_fdata()

    big_label_data = big_label_data[48:-48, :, :]

    # 获取两个label的shape
    big_shape = big_label_data.shape
    small_shape = small_label_data.shape
    print(big_shape, small_shape, 'big_shape, small_shape')

    # # 遍历第一个label的每一个可能的起始坐标
    # for x in tqdm(range(big_shape[0] - small_shape[0] + 1)):
    #     for y in range(big_shape[1] - small_shape[1] + 1):
    #         for z in range(big_shape[2] - small_shape[2] + 1):
    #             # 获取一个子区域
    #             subregion = big_label_data[x:x+small_shape[0], y:y+small_shape[1], z:z+small_shape[2]]
    #
    #             # 如果子区域与第二个label完全匹配，则返回坐标
    #             if np.array_equal(subregion, small_label_data):
    #                 return (x, y, z)

    # 遍历第一个label的每一个可能的起始坐标
    for x in range(big_shape[0] - small_shape[0] + 1):
        for y in tqdm(range(big_shape[1] - small_shape[1] + 1)):
            for z in range(big_shape[2] - small_shape[2] + 1):
                # 获取一个子区域
                subregion = big_label_data[x:x + small_shape[0], y:y + small_shape[1], z:z + small_shape[2]]

                # 如果子区域与第二个label完全匹配，则返回坐标
                if np.array_equal(subregion, small_label_data):
                    return (x, y, z)

    # 如果没有找到匹配的子区域，则返回None
    return None

big_label_path = r"D:\B\data\SMILE-UHURA\train\label\sub018.nii"
small_label_path = r"D:\B\data\SMILE-UHURA\data\val\label\sub018.nii.gz"
coordinates = find_coordinates(big_label_path, small_label_path)
print(coordinates)
