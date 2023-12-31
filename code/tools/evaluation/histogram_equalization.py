import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical
# import tensorflow.keras.backend as K
from keras import backend as K

import SimpleITK as sitk
import scipy.misc
import scipy.ndimage
# from scipy.ndimage import zoom
import os
import numpy as np
import nibabel as nib
import scipy.ndimage.measurements as measure
# import matplotlib.pyplot as plt
import re
import shutil
from tqdm import tqdm
from sklearn.utils import shuffle
import gc
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


########## (1/3)preprocessing start ../data/step0 to ../data/step1  ##########

def load_nii(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    np_img = sitk.GetArrayFromImage(itkimage)
    return np_img, itkimage


def save_nii(filename, data, itkimage, new_spacing):
    dataITK = sitk.GetImageFromArray(data)
    dataITK.SetSpacing(new_spacing)
    dataITK.SetDirection(itkimage.GetDirection())
    dataITK.SetOrigin(itkimage.GetOrigin())
    sitk.WriteImage(dataITK, filename)


def resample(image, itkimage, new_spacing=[0.35, 0.35, 0.35]):
    # Determine current pixel spacing
    spacing = np.array(itkimage.GetSpacing(), dtype=np.float)
    spacing = [spacing[0], spacing[0], spacing[0]]
    # print(spacing)
    resize_factor = np.flip(spacing / np.array(new_spacing, dtype=np.float), 0)
    # print(resize_factor)

    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=3, mode='nearest')

    return image


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def HistogramEqualization(img):
    img = img.astype(np.int64)

    # img = scipy.ndimage.interpolation.zoom(img, 0.6, order=3, mode='nearest')

    image = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    image_int = image.astype(np.int64)

    hist, bins = np.histogram(image_int.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    array = cdf_normalized

    value1 = 0.01 * cdf_normalized[255]
    aa1 = find_nearest(array, value1)
    bb1 = np.where(cdf_normalized == aa1)
    # print(bb1[0][0])

    # value2 = 1 * cdf_normalized[255]
    value2 = 0.999 * cdf_normalized[255]
    aa2 = find_nearest(array, value2)
    bb2 = np.where(cdf_normalized == aa2)
    # print(bb2[0][0])

    image_norm = image_int.copy()
    image_norm[image_norm < bb1[0][0]] = bb1[0][0]
    image_norm[image_norm > bb2[0][0]] = bb2[0][0]
    image_norm = ((image_norm - np.min(image_norm)) / (np.max(image_norm) - np.min(image_norm))) * 255
    image_norm_int = image_norm.astype(np.int)

    return image_norm_int


if __name__ == '__main__':
    INPUT_FOLDER = '../00GT/image_org/'
    patients = os.listdir(INPUT_FOLDER)
    print(patients)
    patients.sort()
    new_spacing = [0.35, 0.35, 0.35]
    for patient in patients:
        filename = INPUT_FOLDER + patient
        print(filename)
        np_img = nib.load(filename).get_data()
        np_img[np_img == np.min(np_img)] = np.median(np_img)

        # Histogram Equalization
        image_norm_int = HistogramEqualization(np_img)
        output = filename.replace('_image_standard', '')
        output = output.replace('image_org', 'image')

        npy = image_norm_int
        new_image = nib.Nifti1Image(npy, np.eye(4))
        new_image.set_data_dtype(np.uint8)
        nib.save(new_image, output)
                                                                                                                                                                                                                                                                                                                                                   