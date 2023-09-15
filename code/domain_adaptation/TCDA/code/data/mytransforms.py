import nibabel as nib
import monai.transforms as transforms
from monai.utils.type_conversion import convert_to_tensor
import numpy as np
from typing import Callable, Dict, Hashable, Mapping, Optional
from monai.transforms.transform import MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor

smooth=1e-5


class MyHistogramEqualizationTransform(transforms.Transform):
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        # img = img.numpy()
        image = (img - np.min(img)) / (np.max(img) - np.min(img))
        image = image * image * 255

        image_int = image.astype(np.int64)

        hist, bins = np.histogram(image_int.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        array = cdf_normalized

        # value1 = 0.010 * cdf_normalized[255]
        value1 = 0.010 * cdf_normalized[255]

        aa1 = self.find_nearest(array, value1)
        bb1 = np.where(cdf_normalized == aa1)

        value2 = 0.999 * cdf_normalized[255]
        aa2 = self.find_nearest(array, value2)
        bb2 = np.where(cdf_normalized == aa2)

        image_norm = image.copy()
        image_norm[image_norm < bb1[0][0]] = bb1[0][0]
        image_norm[image_norm > bb2[0][0]] = bb2[0][0]
        image_norm = (image_norm - np.min(image_norm) + smooth) / (np.max(image_norm) - np.min(image_norm) + smooth)
        image_norm = convert_to_tensor(image_norm)

        return image_norm

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

class MyHistogramEqualizationTransform_forFDA3DRA(transforms.Transform):
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        # img = img.numpy()
        img = img[0, 0, :, :, :]
        image = (img - np.min(img) + smooth) / (np.max(img) - np.min(img) + smooth)
        image = image * image * 255

        image_int = image.astype(np.int64)

        hist, bins = np.histogram(image_int.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        array = cdf_normalized

        # value1 = 0.010 * cdf_normalized[255]
        value1 = 0.5 * cdf_normalized[255]

        aa1 = self.find_nearest(array, value1)
        bb1 = np.where(cdf_normalized == aa1)

        value2 = 0.999 * cdf_normalized[255]
        aa2 = self.find_nearest(array, value2)
        bb2 = np.where(cdf_normalized == aa2)

        image_norm = image.copy()
        image_norm[image_norm < bb1[0][0]] = bb1[0][0]
        image_norm[image_norm > bb2[0][0]] = bb2[0][0]
        image_norm = (image_norm - np.min(image_norm) + smooth) / (np.max(image_norm) - np.min(image_norm) + smooth)
        # image_norm = convert_to_tensor(image_norm)
        image_norm = image_norm[np.newaxis, np.newaxis, :, :, :]
        return image_norm

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


class MyHistogramEqualizationTransform_forFDAMRA(transforms.Transform):
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        # img = img.numpy()
        img = img[0, 0, :, :, :]
        image = (img - np.min(img) + smooth) / (np.max(img) - np.min(img) + smooth)
        image = image * image * 255

        image_int = image.astype(np.int64)

        hist, bins = np.histogram(image_int.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        array = cdf_normalized

        # value1 = 0.010 * cdf_normalized[255]
        value1 = 0.99 * cdf_normalized[255]

        aa1 = self.find_nearest(array, value1)
        bb1 = np.where(cdf_normalized == aa1)

        value2 = 0.999 * cdf_normalized[255]
        aa2 = self.find_nearest(array, value2)
        bb2 = np.where(cdf_normalized == aa2)

        image_norm = image.copy()
        image_norm[image_norm < bb1[0][0]] = bb1[0][0]
        image_norm[image_norm > bb2[0][0]] = bb2[0][0]
        image_norm = (image_norm - np.min(image_norm) + smooth) / (np.max(image_norm) - np.min(image_norm) + smooth)
        # image_norm = convert_to_tensor(image_norm)
        image_norm = image_norm[np.newaxis, np.newaxis, :, :, :]
        return image_norm

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


class MyHistogramEqualizationTransformd(MapTransform):

    backend = MyHistogramEqualizationTransform.backend

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = MyHistogramEqualizationTransform()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key])
        return d


Histogram_3DRA = MyHistogramEqualizationTransform_forFDA3DRA()
Histogram_MRA = MyHistogramEqualizationTransform_forFDAMRA()
