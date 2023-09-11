import re
from datetime import datetime
import os
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torchvision import transforms
import argparse
import sys
sys.path.append(os.getcwd())

from data.dataset import MRImageData
from uda.MSCDA import MSCDA

import cv2
import nibabel as nib
from scipy.ndimage import zoom

def test_MSCDA(cfg):
    # Data loader
    transform = transforms.Compose([transforms.ToTensor()])
    # target = MRImageData(
    #     folder=cfg.dataset[1], is_supervised=True, modality=cfg.modality[1], transform=transform,
    #     subject_id=cfg.subject_id_test[1], aug=False, frame=cfg.frame[1], req_path=True)
    # one case example
    target = MRImageData(
        folder=cfg.dataset[1], is_supervised=True, modality=cfg.modality[1], transform=transform,
        subject_id=[20], aug=False, frame=cfg.frame[1], req_path=True)

    print('Target: {}'.format(len(target)))
    loader = torch.utils.data.DataLoader(
        target, batch_size=cfg.batch_size, num_workers=4, persistent_workers=True, shuffle=False,
        collate_fn=torch.utils.data.dataloader.default_collate, pin_memory=False, prefetch_factor=2, drop_last=False
    )

    # Initialize model
    model = MSCDA(cfg)
    model.load_networks(epoch=cfg.load_epoch)
    model._eval()
    dice_list = []
    ja_list = []
    hd_list = []
    pr_list = []
    sn_list = []
    roi_dice_list = []
    pixel_list = []
    path_list = []
    subject_list = []
    index_list = []
    class_score_list = []
    epoch_start_time = datetime.now()
    predictions = []
    ground_truths = []
    for i, data in enumerate(loader):
        model.set_input(data, test=True)
        # dice, ja, hd, pr, sn, flg_idx, pixel_idx, class_score = model.get_dice_eval(is_test=True)
        dice, ja, hd, pr, sn, flg_idx, pixel_idx, class_score, prediction, ground_truth = \
            model.get_dice_eval_save(is_test=True, ii=i)

        if i == 0:
            predictions = prediction
            ground_truths = ground_truth
        else:
            predictions = np.concatenate((predictions, prediction), axis=0)
            ground_truths = np.concatenate((ground_truths, ground_truth), axis=0)

        print('predictions.shape', predictions.shape)
        print('ground_truths.shape', ground_truths.shape)

        roi_dice = [dice[mi] for mi, v in enumerate(flg_idx) if v == 1]
        dice_list += dice
        ja_list += ja
        hd_list += hd
        pr_list += pr
        sn_list += sn
        path_list += data[2]
        roi_dice_list += roi_dice
        pixel_list += pixel_idx
        class_score_list += class_score

    # 定义目标大小
    target_shape = (163, 384, 320)
    predictions = zoom(predictions, (
        target_shape[0] / predictions.shape[0],
        target_shape[1] / predictions.shape[1],
        target_shape[2] / predictions.shape[2]))
    print(predictions.shape, 'predictions.shape')
    ground_truths = zoom(ground_truths, (
        target_shape[0] / ground_truths.shape[0],
        target_shape[1] / ground_truths.shape[1],
        target_shape[2] / ground_truths.shape[2]))
    print(ground_truths.shape, 'ground_truths.shape')

    resized_predictions = np.transpose(predictions, (1, 2, 0))
    resized_ground_truths = np.transpose(ground_truths, (1, 2, 0))

    print(resized_predictions.shape, 'resized_predictions.shape')
    print(resized_ground_truths.shape, 'resized_ground_truths.shape')

    # 定义保存路径
    output_folder = "C:/lfm/code/monai/MSCDA-main/test/data/pred"
    os.makedirs(output_folder, exist_ok=True)
    # 创建.nii.gz文件的NIfTI对象
    nifti_img = nib.Nifti1Image(resized_predictions.astype(np.float32), affine=np.eye(4))
    output_file_path = os.path.join(output_folder, "resized_predictions.nii.gz")
    nib.save(nifti_img, output_file_path)
    print(f"保存完成，文件路径为: {output_file_path}")
    nifti_img = nib.Nifti1Image(resized_ground_truths.astype(np.float32), affine=np.eye(4))
    output_file_path = os.path.join(output_folder, "resized_ground_truths.nii.gz")
    nib.save(nifti_img, output_file_path)
    print(f"保存完成，文件路径为: {output_file_path}")

    print(
        '[{}] Test Iter-{}  Time Taken: {}, All-Dice: {:.5}, Dice: {:.5}, Ja: {:.5}, HD: {:.5}, Pr: {:.5}, Sn: {:.5}'.format(
            datetime.now(), i, datetime.now() - epoch_start_time, np.average(dice_list),
            np.average(roi_dice_list),
                               np.sum(np.array(ja_list) * np.array(pixel_list)) / np.sum(pixel_list),
            np.average(np.array(hd_list).astype(np.float32)[~np.isnan(np.array(hd_list).astype(np.float32))]),
                               np.sum(np.array(pr_list) * np.array(pixel_list)) / np.sum(pixel_list),
                               np.sum(np.array(sn_list) * np.array(pixel_list)) / np.sum(pixel_list),
        ))
    print(roi_dice_list)

    for fn in path_list:
        index_list.append(int(fn.split('\\')[-1].split('_')[0].replace('.npz', '')))
        subject_list.append(int(re.search('Subject_(\\d+)', fn).group(1)))

    np.savez(model.save_path[:-4]+'.npz', dice=dice_list, ja=ja_list, hd=hd_list, pr=pr_list, sn=sn_list, pixel=pixel_list, index=index_list, subject=subject_list)
    print('Evaluation data saved !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scenario', help="scenario: '1' or '2'", default=1)
    parser.add_argument('-t', '--task', help="tasks: '4/8/11'", default=4)
    parser.add_argument('-f', '--fold', help="cross-validation fold: '1/2/3'", default=1)
    parser.add_argument('-b', "--batchsize", help="batch size", default=32)
    parser.add_argument("-e", "--epoch", help="load epoch", default=102)
    parser.add_argument("-g", "--gpuid", help="run model on gpu id, e.g., '1,2'", default='0')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if args.scenario == 1:
        from configs.Scenario1_config import cfg
    elif args.scenario == 2:
        from configs.Scenario2_config import cfg
    else:
        raise 'Undefined scenario.'
    cfg.task = args.task
    cfg.fold = args.fold
    # cfg.gpu_ids = np.arange(0, len(args.gpuid.split(','))).tolist()
    # cfg.batch_size = args.batchsize * len(cfg.gpu_ids)
    cfg.batch_size = args.batchsize
    cfg.batch_size_val = cfg.batch_size
    cfg.load_epoch = args.epoch

    test_MSCDA(cfg)
