import nibabel as nib
# from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
# import keras.backend as K
# import tensorflow as tf
# import math
# from scipy.ndimage import zoom

import sys

epsilon = sys.float_info.epsilon
print(epsilon)

def cal_base(y_true, y_pred):
    TP = np.sum(y_pred * y_true)
    TN = np.sum((-(y_pred - 1)) * (-(y_true - 1)))
    FP = np.sum(y_pred * (-(y_true - 1)))
    FN = np.sum((-(y_pred-1)) * y_true)
    return TP, TN, FP, FN


def acc(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + epsilon)
    return ACC


def dice(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    dice = ((2 * TP) + epsilon) / (2 * TP + FP + FN + epsilon)
    return dice


def sensitivity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP / (TP + FN + epsilon)
    return SE


def precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP / (TP + FP + epsilon)
    return PC


def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + epsilon)
    return SP


def f1_socre(y_true, y_pred):
    SE = sensitivity(y_true, y_pred)
    PC = precision(y_true, y_pred)
    F1 = 2 * SE * PC / (SE + PC + epsilon)
    return F1

# def MCC(y_true, y_pred):
#     TP, TN, FP, FN = cal_base(y_true, y_pred)
#     MCC = ((TP*TN)-(FP*TN))/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
#     print('MCC', MCC, 'fenmu',((TP*TN)-(FP*TN)), 'fenzi', np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
#
#     return MCC


def JAC(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    JAC = TP/(TP+FP+FN)
    return JAC

def VS(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    VS = 1-abs(FN-FP)/(2*TP + FP + FN)
    return VS


import scipy.ndimage.measurements as measure


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return np.uint8(largest_cc * 1)


def cal_matrix(i, methods):

    path1 = data_folder + '/00GT/'
    filename1 = os.listdir(path1)
    print(filename1[i])

    # path_image = '../00GT/image/'
    # imagename = os.listdir(path_image)

    path2 = data_folder + methods+'/'
    filename2 = os.listdir(path2)
    print(filename2[i])

    y_true = nib.load(path1 + filename1[i]).get_fdata()
    y_pred = nib.load(path2 + filename2[i]).get_fdata()

    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    # if hist_threshold:
    #     y_pred_hist = nib.load(path_image + imagename[i]).get_fdata()
    #     y_pred_hist[y_pred_hist < 140] = 0
    #     y_pred_hist[y_pred_hist >= 140] = 1
    #     y_pred = y_pred + y_pred_hist
    #     y_pred[y_pred > 1] = 1

    if largest_component:
        y_pred = get_largest_cc(y_pred)

    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)
    print(np.shape(y_true), 'y_true shape')
    print(np.shape(y_pred), 'y_pred shape')

    acc1 = acc(y_true, y_pred)
    sensitivity1 = sensitivity(y_true, y_pred)
    precision1 = precision(y_true, y_pred)
    specificity1 = specificity(y_true, y_pred)
    f1_socre1 = f1_socre(y_true, y_pred)
    dice1 = dice(y_true, y_pred)
    # MCC1 = MCC(y_true, y_pred)
    JAC1 = JAC(y_true, y_pred)
    VS1 = VS(y_true, y_pred)

    return acc1, sensitivity1, precision1, specificity1, f1_socre1, dice1, JAC1, VS1


if __name__ == '__main__':

    data_folder = r"D:\B\Paper\domain\table\SOTA\MRA-SMI\data/"

    methods = 'FullySupervised'
    # hist_threshold = False
    largest_component = False

    accs = []
    sensitivitys = []
    precisions = []
    specificitys = []
    f1_socres = []
    dices = []
    # MCCs = []
    JACs = []
    VSs = []


    for i in range(0, len(os.listdir(data_folder + '/00GT/'))):
        print(i)
        acc1, sensitivity1, precision1, specificity1, f1_socre1, dice1, JAC1, VS1 = cal_matrix(i, methods)
        accs.append(acc1)
        sensitivitys.append(sensitivity1)
        precisions.append(precision1)
        specificitys.append(specificity1)
        f1_socres.append(f1_socre1)
        print(dice1)
        dices.append(dice1)
        # MCCs.append(MCC1)
        JACs.append(JAC1)
        VSs.append(VS1)

    average_acc = np.mean(accs)
    average_sensitivity = np.mean(sensitivitys)
    average_precision = np.mean(precisions)
    average_specificity = np.mean(specificitys)
    average_f1_socre = np.mean(f1_socres)
    average_dice = np.mean(dices)
    # average_MCC = np.mean(MCCs)
    average_JAC = np.mean(JACs)
    average_VS = np.mean(VSs)

    std_acc = np.std(accs)
    std_sensitivity = np.std(sensitivitys)
    std_precision = np.std(precisions)
    std_specificity = np.std(specificitys)
    std_f1_socre = np.std(f1_socres)
    std_dice = np.std(dices)
    # std_MCC = np.std(MCCs)
    std_JAC = np.std(JACs)
    std_VS = np.std(VSs)

    print('################')
    print('accs', accs)
    print('sensitivitys', sensitivitys)
    print('precisions', precisions)
    print('specificitys', specificitys)
    print('JACs', JACs)
    print('VSs', VSs)
    print('dices', dices)
    print('################')

    print('acc: {} ± {}'.format(round(average_acc, 4), round(std_acc, 4)))
    print('sensitivity: {} ± {}'.format(round(average_sensitivity, 4), round(std_sensitivity, 4)))
    print('precision: {} ± {}'.format(round(average_precision, 4), round(std_precision, 4)))
    print('specificity: {} ± {}'.format(round(average_specificity, 4), round(std_specificity, 4)))
    # print('f1_socre: {} ± {}'.format(average_f1_socre, std_f1_socre))
    # print('MCC: {} ± {}'.format(average_MCC, std_MCC))
    print('JAC: {} ± {}'.format(round(average_JAC, 4), round(std_JAC, 4)))
    print('VS: {} ± {}'.format(round(average_VS, 4), round(std_VS, 4)))
    print('dice: {} ± {}'.format(round(average_dice, 4), round(std_dice, 4)))

    # print(len(os.listdir('../00GT/' + task + '-crop/')))
    # print(task)
    print(methods)
