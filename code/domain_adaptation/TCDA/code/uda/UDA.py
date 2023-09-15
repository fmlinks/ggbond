import itertools
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from network.loss.loss import DiceLoss
import network as nt
import os
from data.mytransforms import Histogram_3DRA as augmentation_fda_3dra
from data.mytransforms import Histogram_MRA as augmentation_fda_mra
import skimage.filters as filters

from network.backbone.swinunet import SwinUNETR


class TSNetTrainer(nn.Module):
    def __init__(self):
        super(TSNetTrainer, self).__init__()

        # general settings
        self.device = torch.device('cuda:0')
        self.iters = 1
        self.ema_decay = 0.999
        self.epoch = 1

        self.ckpt_dir = '/weight/'
        self.data_dir = 'C:/lfm/code/monai/domain/'

        self.net_student = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=2, feature_size=12,
                                     use_checkpoint=True).to(device=torch.device("cuda"))
        # self.net_student.load_state_dict(torch.load(os.path.join('C:/lfm/code/monai/Aneurist/weights/',
        #                                                          "train02_01-DSC0.9469.pth")))

        self.net_teacher = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=2, feature_size=12,
                                     use_checkpoint=True).to(device=torch.device("cuda"))
        self.detach_model(self.net_teacher)
        self.nets = ['net_student', 'net_teacher']

        self.criterionDice = DiceCELoss(to_onehot_y=True, softmax=True)

        self.criterionDice2 = DiceLoss(to_onehot_y=True, softmax=True)

        self.criterionCons = nn.MSELoss()
        # self.criterionSimilarity = nn.CosineSimilarity(dim=[1, 2, 3, 4]) # *********************************
        # dim=1 的话 输入tensor大小是 [8, 128, 4, 4 ,4]，这样比完之后生成的loss是[8, 4, 4, 4]。这里需要优化
        self.criterionSimilarity = nn.MSELoss()
        # self.criterionCosineSimilarity = nn.CosineSimilarity(dim=1)

        # 这里需要加负号或者变成loss模式，或者直接写一个复合函数
        # 这里的CosineSimilarity Loss可以大作文章，目前是简化版，后边用单独的函数来写

        # initialize optimizer
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.net_student.parameters(), self.net_teacher.parameters()),
            lr=0.001, betas=(0.9, 0.999))
        self.optimizers = [self.optimizer]
        self.schedulers = [nt.get_scheduler(optimizer, lr_policy='lambda') for optimizer in self.optimizers]

        # data variables
        # source image (3DRA)
        self.img_stu_S = None
        self.label_stu_S = None
        self.seg_stu_S = None
        self.label_tea_T = None
        self.proj_stu_S = None
        self.pred_stu_S = None

        # source-like target image (3DRA' from mra)
        self.img_stu_T = None
        # self.label_stu_T = None # No Label
        self.seg_stu_T = None
        self.proj_stu_T = None
        self.pred_stu_T = None

        # target image (MRA)
        self.img_tea_T = None
        # self.label_tea_T = None  # No Label
        self.seg_tea_T = None
        self.proj_tea_T = None
        self.pred_tea_T = None

        # target-like source image (MRA' from 3dra)
        self.img_tea_S = None
        self.label_tea_S = None
        self.seg_tea_S = None
        self.proj_tea_S = None
        self.pred_tea_S = None

        # pseudo label
        self.pseudo_stu_S = None
        self.pseudo_stu_T = None
        self.pseudo_tea_S = None
        self.pseudo_tea_T = None

        # loss tensors
        self.loss_supervised_loss_student = None
        self.loss_supervised_loss_teacher = None
        self.loss_supervised_loss_mra = None
        self.loss_consistency_loss_source = None
        self.loss_consistency_loss_target = None
        # ###### ****** ****** ****** 这里可大作文章 ****** ****** ****** ######
        self.loss_contrast_stu_proj_vs_tea_pred_S = None
        self.loss_contrast_stu_proj_vs_tea_pred_T = None
        self.loss_contrast_stu_pred_vs_tea_proj_S = None
        self.loss_contrast_stu_pred_vs_tea_proj_T = None

        self.self_supervised_loss_student_RA = None
        self.self_supervised_loss_student_MRA = None
        self.self_supervised_loss_teacher_RA = None
        self.self_supervised_loss_teacher_MRA = None

        # loss variables for monitor
        self.v_loss_supervised_loss_student = None
        self.v_loss_supervised_loss_teacher = None
        self.v_loss_consistency_loss_source = None
        self.v_loss_consistency_loss_target = None
        self.v_loss_contrast_stu_proj_vs_tea_pred_S = None
        self.v_loss_contrast_stu_proj_vs_tea_pred_T = None
        self.v_loss_contrast_stu_pred_vs_tea_proj_S = None
        self.v_loss_contrast_stu_pred_vs_tea_proj_T = None
        self.v_loss = None

    def set_input(self, inputs):
        """ This function is for one-stream setup. """

        self.img_stu_S = inputs[0].to(self.device)
        self.label_stu_S = inputs[1].to(self.device)

        # Training
        # img_stu_T_numpy = torch.from_numpy(augmentation_fda(im_src=inputs[2], im_trg=inputs[0]))
        # Testing
        # img_stu_T_numpy = torch.from_numpy(augmentation_fda(im_src=inputs[2].cpu().numpy(), im_trg=inputs[0].cpu().numpy()))
        # print(inputs[2].shape, 'ccccccc')
        img_stu_T_numpy = torch.from_numpy(augmentation_fda_mra(inputs[2].cpu().numpy()))

        self.img_stu_T = img_stu_T_numpy.to(self.device, dtype=torch.float)

        self.img_tea_T = inputs[2].to(self.device)
        # self.label_tea_T = inputs[3].to(self.device)  # No Label

        # Training
        # img_tea_S_numpy = torch.from_numpy(augmentation_fda(im_src=inputs[0], im_trg=inputs[2]))
        # Testing
        # img_tea_S_numpy = torch.from_numpy(augmentation_fda(im_src=inputs[0].cpu().numpy(), im_trg=inputs[2].cpu().numpy()))
        img_tea_S_numpy = torch.from_numpy(augmentation_fda_3dra(inputs[0].cpu().numpy()))
        # print(img_stu_T_numpy.shape, 'aaaaaa')
        # print(img_tea_S_numpy.shape, 'bbbbb')

        self.img_tea_S = img_tea_S_numpy.to(self.device, dtype=torch.float)
        self.label_tea_S = inputs[1].to(self.device)

        # self.pseudo_stu_S = self.get_pseudo_anchor_label(inputs[0]).to(self.device)
        # self.pseudo_stu_T = self.get_pseudo_anchor_label(img_stu_T_numpy).to(self.device)
        # self.pseudo_tea_S = self.get_pseudo_anchor_label(img_tea_S_numpy).to(self.device)
        # self.pseudo_tea_T = self.get_pseudo_anchor_label(inputs[2]).to(self.device)

        self.pseudo_stu_T = inputs[3].to(self.device)
        self.pseudo_tea_T = inputs[3].to(self.device)





    def forward(self):
        self.seg_stu_S, self.pred_stu_S, self.proj_stu_S = self.net_student(self.img_stu_S)
        self.seg_stu_T, self.pred_stu_T, self.proj_stu_T = self.net_student(self.img_stu_T)
        self.seg_tea_T, self.pred_tea_T, self.proj_tea_T = self.net_teacher(self.img_tea_T)
        self.seg_tea_S, self.pred_tea_S, self.proj_tea_S = self.net_teacher(self.img_tea_S)
        return self.seg_stu_S, self.seg_tea_T, self.seg_stu_T, self.seg_tea_S

    def backward(self):

        # Supervised loss
        self.loss_supervised_loss_student = self.criterionDice(self.seg_stu_S, self.label_stu_S)

        self.loss_supervised_loss_teacher = self.criterionDice(self.seg_tea_S, self.label_tea_S)

        # self.self_supervised_loss_student_RA = self.criterionDice2(self.seg_stu_S, self.pseudo_stu_S)

        self.self_supervised_loss_student_MRA = self.criterionDice2(self.seg_stu_T, self.pseudo_stu_T)

        # self.self_supervised_loss_teacher_RA = self.criterionDice2(self.seg_tea_S, self.pseudo_tea_S)

        self.self_supervised_loss_teacher_MRA = self.criterionDice2(self.seg_tea_T, self.pseudo_tea_T)

        # self.loss_supervised_loss_teacher = 0.2 * dice_coefficient_3d_ra_vessel(self.seg_tea_S, self.label_tea_S) +\
        #                                     0.8 * dice_coefficient_3d_ra_aneurysm(self.seg_tea_S, self.label_tea_S)

        # self.loss_supervised_loss_mra = self.criterionDice(self.seg_tea_T, self.label_tea_T)

        self.loss_consistency_loss_source = self.criterionCons(
            F.softmax(self.seg_tea_S, dim=1).detach(), F.softmax(self.seg_stu_S, dim=1))
        self.loss_consistency_loss_target = self.criterionCons(
            F.softmax(self.seg_tea_T, dim=1).detach(), F.softmax(self.seg_stu_T, dim=1))

        # 目前是简化版本，用于跑通程序
        # ###### ****** ****** ****** 这里可大作文章 ****** ****** ****** ######
        # 这里可以定义成对比学习loss，用来自同一个数据集的当做positive，用来自于不同数据集的当做negative
        # self.loss_contrast_stu_proj_vs_tea_pred_S = self.criterionSimilarity(self.proj_stu_S, self.pred_tea_S)
        # self.loss_contrast_stu_proj_vs_tea_pred_T = self.criterionSimilarity(self.proj_stu_T, self.pred_tea_T)
        # self.loss_contrast_stu_pred_vs_tea_proj_S = self.criterionSimilarity(self.pred_stu_S, self.proj_tea_S)
        # self.loss_contrast_stu_pred_vs_tea_proj_T = self.criterionSimilarity(self.pred_stu_T, self.proj_tea_T)

        self.loss_contrast_stu_proj_vs_tea_pred_S = self.criterionSimilarity(self.seg_stu_S, self.seg_tea_S)
        self.loss_contrast_stu_proj_vs_tea_pred_T = self.criterionSimilarity(self.seg_stu_T, self.seg_tea_T)

        self.loss_tsne = self.criterionCosineSimilarity(self.proj_stu_S, self.proj_tea_S)

        # backward
        # loss_contrast_all = 0.5 * self.loss_contrast_stu_proj_vs_tea_pred_S + \
        #                     0.5 * self.loss_contrast_stu_proj_vs_tea_pred_T + \
        #                     0.5 * self.loss_contrast_stu_pred_vs_tea_proj_S + \
        #                     0.5 * self.loss_contrast_stu_pred_vs_tea_proj_T

        # loss = 0.5 * self.loss_supervised_loss_student + \
        #        0.2 * self.loss_supervised_loss_teacher + \
        #        0.2 * self.loss_consistency_loss + \
        #        0.1 * loss_contrast_all

        # loss = 0.6 * self.loss_supervised_loss_student + \
        #        0.2 * self.loss_supervised_loss_teacher + \
        #        0.1 * self.loss_consistency_loss_source + \
        #        0.1 * self.loss_consistency_loss_target

        # loss = 0.8 * self.loss_supervised_loss_student + 0.2 * self.loss_supervised_loss_teacher

        # loss = 0.3 * self.loss_supervised_loss_student + 0.3 * self.loss_supervised_loss_teacher + \
        #         0.1 * self.self_supervised_loss_student_RA + 0.1 * self.self_supervised_loss_student_MRA + \
        #         0.1 * self.self_supervised_loss_teacher_RA + 0.1 * self.self_supervised_loss_teacher_MRA

        loss = 0.3 * self.loss_supervised_loss_student + \
               0.3 * self.loss_supervised_loss_teacher + \
               0.1 * self.self_supervised_loss_student_MRA + \
               0.1 * self.self_supervised_loss_teacher_MRA + \
               0.1 * self.loss_consistency_loss_source + \
               0.1 * self.loss_consistency_loss_target + \
               0.1 * self.loss_contrast_stu_proj_vs_tea_pred_S + \
               0.1 * self.loss_contrast_stu_proj_vs_tea_pred_T + \
               0.1 * self.loss_tsne

        self.v_loss_supervised_loss_student = self.loss_supervised_loss_student.item()
        self.v_loss_supervised_loss_teacher = self.loss_supervised_loss_teacher.item()
        # self.v_loss_consistency_loss_source = self.loss_consistency_loss_source.item()
        # self.v_loss_consistency_loss_target = self.loss_consistency_loss_target.item()
        # self.v_loss_contrast_stu_proj_vs_tea_pred_S = self.loss_contrast_stu_proj_vs_tea_pred_S.item()
        # self.v_loss_contrast_stu_proj_vs_tea_pred_T = self.loss_contrast_stu_proj_vs_tea_pred_T.item()
        # self.v_loss_contrast_stu_pred_vs_tea_proj_S = self.loss_contrast_stu_pred_vs_tea_proj_S.item()
        # self.v_loss_contrast_stu_pred_vs_tea_proj_T = self.loss_contrast_stu_pred_vs_tea_proj_T.item()
        self.v_loss = loss.item()

        loss.backward()

    @torch.no_grad()
    def ema(self):
        alpha = min(1 - 1 / (self.iters + 1), self.ema_decay)
        for ema_param, param in zip(self.net_teacher.parameters(), self.net_student.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.ema()
        # self.update_centroid()
        # self.update_pixel()
        self.iters += 1

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_loss(self):
        # v_loss = [
        #     self.v_loss_supervised_loss_student,
        #     self.v_loss_supervised_loss_teacher,
        #     self.v_loss_consistency_loss,
        #     self.v_loss_contrast_stu_proj_vs_tea_pred_S,
        #     self.v_loss_contrast_stu_proj_vs_tea_pred_T,
        #     self.v_loss_contrast_stu_pred_vs_tea_proj_S,
        #     self.v_loss_contrast_stu_pred_vs_tea_proj_T,
        #     self.v_loss]
        v_loss = [
            self.v_loss_supervised_loss_student,
            self.v_loss_supervised_loss_teacher,
            self.v_loss]
        return np.array(v_loss)

    @staticmethod
    def detach_model(net):
        for param in net.parameters():
            param.detach_()

    @staticmethod
    def get_pseudo_anchor_label(image):
        # Convert the tensor to a NumPy array
        image_np = image.cpu().numpy()

        # Apply thresholding using scikit-image
        threshold = filters.threshold_otsu(image_np)
        binary_image_np = image_np > threshold
        binary_image_np = binary_image_np.astype(np.uint8)

        # Convert the NumPy array back to a PyTorch tensor
        pseudo_label = torch.from_numpy(binary_image_np)

        # Return the binary image as a PyTorch tensor
        return pseudo_label

    def save_networks(self, save_name=None):
        for net_name in self.nets:
            net = self.__getattr__(net_name)
            if save_name:
                save_filename = '{}_{}_iter{}.pth'.format(net_name, net.module.name, save_name)
            else:
                save_filename = '{}_{}_iter{}.pth'.format(net_name, net.module.name, self.epoch)
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            save_path = os.path.join(self.data_dir, save_filename)
            print('save path:', save_path)

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            torch.save({'state_dict': state_dict, 'epoch': self.epoch}, save_path)

    # def load_networks(self):

    def _eval(self):
        self.net_student.eval()
        self.eval()

    def _train(self):
        self.net_student.train()
        self.train()











