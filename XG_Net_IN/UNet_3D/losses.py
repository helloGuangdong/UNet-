import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, pred, label):
        pred = torch.softmax(pred, 1)
        _, predmax = torch.max(pred, 1)

        predmax = np.squeeze(predmax.data.cpu().numpy())
        label = np.squeeze(label.data.cpu().numpy())
        width, height = np.shape(predmax)
        loss = []
        for i in range(0, width):
            for j in range(0, height):
                if predmax[i][j] != label[i][j] and label[i][j] == 1:
                    loss.append(3)
                elif predmax[i][j] != label[i][j] and label[i][j] == 0:
                    loss.append(1)
                else:
                    loss.append(0)
        loss = np.array(loss)
        loss = torch.from_numpy(loss).float()
        loss = torch.mean(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return 1-torch.mean(F_loss)
        else:
            return 1-F_loss


def DSC_LOSS(pred, Y):
    pred = torch.softmax(pred, 1)
    pred = pred[:, 1, :, :]
    N = pred.size(0)

    smooth = 1e-10

    pred_flat = pred.view(N, -1)
    Y_flat = Y.view(N, -1)
    Dice = 0
    dice = 1
    for i in range(1):
        Y_C = Y_flat
        Y_C[Y_C != i + 1] = -1
        intersection = torch.eq(pred_flat, Y_C).float()

        dice = 2 * (intersection.sum(1)) / (
                pred_flat[pred_flat == i + 1].sum().float() + Y_flat[Y_flat == i + 1].sum().float() + smooth)
        dice = dice.sum() / N

        # print('C {} dice is {:.4f}'.format(i + 1, dice))

    # return Dice/3
    return 1 - dice


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, pred, target):
        """
            pred tesor of shape = (N, C, H, W)
            target tensor of shape = (N, H, W)
        """
        nclass = pred.shape[1]

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        logits = F.softmax(pred, dim=1)

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(nclass):
            dice_loss = binaryDiceLoss(logits[:, i, :, :], target)
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / nclass