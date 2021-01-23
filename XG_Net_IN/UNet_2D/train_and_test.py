import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import SimpleITK as sitk

# from UNet_2D.data_set import MyDataset, testDataset
from unet2d_znw import UNet2D
from data_set import MyDataset, testDataset
from MYLOSS import myLoss, FocalLoss, DSC_LOSS


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
# sys.path.append(r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_2/XG_Net/UNet_2D')



def DSC(pred, Y):
    N = pred.size(0)

    smooth = 1e-10

    pred_flat = pred.view(N, -1)
    Y_flat = Y.view(N, -1)
    Dice = 0
    for i in range(1):
        Y_C = Y_flat
        Y_C[Y_C != i + 1] = -1
        intersection = torch.eq(pred_flat, Y_C).float()

        dice = 2 * (intersection.sum(1)) / (
                pred_flat[pred_flat == i + 1].sum().float() + Y_flat[Y_flat == i + 1].sum().float() + smooth)
        dice = dice.sum() / N

        # print('C {} dice is {:.4f}'.format(i + 1, dice))

    # return Dice/3
    return dice


def DSC_2(pred, Y):
    pred_gt_data = pred.data.cpu().numpy()*Y.data.cpu().numpy()
    pred_sum = np.sum(pred.data.cpu().numpy())
    gt_sum = np.sum(Y.data.cpu().numpy())
    pred_gt_sum = np.sum(pred_gt_data)

    return (2*pred_gt_sum)/(pred_sum+gt_sum)


if __name__ == "__main__":

    train_dataset = MyDataset(path_image='/data/zhangnaiwen442/testNorm/fold_2/Vessel_Training/VesselSeg')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = testDataset(path_image='/data/zhangnaiwen442/testNorm/fold_2/Vessel_Test/VesselSeg')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet2D(n_class=2)
    # device = torch.device("cuda: 2" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.3, weight_decay=0.0, nesterov=False)

    best_loss = 100000
    best_dice = 0
    dicesum = 0
    bestepoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(50 * 0.8), gamma=0.5)
    # train the model
    for epoch in range(2000):
        model.train()
        all_loss = 0
        index = 0
        for batch_ndx, x in enumerate(train_dataloader):
            optimizer.zero_grad()
            data = x['vein']
            label = x['label']
            data, label = data.to(device), label.to(device).long()
            pred = model(data)
            # pred = torch.softmax(pred, 1)
            # print("training pred.shape:", pred.shape, "label.shape:", label.shape)
            # loss = DSC(pred[:, 1, :, :], label)
            loss = F.cross_entropy(pred, label)
            # print('**** ', epoch, ' **** loss:', loss)
            # lossFun = FocalLoss()
            # loss = lossFun(pred, label)
            loss.backward()
            optimizer.step()
            all_loss += loss
            index += 1

        print("Epoch {}, training loss is {:.4f}".format(epoch + 1, all_loss/index))

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                alltestdsc = 0
                for valnum, TestData in enumerate(test_dataloader):
                    test_image, test_label = TestData['vein'].to(device), TestData['label'].to(device).long()

                    test_label = np.squeeze(test_label)
                    vxpred = torch.zeros(test_label.shape)
                    vxpred = vxpred.to(device)
                    for vallayer in range(0, test_image.shape[2]):
                        validpred = model(test_image[:, :, vallayer, :, :])
                        validpred = torch.softmax(validpred, 1)
                        # validpred = validpred[:, 1, :, :]
                        # validpred[validpred > 0.5] = 1
                        # validpred[validpred <= 0.5] = 0
                        _, validpredmax = torch.max(validpred, 1)
                        vxpred[vallayer, :, :] = np.squeeze(validpredmax)
                    Dice = DSC_2(vxpred, test_label)
                    print('test_label.sum', np.sum(test_label.data.cpu().numpy()), 'predimage.sum', np.sum(vxpred.data.cpu().numpy()))
                    predimage = sitk.GetImageFromArray(vxpred.data.cpu().numpy())
                    save_path = '/data/zhangnaiwen442/testNorm/fold_2/predImage_IN/' + str(epoch) + '_1209_' + str(valnum) + '.nii.gz'
                    sitk.WriteImage(predimage, save_path)

                    print("testing dice is {:.4f}".format(Dice))
                    print(" ")
                    alltestdsc += Dice

                if alltestdsc > best_dice:
                    best_dice = alltestdsc
                    bestepoch = epoch
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict()
                    }, "./lr4_IN_2d.pt")
                    # print("Saving model Fold1.pt")

                print('Epoch {},best mean dice is {:.4f}'.format(bestepoch + 1, best_dice / (valnum + 1)))
                # alltestdsc = 0


