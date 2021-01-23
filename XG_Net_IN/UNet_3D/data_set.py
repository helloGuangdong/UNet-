from torch.utils.data import DataLoader, Dataset
import os
import h5py
import numpy as np
import PIL.Image as Image

import nibabel as nib
from torchvision import transforms
import SimpleITK as sitk


def get_filelist_frompath(filepath):
    file_name = os.listdir(filepath)
    artery_file_List = []
    vein_file_List = []
    seg_file_List = []
    for file in file_name:
        path = os.path.join(filepath, file)
        file_name_2 = os.listdir(path)
        for file_2 in file_name_2:
            if file_2 == str(file) + '_' + 'artery.nii.gz':
                artery_file_List.append(os.path.join(path, file_2))

            if file_2 == str(file) + '_' + 'vein.nii.gz':
                vein_file_List.append(os.path.join(path, file_2))

            if file_2 == str(file) + '_' + 'seg.nii.gz':
                seg_file_List.append(os.path.join(path, file_2))
    return artery_file_List, vein_file_List, seg_file_List


class MyDataset(Dataset):
    def __init__(self, path_image='/data/ZNW/all_fold_1130/fold_1/Vessel_Training/VesselSeg', transform=None, target_transform=None):

        artery_file_List, vein_file_List, seg_file_List = get_filelist_frompath(path_image)

        artery_list = []
        vein_list = []
        label_list = []
        for i in range(0, len(artery_file_List)):
            print(artery_file_List[i], '\n', vein_file_List[i], '\n', seg_file_List[i], '\n')
            artery_data = sitk.ReadImage(artery_file_List[i])
            artery_data = sitk.GetArrayFromImage(artery_data)

            vein_data = sitk.ReadImage(vein_file_List[i])
            vein_data = sitk.GetArrayFromImage(vein_data)

            label = sitk.ReadImage(seg_file_List[i])
            label = sitk.GetArrayFromImage(label)
            for ii in range(0, int(artery_data.shape[0]/8)):

                artery_list.append(artery_data[ii*8:(ii+1)*8])
                vein_list.append(vein_data[ii*8:(ii+1)*8])
                label_list.append(label[ii*8:(ii+1)*8])

        self.artery_list = artery_list
        self.vein_list = vein_list
        self.label_list = label_list

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        artery = self.artery_list[index]
        vein = self.vein_list[index]
        label = self.label_list[index]

        artery = artery.astype(np.float32)
        vein = vein.astype(np.float32)
        label = label.astype(np.float32)

        sample = {'artery': np.expand_dims(artery, 0), 'vein': np.expand_dims(vein, 0), 'label': label}
        return sample

    def __len__(self):
        return len(self.artery_list)


class testDataset(Dataset):
    def __init__(self, path_image='/data/ZNW/all_fold_1130/fold_1/Vessel_Test/VesselSeg', transform=None, target_transform=None):

        artery_file_List, vein_file_List, seg_file_List  = get_filelist_frompath(path_image)

        self.artery_file_List = artery_file_List
        self.vein_file_List = vein_file_List
        self.seg_file_List = seg_file_List

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        artery = sitk.ReadImage(self.artery_file_List[index])
        artery = sitk.GetArrayFromImage(artery)
        vein = sitk.ReadImage(self.vein_file_List[index])
        vein = sitk.GetArrayFromImage(vein)
        label = sitk.ReadImage(self.seg_file_List[index])
        label = sitk.GetArrayFromImage(label)

        artery = artery.astype(np.float32)
        vein = vein.astype(np.float32)
        label = label.astype(np.float32)

        sample = {'artery': np.expand_dims(artery, 0), 'vein': np.expand_dims(vein, 0), 'label': label}
        return sample

    def __len__(self):
        return len(self.artery_file_List)



