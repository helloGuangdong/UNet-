# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:18:06 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:42:41 2020

@author: Administrator
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:50:26 2020

@author: ruixin-12
"""
import SimpleITK as sitk
import os,shutil
# import numpy
# import SimpleITK
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from skimage.measure import label,regionprops
# from skimage.io import imread, imshow
# from skimage.filters import gaussian, threshold_otsu,threshold_yen,threshold_li,threshold_isodata
# from skimage import measure
# import pandas as pd
def saveArray2nii(image,path,origin,direction,space):
    
    newImg = sitk.GetImageFromArray(image)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing(space)  
    sitk.WriteImage(newImg, path)

def read_nii_image(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    space = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    return data,space,origin,direction

def mkpathdir(path):
    if os.path.exists(path):
        print('path exist')
    else:
        os.mkdir(path)


if __name__ == "__main__":
    ### brain
    dialatepath = r'/data/zhangnaiwen442/ArteryData'
    rawdatapath = r'/data/zhangnaiwen442/VeinDataThr40'
    GTpath = r'/data/zhangnaiwen442/newGT20201130'
    
    
#fold1     
    testsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_1/Vessel_Test/VesselSeg'
    trainsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_1/Vessel_Training/VesselSeg'
    for i in range(1, 11):
        
        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(testsavepath, str(i), str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(testsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)
        
        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(testsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)        
        
        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(testsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)        
        
    for i in range(11,50):
        
        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)
        
        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     
        
        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)  


#fold2   
    testsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_2/Vessel_Test/VesselSeg'
    trainsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_2/Vessel_Training/VesselSeg'
    for i in range(11,21):
        
        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(testsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(testsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)
        
        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(testsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)        
        
        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(testsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)        
        
    for i in range(1,11):
        
        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)
        
        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     
        
        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)   
        
    for i in range(21,50):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)  

#fold3  
    testsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_3/Vessel_Test/VesselSeg'
    trainsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_3/Vessel_Training/VesselSeg'
    for i in range(21,31):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(testsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(testsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(testsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)        

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(testsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)        

    for i in range(1,21):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)   

    for i in range(31,50):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name) 

#fold4 
    testsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_4/Vessel_Test/VesselSeg'
    trainsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_4/Vessel_Training/VesselSeg'
    for i in range(31,41):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(testsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(testsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(testsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)        

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(testsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)        

    for i in range(1,31):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)   

    for i in range(41,50):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name) 

#fold5
    testsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_5/Vessel_Test/VesselSeg'
    trainsavepath = r'/data/zhangnaiwen442/all_fold_thr40_1205/fold_5/Vessel_Training/VesselSeg'
    for i in range(41,50):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(testsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(testsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(testsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)        

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(testsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)        

    for i in range(1,41):

        dialateorigin_path = os.path.join(dialatepath,'PPC_3D_10_'+str(i)+'.nii.gz')
        dialatenew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_artery.nii.gz')
        mkpathdir(os.path.join(trainsavepath,str(i)))
        shutil.copyfile(dialateorigin_path, dialatenew_file_name)

        rawdataorigin_path = os.path.join(rawdatapath,'PPC_3D_10_MSUM_'+str(i)+'.nii.gz')
        rawdatanew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_vein.nii.gz')
        shutil.copyfile(rawdataorigin_path, rawdatanew_file_name)     

        GTorigin_path = os.path.join(GTpath,'addGT'+str(i)+'d.nii.gz')
        GTnew_file_name = os.path.join(trainsavepath,str(i),str(i)+'_seg.nii.gz')
        shutil.copyfile(GTorigin_path, GTnew_file_name)   
        
 