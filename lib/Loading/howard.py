# Python Modules
import glob
import os
import numpy as np
import time
# Torch Modules
import torch
from torch.utils.data import Dataset
# Personal Modules
import lib.augment3D as augment3D
import scipy
from scipy.ndimage import zoom
import SimpleITK as sitk



class HOWARD(Dataset):
    """
    Code for reading data collected and delineated by Dr. Howard Morgan, this is the final dataset we used for research project:
    Chen, Meixu, et al. "TransAnaNet: Transformer‐based anatomy change prediction network for head and neck cancer radiotherapy." Medical Physics (2025).

    Please modify this module according to the your data
    """

    def __init__(self, args, mode='train', dataset_path='./datasets', label_path='./datasets', classes=2):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = dataset_path
        self.label_path = label_path
        # self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        # self.samples = samples
        self.full_volume = None
        self.classes = classes
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomRotation(min_angle=-10, max_angle=10), augment3D.RandomShift()], p=0.9)
            
        if self.mode == 'test':
            print('will do in the future')
     
        else:
            print("self root",self.root)
            patients = glob.glob(os.path.join(self.root, '*_GTVp_CT.nii.gz'))

            dose=[]
            ct=[]
            ct_gtv=patients
            ct_gtvn=[]

            cbct1=[]
            cbct1_gtv=[]
            cbct1_gtvn=[]

            cbct2=[]
            cbct2_gtv=[]
            cbct2_gtvn=[]

            for p in patients:
                temp = p.split('/')[-1] # name of file
                id = temp.split('_GTVp')[0] # patient ID
                print("find patient", id)

                dose += glob.glob(os.path.join(self.root, id+'_Dose.nii.gz'))

                ct += glob.glob(os.path.join(self.root, id+'_CT.nii.gz'))
                ct_gtvn += glob.glob(os.path.join(self.root, id+'_GTVn_CT.nii.gz'))

                cbct1 += glob.glob(os.path.join(self.root, id+'_CBCT1.nii.gz'))
                cbct1_gtv += glob.glob(os.path.join(self.root, id+'_GTVp_CBCT1.nii.gz'))
                cbct1_gtvn += glob.glob(os.path.join(self.root, id+'_GTVn_CBCT1.nii.gz'))

                cbct2 += glob.glob(os.path.join(self.label_path, id+'_CBCT2.nii.gz'))
                cbct2_gtv += glob.glob(os.path.join(self.label_path, id+'_GTVp_CBCT2.nii.gz'))
                cbct2_gtvn += glob.glob(os.path.join(self.label_path, id+'_GTVn_CBCT2.nii.gz'))


            if self.mode == 'train':
                print('\tTraining data size: ', len(patients))
                self.list = []
                for i in range(len(patients)):
                    sub_list = []
                    # print(len(ct))
                    # print(i)
                    
                    sub_list.append(dose[i])
                    sub_list.append(ct[i])
                    sub_list.append(ct_gtv[i])
                    sub_list.append(ct_gtvn[i])
                    sub_list.append(cbct1[i])
                    sub_list.append(cbct1_gtv[i])
                    sub_list.append(cbct1_gtvn[i])
                    sub_list.append(cbct2[i])
                    sub_list.append(cbct2_gtv[i])
                    sub_list.append(cbct2_gtvn[i])
                    self.list.append(tuple(sub_list))

                    with open('data_loader_training.txt', 'a') as file_1:
                        file_1.write(f"ID: {str(i)}, path: {str(ct[i])}\n") # for debug

            elif self.mode == 'val':
                print('\tValidation data size: ', len(patients))
                self.list = []
                for i in range(len(patients)):
                    sub_list = []
                    sub_list.append(dose[i])
                    sub_list.append(ct[i])
                    sub_list.append(ct_gtv[i])
                    sub_list.append(ct_gtvn[i])
                    sub_list.append(cbct1[i])
                    sub_list.append(cbct1_gtv[i])
                    sub_list.append(cbct1_gtvn[i])
                    sub_list.append(cbct2[i])
                    sub_list.append(cbct2_gtv[i])
                    sub_list.append(cbct2_gtvn[i])
                    self.list.append(tuple(sub_list))

                    with open('data_loader_validation.txt', 'a') as file_1:
                        file_1.write(f"ID: {str(i)}, path: {str(ct[i])}\n") # for debug


    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        if self.mode == 'test':
            print('will do')
        else:
            # dataset specific normalization

            # loading_time = time.time()
            dose, ct, ct_gtv, ct_gtvn, cbct1, cbct1_gtv, cbct1_gtvn, cbct2, cbct2_gtv, cbct2_gtvn = [self.read_data(self.list[index][i]) for i in range(10)] #number of images
            dose /= dose.max()
            dose = (dose*2.0-1.0)*500.

            img = np.stack((dose, ct, cbct1, cbct2), axis=-1).astype(np.float32)
            img /= 500.
            # print('img.shape', img.shape)
            
            gtv = np.stack((ct_gtv, ct_gtvn, cbct1_gtv, cbct1_gtvn, cbct2_gtv, cbct2_gtvn), axis=-1).astype(np.float32)
            gtv /= 255

            # print('gtv.shape', gtv.shape)

            
            #augmentation
            if self.mode == 'train' and self.augmentation==True:
                img, gtv = self.transform(img, gtv)

            img=torch.from_numpy(img.astype(np.float32).copy())
            img = img.permute(3, 0, 1, 2)  
            # print('img.shape', img.shape)
            
            gtv=torch.from_numpy(gtv.astype(np.float32).copy())
            gtv = gtv.permute(3, 0, 1, 2)  
            # print('gtv.shape', gtv.shape)

            return img, gtv

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        if return_numpy:
            return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))
        return sitk.ReadImage(str(path_to_nifti))