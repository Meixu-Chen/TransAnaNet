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


"""Original author Mike, small modifications by Aixa 
This code takes the validation, outcome and train paths to return lists of all the files. Using that list, it returns the images as pytorch tensors. 
Img is input the train image, label is the ground truth. For Mike's code he used one dimensional labels as ground truth  """


class FEEDTUBE(Dataset):
    """
    Code for reading the data from dataset in this research project: 
    Dohopolski, Michael, et al. "Use of deep learning to predict the need for aggressive nutritional supplementation during head and neck radiotherapy." Radiotherapy and Oncology 171 (2022): 129-138.
    """

    def __init__(self, args, mode, dataset_path='./datasets', label_path='./datasets', classes=2):
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
        #     self.transform = augment3D.RandomChoice(
        #         transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip()], p=0.9)
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(), augment3D.RandomRotation(min_angle=-5, max_angle=5)], p=0.9)
                # ,
                #             augment3D.RandomRotation(min_angle=-5, max_angle=5)

        if self.mode == 'test':
            print('will do in the future')
     
        else:
            print("self root",self.root)
            image = glob.glob(os.path.join(self.root, '*.npy'))
            print(f"\tLoading {self.mode} data from", os.path.join(self.root, '*.npy'))
            # print('\n\n')

            
            #print("\n image",image)
            # print('\n\n')
            labels = []

            # loading_time = time.time()
            for path in image:
                #to find the MRN of the patient (its )
                # print("\n print path",path.split('\\'))
                #print("\n print path",path.split('_'))

                # print("\nimage path",image)
                temp = path.split('\\')[-1]#splits path by \: Example path="hey\hi\ha", path.split('\\')=["hey","hi","ha"]"

                #temp = temp.split('_')[0]
                temp = temp.split('.')[0]

                print(temp)

                #print("self label path",self.label_path)

                labels += glob.glob(os.path.join(self.label_path, temp+'.npy'))
                # print("label 1",os.path.join(self.label_path, temp+'.npy'))
                # print("label 2",os.path.join(self.label_path, temp+'.npy'))

            # print(labels)

            if self.mode == 'train':
                print('\tTraining data size: ', len(image))
                self.list = []
                for i in range(len(image)):
                    sub_list = []
                    sub_list.append(image[i])
                    sub_list.append(labels[i])
                    self.list.append(tuple(sub_list))

                    with open('data_loader_training.txt', 'a') as file_1:
                        file_1.write(f"ID: {str(i)}, path: {str(image[i])}\n")

            elif self.mode == 'val':
                print('\tValidation data size: ', len(image))
                self.list = []
                for i in range(len(image)):
                    print('path of {}nd image is '.format(i) + str(image[i]))

                    with open('data_loader.txt', 'a') as file_2:
                        file_2.write(f"ID: {str(i)}, path: {str(image[i])}\n")
                    sub_list = []
                    sub_list.append(image[i])
                    sub_list.append(labels[i])
                    self.list.append(tuple(sub_list))
                # print(self.list)


    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        if self.mode == 'test':
            print('will do')
        else:
            f_img, f_label = self.list[index]
            # name = self.list[index]
            # print("\n",name)


            # loading_time = time.time()
            img, lab = np.load(f_img), np.load(f_label)

            # img = img[::2,::2,::2,:]
            # lab = lab[::2,::2,::2]
            # img = img[:,:,24:88,:]
            # lab = lab[:,:,24:88]
            img = img[::2,::2,24:88:2,:]
            lab = lab[::2,::2,24:88:2]
            
            #augmentation
            if self.mode == 'train' and self.augmentation==True:
                img, lab = self.transform(img, lab)
                # img=np.clip(img,0,None)
                # lab=np.clip(lab,0,None)

                # f_name = 'img_aug.npy'#save image #f I want to check which images are loaded
                # np.save(f_name, img)
                # f_name = 'lab_aug.npy'#save image
                #np.save("name", lab)
                # f_name = 'name_aug.npy'
                # np.save(f_name, name)

                # with open('data_loader.txt', 'a') as file_1:
                #     file_1.write(f"\tname: {name}\n")
                # print("\n min, max values of ground truth channels")
                # print(np.min(lab),np.max(lab))   
                # print("\n min, max values of 3 channels")            
                
                # print("CBCT",np.min(img[:,:,:,0]),np.max(img[:,:,:,0])) 
                # print("CT",np.min(img[:,:,:,1]),np.max(img[:,:,:,1])) 

                # print("\n dose min, max values")             

                # print(np.min(img[:,:,:,2]),np.max(img[:,:,:,2])) 
            # 
            if not np.size(lab) == 1:
                lab=np.expand_dims(lab, axis=3)
                # np.expand_dims() # need to expand (262,262,185, 1)
                #lab=np.array(lab[2:,2:,1:,:])
                #lab= zoom(lab,(0.504,0.504,0.625,1))
                # lab= zoom(lab,(0.977,0.977,0.923,1)) stopped in 3
                #lab= zoom(lab,(0.5,0.5,0.5,1))
                #lab= zoom(lab,(0.977,0.977,0.518,1))
                

                #print("before permutation",np.shape(lab))

                #print("empty chanel",lab[0,0,0,0])
                #lab = torch.Tensor(lab)
                lab=torch.from_numpy(lab.copy())
                # lab=torch.from_numpy(np.flip(lab,axis=0).copy())
                lab = lab.permute(3, 2, 0, 1)
            else:
                pass
            #img=np.array(img[2:,2:,1:,:])
            #img= zoom(img,(0.5,0.5,0.5,1))
            #img=zoom(img,(0.504,0.504,0.625,1))
            # img=zoom(img,(0.977,0.977,0.923,1))

            #img=zoom(img,(0.977,0.977,0.518,1))
            #print("before permutation",np.shape(img))
            #img = torch.FloatTensor(img)
            img=torch.from_numpy(img.copy())
            # img=torch.from_numpy(np.flip(img,axis=0).copy())


            # Need to confirm w/ Kai but I believe this adjust the x,y,z, channel axis

            img = img.permute(3, 2, 0, 1)  

            return img, lab
            # return img[:,16:64+16,:], lab[:,16:64+16,:]

