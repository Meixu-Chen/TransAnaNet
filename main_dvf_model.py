# Python libraries
import argparse, os
import datetime

# Pytorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader


from lib.pytorch_ssim import SSIM3D_Loss
from lib.losses3D import Grad
from lib.losses3D import DiceLoss

from lib.Loss import dice
#from lib.Loss import dice as dice_mike

# lib files
import lib.utils as utils
import lib.Loading as medical_loaders
from lib.Loading.feedtube import FEEDTUBE
from lib.Loading.howard import HOWARD
import lib.medzoo as medzoo
import lib.Trainers.pytorch_trainer_howard as pytorch_trainer

import numpy as np
from scipy.ndimage import zoom


from lib import pytorch_ssim
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 17
torch.manual_seed(seed)


# try to address speed issues?
torch.backends.cudnn.benchmark = True


def main():
    time_stamp = datetime.datetime.now()
    print("Time stamp " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), '\n\n')

    print("Arguments Used")
    args = get_arguments()
    print(args)
    print(args.info)
    print('')

    print(f"Setting seed for reproducibility\n\tseed: {seed}")
    utils.general.reproducibility(args, seed)
    print(f"Creating saved folder: {args.save}")
    utils.general.make_dirs(args.save)

    print("\nCreating custom training and validation generators")
    print(f"\tIs data augmentation being utilized?: {args.augmentation}")
    print(f"\tBatch size: {args.batchSz}")

    training_dataset = HOWARD(args, mode='train', dataset_path=args.train_path, label_path=args.label_path)
    training_generator = DataLoader(training_dataset, batch_size=args.batchSz, shuffle=True, pin_memory=True, num_workers=8)
                                   # num_workers=1)

    validation_dataset = HOWARD(args, mode='val', dataset_path=args.val_path, label_path=args.label_path)
    val_generator = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=8)
    # num_workers=1)

    # Setting model and optimizer
    print('')
    model, optimizer = medzoo.create_model(args)
    if args.resume is not None:
        model.restore_checkpoint(args.resume)

    print(model)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 20)
    scheduler = MultiStepLR(optimizer, milestones=[100, 250, 400], gamma=0.1)

    criterion_pre=SSIM3D_Loss()
    criterion_gtv=DiceLoss()
    criterion_smooth=Grad()

    print("\n dimensions before error",np.shape(torch.device('cuda')))
    criterion_pre = criterion_pre.to(torch.device('cuda'))
    criterion_gtv = criterion_gtv.to(torch.device('cuda'))
    criterion_smooth = criterion_smooth.to(torch.device('cuda'))
    print(f"\nCurrent loss function: {criterion_pre}\n")
    print(f"\nCurrent loss function: {criterion_gtv}\n")
    print(f"\nCurrent loss function: {criterion_smooth}\n")
    print("Assessing GPU usage")
    if args.cuda:
        print(f"\tCuda set to {args.cuda}\n")
        model = model.to(torch.device('cuda'))

    print("Initializing training")
    trainer = pytorch_trainer.Trainer(args, model, criterion_pre, criterion_gtv, criterion_smooth, optimizer, train_data_loader=training_generator,\
        valid_data_loader=val_generator, lr_scheduler=scheduler)

    #print(trainer)

    print("Start training!")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()


    parser.add_argument('--train_path', default="../crop_normed_data/with_np_dose/input_train/", type=str,
                        metavar='PATH',
                        help='path to training data')
        
    parser.add_argument('--val_path', default="../crop_normed_data/with_np_dose/input_val/", type=str, metavar='PATH',
                        help='path to validation/testing data')
    parser.add_argument('--label_path', default="../crop_normed_data/with_np_dose/label/", type=str, metavar='PATH',
                        help='path to training/validation/testing label')

    parser.add_argument('--test', default=False)
    parser.add_argument('--dataset_name', type=str, default="Howard")

    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=500)
    parser.add_argument('--inChannels', type=int, default=7)
    parser.add_argument('--inModalities', type=int, default=1) #useless

    parser.add_argument('--input_W', type=int, default=128)
    parser.add_argument('--input_H', type=int, default=128)
    parser.add_argument('--input_D', type=int, default=32)
    
    parser.add_argument('--classes', type=int, default=2) #useless
    parser.add_argument('--lamda', default=1e-2, type=float,
                        help='penalty for smoothness')#before 2e-2

    parser.add_argument('--terminal_show_freq', default=1)
    parser.add_argument('--augmentation', action='store_true', default=True)

    parser.add_argument('--normalization', default='global_mean', type=str,
                        help='Tensor normalization: options max, mean, global')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
                        #to restart 1e-2 change it later*******

    parser.add_argument('--cuda', action='store_true', default=True)

    parser.add_argument('--loadData', default=True)
 
    parser.add_argument('--model', type=str, default='TRANSMORPH',
                        choices=('V_YNET', 'V_YNET2', 'U_YNET3D', 'U_YNET3D_2', 'U_YNET3D_3', 'AlexNet', 'AlexNet2', 'ResNet18','ResNet50',
                        'ResNet18_2', 'ResNet50_2', 'MobileNetV2', 'MobileNetV2_2', 'ShuffleNetV2', 'SqueezeNet','TRANSMORPH','NNFORMER', 'VoxelMorph','VitMorph'))

    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop', 'adamax'))
    parser.add_argument('--log_dir', type=str,
                        default='UNET/') # was originally AlexNet
    parser.add_argument('--info', type=str,
                        default='\n\nCode Initializing....')

    args = parser.parse_args()

    args.save = 'Output/saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}'.format(
        utils.general.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
