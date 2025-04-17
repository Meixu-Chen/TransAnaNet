import json
import os
import random
import shutil
import time
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def reproducibility(args, seed):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    # FOR FASTER GPU TRAINING WHEN INPUT SIZE DOESN'T VARY
    # LET'S TEST IT
    cudnn.benchmark = True


def save_arguments(args, path):
    with open(path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()


def datestr():
    now = time.gmtime()
    return '{:02}_{:02}___{:02}_{:02}'.format(now.tm_mday, now.tm_mon, now.tm_hour, now.tm_min)


def shuffle_lists(*ls, seed=777):
    l = list(zip(*ls))
    random.seed(seed)
    random.shuffle(l)
    return zip(*l)

def cv_lists(image, label, growth, cv=1):
    # l = list(zip(*ls))
    length_img = len(image)
    length_label = len(label)
    length_growth = len(growth)
    assert length_img==length_label, "Problem reading data. Check the data paths."
    assert length_img==length_growth, "Problem reading data. Check the data paths."

    split = int((cv-1)*0.2 * length_img)
    image = image[split:]+image[:split]
    label = label[split:]+label[:split]
    growth = growth[split:]+growth[:split]
    return image, label, growth


def prepare_input(input_tuple, inModalities=-1, inChannels=-1, cuda=False, args=None):
    if args is not None:
        # modalities = args.inModalities
        # channels = args.inChannels
        in_cuda = args.cuda
    else:
        # modalities = inModalities
        # channels = inChannels
        in_cuda = cuda
    input_tensor, target = input_tuple

    # print('looking at in_cuda: ', in_cuda)

    if in_cuda:

        input_tensor, target = input_tensor.to(torch.device('cuda')), target.to(torch.device('cuda'))

    return input_tensor, target


def prepare_test_input(input_tuple, inModalities=-1, inChannels=-1, cuda=False, args=None):
    if args is not None:
        modalities = args.inModalities
        channels = args.inChannels
        in_cuda = args.cuda
    else:
        modalities = inModalities
        channels = inChannels
        in_cuda = cuda
    if modalities == 1:
        input_tensor, name = input_tuple

    if in_cuda:
        input_tensor = input_tensor.to(torch.device('cuda'))

    return input_tensor, name


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)


def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)
    return list_file
