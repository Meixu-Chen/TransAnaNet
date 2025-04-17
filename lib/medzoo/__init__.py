import torch.optim as optim

from .COVIDNet import CovidNet, CNN
from .DenseVoxelNet import DenseVoxelNet
from .Densenet3D import DualPathDenseNet, DualSingleDenseNet, SinglePathDenseNet
from .HighResNet3D import HighResNet3D
from .HyperDensenet import HyperDenseNet, HyperDenseNet_2Mod
from .ResNet3DMedNet import generate_resnet3d
from .ResNet3D_VAE import ResNet3dVAE
from .SkipDenseNet3D import SkipDenseNet3D
from .Unet2D import Unet
from .Unet3D import UNet3D
from .Vnet import VNet, VNetLight
from .TransMorph import TransMorph_Model
from .VoxelMorph import VxmDense as VoxelMorph
from .VitMorph import VitMorph
import lib.medzoo.nnformer.nnFormer_synapse as nnFormer

model_list = ['UNET3D', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
              "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN",
              "HIGHRESNET",'TRANSMORPH','NNFORMER','VoxelMorph','VitMorph']


def create_model(args):
    model_name = args.model
    assert model_name in model_list
    optimizer_name = args.opt
    lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001
    print("Building Model . . . . . . . ." + model_name)

    if model_name == 'VNET2':
        model = VNetLight(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'VNET':
        model = VNet(in_channels=in_channels, elu=False, classes=num_classes)
    elif model_name == 'UNET3D':
        model = UNet3D(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
    elif model_name == 'VitMorph':
        model = VitMorph(img_size=[args.input_D, args.input_H, args.input_W], in_channels = 7, out_channels = 3, feature_size = 16)
    elif model_name == 'VoxelMorph':
        model = VoxelMorph(inshape=(args.input_D, args.input_H, args.input_W), src_feats=4, trg_feats=3)
    elif model_name == 'TRANSMORPH':
        model = TransMorph_Model(img_size=(args.input_D, args.input_H, args.input_W), patch_size=4, in_chans=in_channels, embed_dim=96*4)
    elif model_name == 'NNFORMER':
        model = nnFormer.nnFormer(crop_size=[args.input_D, args.input_H, args.input_W],patch_size=[2,4,4], \
            deep_supervision = False, embedding_dim=96, input_channels=in_channels, num_classes=num_classes)
    elif model_name == 'DENSENET1':
        model = SinglePathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET2':
        model = DualPathDenseNet(in_channels=in_channels, classes=num_classes)
    elif model_name == 'DENSENET3':
        model = DualSingleDenseNet(in_channels=in_channels, drop_rate=0.1, classes=num_classes)
    elif model_name == "UNET2D":
        model = Unet(in_channels, num_classes)
    elif model_name == "RESNET3DVAE":
        model = ResNet3dVAE(in_channels=in_channels, classes=num_classes, dim=args.dim)
    elif model_name == "SKIPDENSENET3D":
        model = SkipDenseNet3D(growth_rate=16, num_init_features=32, drop_rate=0.1, classes=num_classes)
    elif model_name == "COVIDNET1":
        model = CovidNet('small', num_classes)
    elif model_name == "COVIDNET2":
        model = CovidNet('large', num_classes)
    elif model_name == "CNN":
        model = CNN(num_classes, 'resnet18')
    elif model_name == "HYPERDENSENET":
        if in_channels == 2:
            model = HyperDenseNet_2Mod(classes=num_classes)
        elif in_channels == 3:
            model = HyperDenseNet(classes=num_classes)
        else:
            raise NotImplementedError
    elif model_name == "DENSEVOXELNET":
        model = DenseVoxelNet(in_channels=in_channels, classes=num_classes)
    elif model_name == "HIGHRESNET":
        model = HighResNet3D(in_channels=in_channels, classes=num_classes)
    elif model_name == "RESNETMED3D":
        depth = 18
        model = generate_resnet3d(in_channels=in_channels, classes=num_classes, model_depth=depth)

    print(model_name, 'Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, optimizer
