import torch
from torch import nn
from lib.Models import resnet, resnet_class, resnet_fork


def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.in_modality == 3 and opt.is_classi:
            print('input CBCT, CT, Dose as 3 channels for classification')

            if opt.model_depth == 10:
                model = resnet_fork.resnet10(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 18:
                model = resnet_fork.resnet18(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 34:
                model = resnet_fork.resnet34(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 50:
                model = resnet_fork.resnet50(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 101:
                model = resnet_fork.resnet101(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 152:
                model = resnet_fork.resnet152(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 200:
                model = resnet_fork.resnet200(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)

        elif opt.is_classi:
            print('Generating classification model!')

            if opt.model_depth == 10:
                model = resnet_class.resnet10(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 18:
                model = resnet_class.resnet18(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 34:
                model = resnet_class.resnet34(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 50:
                model = resnet_class.resnet50(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 101:
                model = resnet_class.resnet101(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 152:
                model = resnet_class.resnet152(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
            elif opt.model_depth == 200:
                model = resnet_class.resnet200(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_classes=opt.n_classes)
        else:
            print('Generating segmentation model!')

            if opt.model_depth == 10:
                model = resnet.resnet10(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_seg_classes=opt.n_seg_classes)
            elif opt.model_depth == 18:
                model = resnet.resnet18(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_seg_classes=opt.n_seg_classes)
            elif opt.model_depth == 34:
                model = resnet.resnet34(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_seg_classes=opt.n_seg_classes)
            elif opt.model_depth == 50:
                model = resnet.resnet50(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_seg_classes=opt.n_seg_classes)
            elif opt.model_depth == 101:
                model = resnet.resnet101(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_seg_classes=opt.n_seg_classes)
            elif opt.model_depth == 152:
                model = resnet.resnet152(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_seg_classes=opt.n_seg_classes)
            elif opt.model_depth == 200:
                model = resnet.resnet200(
                    sample_input_W=opt.input_W,
                    sample_input_H=opt.input_H,
                    sample_input_D=opt.input_D,
                    shortcut_type=opt.resnet_shortcut,
                    no_cuda=opt.no_cuda,
                    num_seg_classes=opt.n_seg_classes)

    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path and opt.is_transfer:
        print('Loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        # print('New_layer_names: {}'.format(opt.new_layer_names))
        # print(type(opt.new_layer_names))
        for name in opt.new_layer_names:
            print(name)
        for pname, p in model.named_parameters():
            # print('Named_parameters: {}'.format(pname))
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    # print('Append {} to new_parameters'.format(pname))
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
