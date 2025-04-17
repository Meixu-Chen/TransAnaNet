'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default='./data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--short_note',
        default='',
        type=str,
        help='short note in model saving name')
    parser.add_argument(
        '--img_list',
        default='./data/train.txt',
        type=str,
        help='Path for image list file')
    parser.add_argument(
        '--train_dataset_path',
        default='./data',
        type=str,
        help='Path to training images')
    parser.add_argument(
        '--train_label_path',
        default='./data',
        type=str,
        help='Path to training labels')
    parser.add_argument(
        '--val_dataset_path',
        default='./data',
        type=str,
        help='Path to validation images')
    parser.add_argument(
        '--val_label_path',
        default='./data',
        type=str,
        help='Path to validation labels')
    parser.add_argument(
        '--test_dataset_path',
        default='./data',
        type=str,
        help='Path to test images')
    parser.add_argument(
        '--test_label_path',
        default='./data',
        type=str,
        help='Path to test labels')
    parser.add_argument(
        '--in_modality',
        default=1,
        type=int,
        help="number of imaging modalities in the input"
    )
    parser.add_argument(
        '--is_classi',
        default=False,
        type=bool,
        help="Transfer learning or not"
    )
    parser.add_argument(
        '--is_transfer',
        default=False,
        type=bool,
        help="Transfer learning or not"
    )
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help="Number of classification classes"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--gamma',  # set to 0.001 when finetune
        default=0.95,
        type=float,
        help=
        'Initial gamma')
    parser.add_argument(
        '--num_workers',
        default=0,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
        default=56,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=448,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=448,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.'
    )
    parser.add_argument(
        '--pretrain_path',
        default='pretrain/resnet_50.pth',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--new_layer_names',
        #default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=['conv_seg'],
        nargs='+',
        # type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument(
        '--cuda', action='store_true', help='If true, cuda is used.')
    parser.set_defaults(no_cuda=False, cuda=True)

    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.')
    parser.add_argument(
        '--augmentation', default='false', help='if use augmentations during training')

    parser.add_argument(
        '--resnet_lr_factor',
        default=0,
        type=float,
        help='slightly train pretrained weights; should be less than 1')

    parser.add_argument(
        '--class_weights',
        default="[1.0, 1.0]",
        type=str,
        help='class weights for loss function; shape needs to match number of classes')

    # added for excluding mrns of patients
    parser.add_argument(
        '--exclude_mrn',
        default='false',
        type=str,
        help='should we exclude mrns of selected patients')
    parser.add_argument(
        '--exclude_mrn_path',
        default=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/exclude_mrn_feeding_tube_days/',
        type=str,
        help='Path exclude mrn files')
    parser.add_argument(
        '--exclude_mrn_filename',
        default='less_than_30_days.csv',
        type=str,
        help='file to load which holds list of mrns to exclude')
    parser.add_argument(
        '--fraction_key_path',
        default='./fraction_key_path/',
        type=str,
        help='Path fraction key')

    parser.add_argument(
        '--min_max_key_path',
        default='./fraction_min_max_key_path/',
        type=str,
        help='Path min max key')

    # added for clinical data
    parser.add_argument(
        '--clinical_data_path',
        default='/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Data/ft_data_5/clinical_data',
        type=str,
        help='PATH TO CLINICAL DATA')
    parser.add_argument(
        '--clinical_data_filename',
        default='processed_unnormalized_data_350.csv',
        type=str,
        help='CLINICAL DATA FILENAME')
    parser.add_argument(
        '--selected_clinical_features',
        default=['Age at Diagnosis', 'Clinical N Stage','Clinical T Stage', 'Concurrent Chemo Regimen_cat_4', \
                 'Ethnicity_cat_1', 'ae_prior_select_date'],
        help='LIST OF SELECTED CLINICAL FEATURES')

    parser.add_argument(
        '--clinical_model_type',
        default='lr',
        type=str,
        help='select clinical model to use (only LR and SVM at the moment')

    # CV parameter
    parser.add_argument(
        '--cv_num',
        default=1,
        type=int,
        help='number of cross validation performed in the image/clinic model')

    parser.add_argument(
        '--do_normalization',
        default='false',
        type=str,
        help='number of cross validation performed in the image/clinic model')
    parser.add_argument(
        '--aug_percent',
        default=0.05,
        type=float,
        help='number of cross validation performed in the image/clinic model')


    args = parser.parse_args()
    args.save_folder = "./trails/{}models/{}_{}".format(args.short_note, args.model, args.model_depth)
    
    print(args)
    return args
