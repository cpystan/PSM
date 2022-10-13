import argparse

parser = argparse.ArgumentParser(description='parameters')


parser.add_argument('--mode', type=str, default='generate_label',
                    help='train or test')
parser.add_argument('--resume_path', type=str, default='None',
                  help='check point path')
parser.add_argument('--method', type=str, default='gradcam',
                    help='method')
# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--set_gpu', type=str, default='0',
                    help='which gpu to deploy')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--cudnn', type=str, default='True',
                    help='set cudnn')

# Data specifications
parser.add_argument('--data_train', type=str, default='/data2/chenpy/point_seg/Public_MoNuSeg/MoNuSeg 2018 Training Data',
                    help='train dataset')
parser.add_argument('--data_test', type=str, default='/data2/chenpy/point_seg/Public_MoNuSeg/MoNuSegTestData',
                    help='test dataset')
parser.add_argument('--crop_edge_size', type=int, default=512,
                    help='crop edge size')
parser.add_argument('--scale', type=int, default=1,
                    help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')

# Model specifications
parser.add_argument('--model', default='./checkpoint/self_stage_net_best.pth',
                    help='model name')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_interval', type=int, default=10,
                    help='do test per every N epoch')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--crop_batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='./checkpoint/',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

# transformer
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=288)
parser.add_argument('--patch_dim', type=int, default=16)
parser.add_argument('--num_heads', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--no_norm', action='store_true')
parser.add_argument('--freeze_norm', action='store_true')
parser.add_argument('--post_norm', action='store_true')
parser.add_argument('--no_mlp', action='store_true')
parser.add_argument('--pos_every', action='store_true')
parser.add_argument('--no_pos', action='store_true')
parser.add_argument('--num_queries', type=int, default=1)


args, unparsed = parser.parse_known_args()



for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False