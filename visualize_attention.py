import argparse
import pdb

import cv2
import numpy as np
import torch
import timm
import os
from PIL import Image
from torchvision.transforms import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from sklearn.metrics import f1_score
# from model.model_res2net_shallow import res2net_2class




def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path',type=str,
                        default='/data2/chenpy/point_seg/Public_MoNuSeg/MoNuSegTestData/TCGA-44-2665-01B-06-BS6.tif',
                        help='Input image path')
    parser.add_argument('--checkpoint-path',type=str,
                        default='/data2/chenpy/point_seg/self_supervised_seg/checkpoint/ipt_299.pth',
                        # default = '/data2/Caijt/vit-pytorch/TransCrowd/save_file/BCD_baseline/negative_swin_918/model_best.pth',
                        help='checkpoint path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth',action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')
    parser.add_argument('--method',type=str,
                        default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
    """
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model = timm.create_model('res2net101_26w_4s' ,pretrained=False).cuda()
    #model = torch.nn.DataParallel(model, device_ids=[0])

    #checkpoint = torch.load(args.checkpoint_path, map_location={'cuda:0':'cuda:2'})
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)


    if args.use_cuda:
        model = model.cuda()
    model.eval()
    # target_layer = model.module.pos_atten_2class3
    target_layer_pos = model.layer1

    #target_layer_neg = model.module.layer5
    # target_layer =torch.nn.Sequential(target_layer.conv1, target_layer.bn1, target_layer.convs, target_layer.bns,
    #                                   target_layer.conv3, target_layer.bn3, target_layer.relu)


    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam_pos = methods[args.method](model=model,
                               target_layers=target_layer_pos,
                               use_cuda=args.use_cuda,
                               )

    # transform = transforms.Compose([transforms.Resize((224, 224)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #transform = transforms.Compose([transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop(1000)])

    image = Image.open(args.image_path).convert('RGB')

    image = np.asarray(image)
    image = Image.fromarray(np.uint8(image))
    img = transform(image)
    cam_pos.batch_size = 1

    grayscale_cam_pos = cam_pos(input_tensor=img.unsqueeze(0).cuda(),
                        target_category=0,
                        eigen_smooth=args.eigen_smooth,
                       aug_smooth=args.aug_smooth)


    # Here grayscale_cam has only one image in the batch

    # grayscale_cam = 1-(grayscale_cam_pos[0, :] * 0.4 + grayscale_cam_neg[0, :] *0.6)



    # print(np.max(grayscale_cam), np.min(grayscale_cam))

    # pdb.set_trace()
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    #rgb_img = rgb_img[244:756,244:756]
    rgb_img = np.float32(rgb_img) / 255

    cam_image = show_cam_on_image(rgb_img, grayscale_cam_pos)

    gt_img = cv2.imread('/data2/chenpy/point_seg/Public_MoNuSeg/MoNuSegTestData/TCGA-44-2665-01B-06-BS6_binary.png', 0)
    #gt_img = gt_img[244:756, 244:756]
    f1 = f1_score(np.uint8(gt_img/255).reshape((-1)), np.uint8(cam_image/255).reshape((-1)))
    print(f1)
    #cam_image = cv2.kmeans(,2)
    save_dir = "/data2/chenpy/point_seg/self_supervised_seg/share"
    os.makedirs(save_dir, exist_ok= True)
    cv2.imwrite(os.path.join(save_dir,os.path.basename(args.image_path)[:-4]+'pos.png'), cam_image)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(args.image_path)[:-4] + 'original.png'), np.uint8(rgb_img*255))
    cv2.imwrite(os.path.join(save_dir, os.path.basename(args.image_path)[:-4] + 'gt.png'), gt_img)