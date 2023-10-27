import argparse
import pdb
import copy
import cv2
import numpy as np
import torch
import timm
import os
from PIL import Image
from torchvision.transforms import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
from pytorch_grad_cam.utils.image import sgg
from torchvision.models import resnet50
from sklearn.metrics import f1_score
from skimage import morphology
# from model.model_res2net_shallow import res2net_2class




def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def psm_for_seg(x,y,model,args,tag):
    """
    modified from grad-cam
    """

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


    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    target_layer_pos = model.layer1

    cam_pos = methods[args.method](model=model,
                               target_layers=target_layer_pos,
                               use_cuda=False,
                               )

    #
    # activation map fusion module to generate coarse segmentation
    if tag == 'test_set':
        path = args.data_test + '/' + x
    elif tag == 'train_set':
        path = args.data_train + '/Tissue Images/' + x
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop(512)])

    image = Image.open(path).convert('RGB')

    image = np.asarray(image)
    image = Image.fromarray(np.uint8(image))
    img = transform(image)

    cam_pos.batch_size = 1

    grayscale_cam_pos = cam_pos(input_tensor=img.unsqueeze(0).cuda(),
                        target_category=0,
                        eigen_smooth=False,
                       aug_smooth=False)


    rgb_img = cv2.imread(path)[:, :, ::-1]
    rgb_img = rgb_img[244:756,244:756]
    rgb_img = np.float32(rgb_img) / 255

    cam_image,cam_color= sgg(rgb_img, grayscale_cam_pos,x,1)

    cam_images = copy.deepcopy(cam_image)
    cam_images = morphology.remove_small_objects(cam_images, 200)
    cam_images = morphology.remove_small_holes(cam_images)
    cam_image_positive = cam_images*255





    #####
    if tag == 'test_set':
        save_dir = "./data_second_stage_test"
    elif tag == 'train_set':
        save_dir = './data_second_stage_train'
    os.makedirs(save_dir, exist_ok= True)

    basename = x.split('.')[0]


    cv2.imwrite(os.path.join(save_dir,basename+'_pos.png'), cam_image_positive)
    cv2.imwrite(os.path.join(save_dir, basename + '_heat.png'), cam_color)
    cv2.imwrite(os.path.join(save_dir, basename + '_original.png'), np.uint8(rgb_img*255))
    if tag == 'test_set':
        gt_img = cv2.imread(os.path.join(args.data_test,basename+'_binary.png'),cv2.IMREAD_GRAYSCALE)
        gt_img = gt_img[244:756,244:756]
        cv2.imwrite(os.path.join(save_dir, basename + '_gt.png'), gt_img)
    return
