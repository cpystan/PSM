import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor


def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def sgg(img: np.ndarray,
                      mask: np.ndarray,
                      name: str,
                      beta: float,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    mask = mask.transpose(1,2,0)


    heatmap = cv2.applyColorMap(np.uint8(255 * (mask)) ,colormap)





    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = 1*heatmap + beta*img
    cam_color = np.uint8(255*((heatmap+beta*img)/np.max(heatmap+beta*img)))


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, output, centers = cv2.kmeans(255*cam.reshape((cam.shape[0]*cam.shape[1],3)),3,None,criteria,10,flags)
    output = output.reshape((cam.shape[0],cam.shape[1]))

    mask1 = np.expand_dims((output == 0),axis=2)
    ave_vector1= np.sum(mask1*cam*255,axis=(0,1))/np.sum(mask1)

    mask2 = np.expand_dims((output == 1),axis=2)
    ave_vector2= np.sum(mask2*cam*255,axis=(0,1))/np.sum(mask2)

    mask3 = np.expand_dims((output == 2),axis=2)
    ave_vector3= np.sum(mask3*cam*255,axis=(0,1))/np.sum(mask3)

    #color prior
    ave_feature= np.array([ave_vector1[1]+ave_vector1[0],ave_vector2[1]+ave_vector2[0],ave_vector3[1]+ave_vector3[0]])
    vectors = np.array([mask1,mask2,mask3])
    sorted_indices = np.argsort(ave_feature)
    #ave_features = vectors[sorted_indices,:,:,:]
    fg_id = sorted_indices[0]
    output = vectors[fg_id,:,:,:].squeeze(2)*1


    #import sys
    #sys.exit()
    #heatmap = 1-heatmap

    #cam_threshold = cv2.adaptiveThreshold(cam_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,15)
    
    # = cam / np.max(cam)
  
    return output, cam_color

    #heatmap[heatmap<2] =0
    #return np.uint8(127.5*heatmap)

