import cv2
import skimage
from skimage.measure import label
import numpy as np
import copy


def peak_point(output_i, thresh,bp_shresh):


    points=[]

    min_d = 6

    output_i = cv2.GaussianBlur(output_i, (25, 25), 0)

    bp = copy.deepcopy(output_i)
    output_i[output_i < 0.5] = 0


    #np.save('prob_map.npy',output_i)

    coordinates = skimage.feature.peak_local_max(output_i, min_distance=min_d, exclude_border=6 // 2)  # N by 2

    for p in coordinates:
        flag = 0
        for cell in points:
            flag=0
            for item in cell:
                if np.sum(np.abs(p-item))<thresh:
                    flag =1

            if flag:
                cell.append(p)
                break
        if not flag:
            points.append([p])

    points_output=[]
    for cell in points:
        point = np.mean(np.asarray(cell),axis=0).astype('uint16')
        points_output.append(point)


    bp[bp<bp_shresh] = 0
    background = np.ones_like(bp) *255
    background[bp==0] = 0

    return np.asarray(points_output), background, output_i