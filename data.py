import h5py
#from torch.utils.data import dataloader
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import os

class Dataset(data.Dataset):
    def __init__(self, x , y ):
        self.input = x
        self.label = y

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        input = self.input[item]
        label = self.label[item]
        return input , label


class Data:
    def __init__(self,args,input_train, label_train, input_val, label_val,input_test,label_test):
        self.train = args.data_train
        self.test  = args.data_test
        self.train_loader = None
        self.val_loader =None
        self.test_loader  = None



        self.train_loader = data.DataLoader(
                Dataset(input_train ,label_train),
                batch_size = args.batch_size,
                shuffle = True,
                pin_memory=False,
                num_workers=args.n_threads,
                drop_last=False
            )


        self.val_loader = data.DataLoader(
                Dataset(input_val ,label_val),
                batch_size = args.batch_size,
                shuffle = True,
                pin_memory=False,
                num_workers=args.n_threads,
            )

        self.test_loader = data.DataLoader(
                Dataset(input_test ,label_test),
                batch_size = args.batch_size,
                shuffle = True,
                pin_memory=False,
                num_workers=args.n_threads,
            )




def get_monuseg(epoch, args):
    vali_fold = epoch%10
    list_all = None
    image_list = None
    anno_list = None
    num = 0
    if not args.mode == 'train_second_stage':
        for i,j,k in os.walk(args.data_train + '/Tissue Images'):
            image_list = k[:]
            image_list = sorted(image_list)

        for i, j, k in os.walk(args.data_train + '/Annotations'):
            anno_list  = k[:]
            anno_list  = [item for item in anno_list if item.endswith('_binary.png')]
            anno_list  = sorted((anno_list))

        test_list = None
        for i, j, k in os.walk(args.data_test):
            test_list = k[:]
        input_test = sorted([item for item in test_list if item.endswith('tif')])
        label_test = sorted([item for item in test_list if item.endswith('_binary.png')])

    if args.mode in( 'train_second_stage' , 'generate_voronoi','train_final_stage'):
        path = '/'.join(args.data_train.split('/')[:-1])+ '/data_second_stage_train'
        for i , j , k in os.walk('/'.join(args.data_train.split('/')[:-1])+ '/data_second_stage_train'):
            list_all = sorted(k[:])
        image_list = sorted([item for item in list_all if item.endswith('_original.png')])

        if args.mode == 'generate_voronoi':
            anno_list = sorted([item for item in list_all if item.endswith('_pospos.png')])
        else:
            anno_list = sorted([item for item in list_all if item.endswith('_pos.png')])

        for i, j, k in os.walk('/'.join(args.data_test.split('/')[:-1])+ '/data_second_stage_test'):
            test_list = k[:]
        input_test = sorted([item for item in test_list if item.endswith('_original.png')])
        label_test = sorted([item for item in test_list if item.endswith('_gt.png')])

    num = len(image_list) // 10 + 1



    if vali_fold == 9:
        input_train, label_train, input_val, label_val = image_list[:num * vali_fold], anno_list[:num * vali_fold]\
                                                            ,image_list[num * vali_fold:], anno_list[num * vali_fold:]
    else:
        input_train, label_train, input_val, label_val = image_list[:num * vali_fold] + image_list[num * (vali_fold+1):] , \
                                                           anno_list[:num * vali_fold] + anno_list[num * (vali_fold+1):]\
                                                            ,image_list[num * vali_fold: num * (vali_fold+1)],\
                                                           anno_list[num * vali_fold:num * (vali_fold+1)]


    o = Data(args, input_train, label_train, input_val, label_val ,input_test,label_test)

    return o