from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np
from options import args
from utils.optimization import make_optimizer
from utils.generate_point_label import  peak_point
from utils.generate_voronoi import  create_Voronoi_label
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import os
import cv2
from tqdm import tqdm
import csv
from model.resunet import ResUNet34
from model.DeepLab_V3plus.deeplab import DeepLab
from model.TransUnet.vit_seg_modeling import VisionTransformer,CONFIGS
from model.unet import UNet
from skimage import io
import timm
import skimage
import torchvision.transforms as transforms
from metrics import metrics
import psm
import warnings
import random
warnings.filterwarnings('ignore')

writer = SummaryWriter()
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.set_gpu
EPOCH = args.epochs

def save_model(dict, epoch, name):
    if not os.path.exists('checkpoint_monu'):
        os.makedirs('checkpoint_monu')

    torch.save(dict, f'checkpoint_monu/{name}_{epoch}.pth')

def crop(x, size):
    if not size:
        return x
    a, c, h, w = x.shape
    H = int(h / 2 - size / 2)
    W = int(w / 2 - size / 2)
    o = x[:, :, H:H + size, W:W + size]
    o = o.contiguous()
    return o

def crop_random(x,size):
    a, c, h, w = x.shape
    xmin = random.randint(0,h-size)
    ymin = random.randint(0,w-size)

    o = x[:,:,xmin:xmin+size,ymin:ymin+size]
    o = o.contiguous()
    return o

def preProcess_train(x ,y,args):
    input = []
    label = []
    edge = []
    for i in range(len(x)):
        x_data = x[i]
        y_data = y[i]

        assert x_data[:16] == y_data[:16]
        path1 = args.data_train + '/Tissue Images/' +x_data
        path2 = args.data_train + '/Annotations/' +y_data
        if args.mode in ('train_second_stage' , 'generate_voronoi', 'train_final_stage'):
            path1 = '/'.join(args.data_train.split('/')[:-1])+ '/data_second_stage_train/' + x_data
            path2 = '/'.join(args.data_train.split('/')[:-1])+ '/data_second_stage_train/' + y_data
            path_edge = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train/' + y_data.split('_')[-2] + '_edge.png'
            if args.mode == 'train_final_stage':
                path_edge = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train/' + y_data.split('_')[-2] + '_vor.png'

        x_data ,y_data = io.imread(path1) , io.imread(path2)
        x_data, y_data = torch.from_numpy(x_data).unsqueeze(0) , torch.from_numpy(y_data).unsqueeze(0).unsqueeze(3)
        input.append(x_data)
        label.append(y_data)
        if args.mode in ('train_second_stage' , 'generate_voronoi', 'train_final_stage'):
            edge_data = io.imread(path_edge)
            edge_data = torch.from_numpy(edge_data).unsqueeze(0).unsqueeze(3)
            edge.append(edge_data)
    input ,label = torch.cat(input,0) , torch.cat(label,0)
    input ,label = input.permute(0,3,1,2).contiguous().float().cuda() , label.permute(0,3,1,2).contiguous().float().cuda()
    input ,label = crop(input,args.crop_edge_size), crop(label, args.crop_edge_size)

    if args.mode in ('train_second_stage' , 'generate_voronoi', 'train_final_stage'):
        edge_data = torch.cat(edge)
        edge_data = edge_data.permute(0,3,1,2).contiguous().float().cuda()
        edge_data = crop(edge_data,args.crop_edge_size)
        if args.mode == 'train_final_stage':

            label[label==255] =2
            edge_data[edge_data ==0 ]=2
            edge_data[edge_data==255] =0
            edge_data[edge_data==120] =1
            return input, label, edge_data
        return input,label/255,edge_data/255

    return input,label/255


def preProcess_test(x ,y,args):
    input = []
    label = []
    for i in range(len(x)):
        x_data = x[i]
        y_data = y[i]
        assert x_data[:16] == y_data[:16]
        path1 = args.data_test+ '/' +x_data
        path2 = args.data_test + '/' +y_data
        if args.mode in ('train_second_stage' , 'generate_voronoi', 'train_final_stage'):
            path1 = '/'.join(args.data_test.split('/')[:-1])+ '/data_second_stage_test/' +x_data
            path2 = '/'.join(args.data_test.split('/')[:-1])+ '/data_second_stage_test/' +y_data
        x_data ,y_data = io.imread(path1) , io.imread(path2)
        x_data, y_data = torch.from_numpy(x_data).unsqueeze(0) , torch.from_numpy(y_data).unsqueeze(0).unsqueeze(3)
        input.append(x_data)
        label.append(y_data)
    input ,label = torch.cat(input,0) , torch.cat(label,0)
    input ,label = input.permute(0,3,1,2).contiguous().float().cuda() , label.permute(0,3,1,2).contiguous().float().cuda()
    input ,label = crop(input,args.crop_edge_size), crop(label, args.crop_edge_size)
    return input,label/255


def trainer_selfsupervised(EPOCH,args,model, loss_func, optimizer):
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)
        for x, y in tqdm(loader.train_loader, desc='trianing'):
            x, y = preProcess_train(x, y , args)
            # print(x.shape)
            x_rotate = transforms.functional.rotate(x,180)

            output1 = model(x)
            output2 = model(x_rotate)

            loss = torch.sum(torch.abs(output1-output2))/output1.shape[0]


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        writer.add_scalar('train/mean_loss', loss_mean, epoch)
        print(F'loss:{loss_mean}')

def trainer_selfsupervised_contrastive(EPOCH,args,model, loss_func, optimizer):
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)
        for x, y in tqdm(loader.train_loader, desc='trianing'):
            x, y = preProcess_train(x, y , args)
            # print(x.shape)
            anchor = x[:3,:,:,:]
            positive = transforms.functional.rotate(anchor, 180)
            negative = x[3:,:,:,:]

            output1 = model(anchor.cuda().float())
            output2 = model(positive.cuda().float())
            output3 = model(negative.cuda().float())

            loss_sim = loss_func(output2, output1)
            loss_cont = loss_func(output3, output1)
            loss = torch.max( loss_sim - loss_cont + 10,torch.tensor([0]).cuda())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        writer.add_scalar('train/mean_loss', loss_mean, epoch)
        print(F'loss:{loss_mean}')
        if (epoch + 1) % args.test_interval == 0:

            save_model(model.state_dict(), epoch, args.model)

def trainer_selfsupervised_random_rotate(EPOCH,args,model, loss_func, optimizer):
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)
        for x, y in tqdm(loader.train_loader, desc='trianing'):
            x, y = preProcess_train(x, y , args)
            # print(x.shape)
            n= random.randint(0,3)
            x_rotate = transforms.functional.rotate(x,90*n)

            output2 = model(x_rotate)

            loss = loss_func(output2,torch.tensor([n]).cuda().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        writer.add_scalar('train/mean_loss', loss_mean, epoch)
        print(F'loss:{loss_mean}')
        if (epoch + 1) % args.test_interval == 0:

            save_model(model.state_dict(), epoch, args.model)

def trainer_selfsupervised_simsiam(EPOCH,args,model, loss_func, optimizer):
    encoder = model
    predictor1 = nn.Linear(1000,1).cuda()
    predictor2 = nn.Linear(1000,1).cuda()
    model_whole = nn.Sequential(encoder,predictor1)
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)
        for x, y in tqdm(loader.train_loader, desc='trianing'):
            x, y = preProcess_train(x, y , args)
            # print(x.shape)
            n= random.randint(0,3)
            x_rotate = transforms.functional.rotate(x,90*n)
            x , x_rotate = x.cuda(),x_rotate.cuda()

            x1,x2 = encoder(x),encoder(x_rotate)
            z1,z2 = predictor1(x1), predictor1(x2)
            p1,p2 = predictor2(x1), predictor2(x2)

            def D(p,z): #negative cosine similarity
                z = z.detach()

                return torch.abs(p-z).sum(axis=1).mean()



            loss = D(p1,z2)/2 + D(p2,z1)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        writer.add_scalar('train/mean_loss', loss_mean, epoch)
        print(F'loss:{loss_mean}')
        if (epoch + 1) % args.test_interval == 0:

            save_model(model_whole.state_dict(), epoch, args.model)

def trainer_selfsupervised_mean_value(EPOCH,args,model, loss_func, optimizer):
    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loader = data.get_monuseg(epoch, args)  # 十折交叉验证
        for x, y in tqdm(loader.train_loader, desc='trianing'):
            x, y = preProcess_train(x, y , args)
            # print(x.shape)
            n= torch.mean(torch.reshape(x,(x.shape[0],-1)),dim=1).unsqueeze(1)

            output = model(x)

            loss = sum(torch.abs(output-n))/len(output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)

        loss_mean = sum(loss_list) / len(loss_list)
        writer.add_scalar('train/mean_loss', loss_mean, epoch)
        print(F'loss:{loss_mean}')
        if (epoch + 1) % args.test_interval == 0:

            save_model(model.state_dict(), epoch, args.model)


def trainer_second_stage(EPOCH,args,model, loss_func, optimizer):
    best_aji = 0
    best_epoch = 0
    best_model = 0

    for epoch in range(EPOCH):
        print(F'--EPOCH : {epoch} ')
        loss_list = []
        loss_seg =[]
        loss_edge = []
        loader = data.get_monuseg(epoch, args)  # 十折交叉验证
        model.train()

        for x, y in tqdm(loader.train_loader, desc='trianing'):
            x, y ,edge = preProcess_train(x, y , args)
            io.imsave('try0.png',x[0].cpu().permute(1,2,0).numpy().astype('uint8'))
            io.imsave('try1.png',(y[0]*255).cpu().permute(1,2,0).numpy().astype('uint8'))
            io.imsave('try2.png',(edge[0]*120).cpu().permute(1,2,0).numpy().astype('uint8'))

            x = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(x/255)

            '''
            path = r'/data2/chenpy/point_seg/Public_MoNuSeg/data_second_stage_train/TCGA-XS-A8TJ-01Z-00-DX1_original.png'
            x = io.imread(path)
            x = torch.from_numpy(x).unsqueeze(0).permute(0,3,1,2).contiguous().float().cuda()
            path = r'/data2/chenpy/point_seg/Public_MoNuSeg/data_second_stage_train/TCGA-XS-A8TJ-01Z-00-DX1_pos.png'
            y = io.imread(path)
            y = torch.from_numpy(y).unsqueeze(0).unsqueeze(3).permute(0,3,1,2).contiguous().float().cuda()
            '''
            output = model(x)

            prob_maps = F.softmax(output,dim=1)
            log_prob_maps = F.log_softmax(prob_maps, dim=1)


            loss1 = loss_func(log_prob_maps,y.long().squeeze(1))
            loss2 = loss_func(log_prob_maps,edge.long().squeeze(1))


            loss_all = loss1 + 1.5 * loss2

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            loss_list.append(loss_all)
            loss_seg.append(loss1)
            loss_edge.append(loss2)


        loss_mean = sum(loss_list) / len(loss_list)
        loss_seg_mean = sum(loss_seg)/len(loss_seg)
        loss_edge_mean = sum(loss_edge)/len(loss_edge)
        writer.add_scalar('train/mean_loss', loss_mean, epoch)
        writer.add_scalar('train/mean_seg_loss', loss_seg_mean, epoch)
        writer.add_scalar('train/mean_edge_loss', loss_edge_mean, epoch)
        print(F'seg_loss: {loss_seg_mean}, edge_loss: {loss_edge_mean}, all_loss: {loss_mean}')

        if (epoch + 1) % args.test_interval == 0:

            print('validating:')
            print('-----------------------')
            model.eval()
            with torch.no_grad():
                loss_list = []

                dic=[]

                for x0, y0 in tqdm(loader.test_loader, desc='testing'):
                    x1,y = preProcess_test(x0, y0, args)
                    x= transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x1/255)

                    output = model(x)

                    prob_maps = F.softmax(output, dim=1)
                    pred = np.argmax(prob_maps.cpu(), axis=1)


                    for i in range(pred.shape[0]):
                        path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_test/' + y0[i]
                        label = io.imread(path)
                        metric = metrics.compute_metrics(pred[i],label/255,['p_F1','aji','iou'])
                        #cv2.imwrite('./show2/' + x0[i],pred[i].cpu().numpy()*255)
                        '''
                        point_list = peak_point(prob_maps.cpu()[i][1].numpy(),20)
                        
                        
                        #generate voronoilabel
                        voronoi_label = create_Voronoi_label(point_list,label.shape)
                        cv2.imwrite('./show/' + x0[i].split('_')[-2] + '_vor.png', voronoi_label)

                        fig_for_save = x1[i].cpu().permute(1,2,0).contiguous().numpy()
                        for j in range(point_list.shape[0]):
                            cv2.line(fig_for_save, (point_list[j][1]-3,point_list[j][0]),(point_list[j][1]+3,point_list[j][0]),color=(0,0,255),thickness=1)
                            cv2.line(fig_for_save, (point_list[j][1], point_list[j][0]-3), (point_list[j][1], point_list[j][0]+3),color=(0,0,255),thickness=1)

                        cv2.imwrite('./show/' + x0[i], fig_for_save)
                        '''
                        dic.append(metric)


                for key in dic[0].keys():
                    num = sum([i[key] for i in dic ])/len(dic)
                    var = np.var(np.array([i[key] for i in dic ]))
                    print(F'{key}: {num} var: {var}')
                    writer.add_scalar(F'{key}',num,epoch)

                    if key== 'aji':
                        if num>best_aji:
                            best_aji=num
                            best_epoch = epoch
                            import copy
                            best_model = copy.deepcopy(model.state_dict())
                #loss_mean = sum(loss_list) / len(loss_list)
                #.add_scalar('val/mean_loss', loss_mean, epoch)

        writer.add_scalar('lr', optimizer.get_lr(), epoch)
        optimizer.schedule()
    save_model(best_model, best_epoch, args.model)

    print(f'best-epoch: {best_epoch}, aji: {best_aji}')

def test_stage(EPOCH, args, model):
    loader = data.get_monuseg(0, args)
    dic=[]
    for x0, y0 in tqdm(loader.test_loader, desc='testing'):
        x1, y = preProcess_test(x0, y0, args)
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x1 / 255)

        output = model(x)

        prob_maps = F.softmax(output, dim=1)
        pred = np.argmax(prob_maps.detach().cpu(), axis=1)

        for i in range(pred.shape[0]):
            path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_test/' + y0[i].replace('binary','gt')
            label = y[i].squeeze(0).cpu().numpy().astype(np.uint8)
            #metric = metrics.compute_metrics(pred[i], label / 255, ['p_F1', 'aji', 'iou'])

            img_save = pred[i].detach().cpu().numpy().astype(np.uint8)
            img_save_label = skimage.measure.label(img_save)
            label = skimage.measure.label(label)

            img_save = (skimage.color.label2rgb(img_save_label)*255).astype(np.uint8)
            label_save = (skimage.color.label2rgb(label)*255).astype(np.uint8)
            for m in range(img_save_label.shape[0]):
                for n in range(img_save_label.shape[1]):
                    if img_save_label[m,n]==0:
                        img_save[m,n,0],img_save[m,n,1],img_save[m,n,2]=0,0,0
                    if label[m,n]==0:
                        label_save[m, n, 0], label_save[m, n, 1], label_save[m, n, 2] = 0, 0, 0

            cv2.imwrite('./show2/' + x0[i].replace('tif','png'), img_save)
            cv2.imwrite('./show2/' + x0[i].replace('.tif','_label.png'), label_save)
            cv2.imwrite('./show2/' + x0[i].replace('.tif','_gt.png'),x1[i].permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
            #dic.append(metric)
    '''
    for key in dic[0].keys():
        num = sum([i[key] for i in dic]) / len(dic)
        var = np.var(np.array([i[key] for i in dic]))
        print(F'{key}: {num} var: {var}')
    '''


def generate_voronoi_label( args, model):
    loader = data.get_monuseg(0, args)
    model.eval()
    for sub_loader in [loader.train_loader,loader.val_loader,loader.test_loader]:
        for x0, y0 in tqdm(sub_loader, desc='testing'):
            if sub_loader == loader.test_loader:
                x1,y = preProcess_test(x0,y0,args)
            else:
                x1, y ,edge = preProcess_train(x0, y0, args)
            x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x1 / 255)

            output = model(x)

            prob_maps = F.softmax(output, dim=1)
            pred = np.argmax(prob_maps.cpu().detach().numpy(), axis=1)

            for i in range(pred.shape[0]):
                if sub_loader == loader.test_loader:
                    path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_test/' + y0[i]
                else:
                    path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train/' + y0[i]
                label = io.imread(path)
                #metric = metrics.compute_metrics(pred[i], label / 255, ['p_F1', 'aji', 'iou'])

                point_list,bp ,prob= peak_point(prob_maps.detach().cpu()[i][1].numpy(), 20,0.6)

                # generate voronoilabel
                if point_list.shape[0]<4:
                   point_list = np.random.randint(0,255,(4,2))
                voronoi_label = create_Voronoi_label(point_list, label.shape)
                if sub_loader == loader.test_loader:
                    cv2.imwrite('../Public_MoNuSeg/' +'data_second_stage_test/' + x0[i].split('_')[-2] + '_vor.png', voronoi_label)


                else:
                    cv2.imwrite('../Public_MoNuSeg/' +'data_second_stage_train/' + x0[i].split('_')[-2] + '_vor.png', voronoi_label)


                fig_for_save = x1[i].cpu().permute(1, 2, 0).contiguous().numpy()
                for j in range(point_list.shape[0]):
                    cv2.line(fig_for_save, (point_list[j][1] - 3, point_list[j][0]),
                             (point_list[j][1] + 3, point_list[j][0]), color=(0, 0, 255), thickness=1)
                    cv2.line(fig_for_save, (point_list[j][1], point_list[j][0] - 3),
                             (point_list[j][1], point_list[j][0] + 3), color=(0, 0, 255), thickness=1)

                cv2.imwrite('./share/' + x0[i].split('_')[-2] + '_point.png' , fig_for_save)
                cv2.imwrite('./share/' + x0[i].split('_')[-2] + '_prob.png' , prob*200)

                #print('here')

    print('end')

def fully_supervised(EPOCH, args, model,optimizer,loss_func):
    loader = data.get_monuseg(0, args)

    for epoch in range(EPOCH):
        #loader = data.get_ten_fold_data_monuseg(epoch, args)
        for x0, y0 in tqdm(loader.train_loader, desc='training'):
            x1, y = preProcess_train(x0, y0, args)
            x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x1 / 255)

            output = model(x)
            '''
            t= x1[0].cpu().numpy().transpose(1,2,0)
            io.imsave('show0.png',x1[0].cpu().numpy().transpose(1,2,0))
            io.imsave('show1.png', (y[0].cpu().numpy()*255).transpose(1,2,0))
            import sys
            sys.exit()
            '''

            prob_maps = F.softmax(output, dim=1)
            log_prob_maps = F.log_softmax(prob_maps, dim=1)


            loss = loss_func(log_prob_maps,y.long().squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % args.test_interval == 0:
            dic = []
            for x0, y0 in tqdm(loader.test_loader, desc='testing'):
                x1, y = preProcess_test(x0, y0, args)
                x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x1 / 255)

                output = model(x)

                prob_maps = F.softmax(output, dim=1)
                pred = np.argmax(prob_maps.detach().cpu(), axis=1)

                for i in range(pred.shape[0]):
                    #path = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_test/' + y0[i].replace('binary','gt')
                    label = y[i].cpu().permute(1,2,0).squeeze(2).long()
                    metric = metrics.compute_metrics(pred[i], label , ['p_F1', 'aji', 'iou'])
                    cv2.imwrite('./show3_resunet34/' + x0[i], pred[i].detach().cpu().numpy() * 255)
                    dic.append(metric)

            for key in dic[0].keys():
                num = sum([i[key] for i in dic]) / len(dic)
                var = np.var(np.array([i[key] for i in dic]))
                print(F'{key}: {num} var: {var}')




if __name__ == "__main__":
    #loader = data.Data(args)


    #if args.check_path:
     #   model.load_state_dict(torch.load(args.check_path))
    if args.mode == 'train_base':
        args.batch_size=1
        model = timm.create_model('res2net101_26w_4s', pretrained=False).cuda()
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised(EPOCH,args,model, loss_func, optimizer)

    elif args.mode == 'train_contrastive':
        model = timm.create_model('res2net101_26w_4s', num_classes=1,pretrained=False).cuda()
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_contrastive(EPOCH,args,model, loss_func, optimizer)

    elif args.mode == 'train_random_rotate':
        model = timm.create_model('res2net101_26w_4s', pretrained=False).cuda()
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_random_rotate(EPOCH,args,model, loss_func, optimizer)

    elif args.mode == 'train_simsiam':
        model = timm.create_model('res2net101_26w_4s', pretrained=False).cuda()

        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_simsiam(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'train_mean_value':
        model = timm.create_model('res2net101_26w_4s', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features,1)
        model = model.cuda()
        optimizer = make_optimizer(args, model)
        loss_func = nn.MSELoss()
        model.train()
        trainer_selfsupervised_mean_value(EPOCH, args, model, loss_func, optimizer)



    elif args.mode == 'generate_label':
        print('testing:')
        print('-----------------------')
        model = timm.create_model('res2net101_26w_4s', num_classes=1,pretrained=True).cuda()

        model.load_state_dict(torch.load(args.model))
        optimizer = make_optimizer(args, model)

        loader = data.get_monuseg(0, args)
        model.eval()

        loss_list = []

        for step, (x, y) in enumerate(loader.test_loader):
            for i in range(len(x)):
                psm.psm_for_seg(x[i],y[i],model,args,'test_set')




        for step, (x, y) in enumerate(loader.train_loader):
            for i in range(len(x)):
                psm.psm_for_seg(x[i],y[i],model,args,'train_set')





        for step, (x, y) in enumerate(loader.val_loader):
            for i in range(len(x)):
                psm.psm_for_seg(x[i],y[i],model,args,'train_set')


        print('end')





    elif args.mode == 'train_second_stage':
        model = ResUNet34(pretrained=True).cuda()
        optimizer = make_optimizer(args, model)
        #loader = data.get_ten_fold_data_monuseg(0, args)
        loss_func = torch.nn.NLLLoss(ignore_index=2).cuda()
        trainer_second_stage(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'generate_voronoi':
        model = ResUNet34(pretrained=True).cuda()
        model.load_state_dict(torch.load('/data2/chenpy/point_seg/self_supervised_seg/checkpoint_monu/net_19.pth'))
        generate_voronoi_label(args, model)

    elif args.mode == 'train_final_stage':
        model = ResUNet34(pretrained=True).cuda()
        optimizer = make_optimizer(args, model)
        loss_func = torch.nn.NLLLoss(ignore_index=2).cuda()
        trainer_second_stage(EPOCH, args, model, loss_func, optimizer)

    elif args.mode == 'test':
        model = ResUNet34(pretrained=True).cuda()
        best_path = '/data2/chenpy/point_seg/self_supervised_seg/checkpoint_monu/beta=infinity.pth'
        model.load_state_dict(torch.load(best_path))
        model.eval()
        test_stage(EPOCH, args, model)

    elif args.mode == 'fully-supervised':
        model = ResUNet34(pretrained=True).cuda()
        #model = VisionTransformer(CONFIGS['R50-ViT-B_16'],img_size=256).cuda()
        #model = UNet(n_channels=3,n_classes=2).cuda()
        #model = DeepLab(num_classes=2).cuda()
        #best_path = '/data2/chenpy/point_seg/self_supervised_seg/checkpoint/final_stage_net_aji: 0.5213.pth'
        #model.load_state_dict(torch.load(best_path))
        optimizer = make_optimizer(args, model)
        loss_func = torch.nn.NLLLoss(ignore_index=2).cuda()
        model.train()
        fully_supervised(EPOCH, args, model,optimizer,loss_func)

    else:
        raise NotImplementedError(F"process mode  be train/tes, not {args.mode}.")

