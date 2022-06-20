import argparse
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from torch.autograd import Variable
from torch.utils.data import DataLoader
from BSDdataloader import get_training_set,get_test_set
import torchvision
import re
import functools
from PIL import Image, ImageStat
from skimage import io
import time
from models.docs import DOCSNet,DOCSNeteunet
from distributed import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
def map01(tensor):
    #input/output:tensor
    maxa=np.copy(tensor.numpy())
    mina=np.copy(tensor.numpy())
    maxa[:,0,:,:]=255.0
    maxa[:,1,:,:]=255.0
    maxa[:,2,:,:]=255.0
    mina[:,0,:,:]=0.0
    mina[:,1,:,:]=0.0
    mina[:,2,:,:]=0.0
    return torch.from_numpy( (tensor.numpy() - mina) / (maxa-mina) )
    
def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()
    imPred += 1
    imLab += 1
    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
    intersection, bins=numClass, range=(1, numClass))
    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)

def test(args,model,testing_data_loader1,optimizer):
    totalloss = 0
    acc_meter1 = AverageMeter()
    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()
    acc_meter2 = AverageMeter()
    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    testloader_iter1 = iter(testing_data_loader1)

    
    for i_iter in range(len(testing_data_loader1)):
        #optimizer.zero_grad()
        batch1=testloader_iter1.next()
        input1 = Variable(batch1[0])[:,0:3,:,:]
        target1 = Variable(batch1[1])[:,0,:,:]
        input2 = Variable(batch1[0])[:,3:6,:,:]
        target2 = Variable(batch1[1])[:,1,:,:]
        criterion = nn.CrossEntropyLoss() 
        if args.cuda:
            input1 = input1.cuda()
            target1 = target1.cuda()
            input2 = input2.cuda()
            target2 = target2.cuda()
            criterion = criterion.cuda()
        criterion = nn.NLLLoss2d() 
        if args.cuda:
            input1 = input1.cuda()
            target1 = target1.cuda()
            input2 = input2.cuda()
            target2 = target2.cuda()
            criterion = criterion.cuda()
        optimizer.zero_grad()
        model.eval()
        prediction1,prediction2 = model(input1,input2)
        target1 =target1.squeeze(1)
        target2 =target2.squeeze(1)
        loss = criterion(prediction1, target1.long())+0.00001*criterion(prediction2, target2.long())
        totalloss += loss.data
        npimgra1=target1.squeeze(0)
        npimgr1=npimgra1.cpu().numpy()     
         
        npimgra2=target2.squeeze(0)
        npimgr2=npimgra2.cpu().numpy()
        imgout1 = prediction1.data
        imgout1=imgout1.squeeze(0)      
        valuesa,imgout1a=imgout1.max(0) 
        npimg1 = imgout1a.cpu().numpy()        
        acc1, pix1 = accuracy(npimg1, npimgr1)
        intersection1, union1 = intersectionAndUnion(npimg1, npimgr1,3)
        acc_meter1.update(acc1, pix1)
        intersection_meter1.update(intersection1)
        union_meter1.update(union1)
        imgout2 = prediction2.data
        imgout2=imgout2.squeeze(0)      
        valuesb,imgout2a=imgout2.max(0) 
        npimg2 = imgout2a.cpu().numpy()        
        acc2, pix2 = accuracy(npimg2, npimgr2)
        intersection2, union2 = intersectionAndUnion(npimg2, npimgr2,3)
        acc_meter2.update(acc2, pix2)
        intersection_meter2.update(intersection2)
        union_meter2.update(union2)
        #npimgr1=np.where(npimgr1>0,255,0)
        #npimgr2=np.where(npimgr2>0,255,0)
        #path1='./a/'+str(i_iter)+'.png'
        #path2='./b/'+str(i_iter)+'.png'
        #io.imsave(path1,npimgr1.astype(np.uint8))
        #io.imsave(path2,npimgr2.astype(np.uint8))
        #input1=input1.squeeze(0)
        #npimga = input1.cpu().numpy()
        #npimga = np.transpose(npimga, (1, 2, 0))
        #npimga=255*npimga
        #filenamea='./a/'+str(i_iter)+'_sa.png'
        #io.imsave(filenamea,npimga.astype(np.uint8))
        #input2=input2.squeeze(0)
        #npimgb = input2.cpu().numpy()
        #npimgb = np.transpose(npimgb, (1, 2, 0))
        #npimgb=255*npimgb
        #filenameb='./b/'+str(i_iter)+'_sa.png'
        #io.imsave(filenameb,npimgb.astype(np.uint8))
    avg_test_loss=totalloss / len(testing_data_loader1)
    iou1 = intersection_meter1.sum / (union_meter1.sum + 1e-10)
    iou2 = intersection_meter2.sum / (union_meter2.sum + 1e-10)
    return avg_test_loss,iou1,acc_meter1,iou2,acc_meter2


def checkpoint(args,model,iteration):
    model_out_path = args.checkpoint+'/'+'best_model.pth'
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main(args):
    if args.cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    if args.cuda:
      torch.cuda.manual_seed(args.seed)
    print('===> Loading datasets')

#loading training dataset
    train_set1 = get_training_set(args.root_dataset1,args.img_size, target_mode= args.target_mode, colordim=args.colordim)
    #training_data_loader1 = DataLoader(dataset=train_set1, num_workers=args.threads, batch_size=args.trainbatchsize, shuffle=True)
    train_sampler = make_data_sampler(train_set1, shuffle=True,distributed=False)
    train_batch_sampler = make_batch_data_sampler(train_sampler, args.trainbatchsize, args.num_steps)
    training_data_loader1 = DataLoader(dataset=train_set1, num_workers=args.threads, batch_sampler=train_batch_sampler)
#loading validation dataset
    test_set1 = get_test_set(args.root_dataset1,args.img_size, target_mode= args.target_mode, colordim=args.colordim)
    testing_data_loader1 = DataLoader(dataset=test_set1, num_workers=args.threads, batch_size=args.validationbatchsize, shuffle=False)
    num_class=args.num_class
    model=DOCSNeteunet(in_channels=3,out_channels=2)
    #model = torch.nn.DataParallel(model, device_ids=[0,1])
    if args.cuda:
      model=model.cuda()
    if args.pretrained:
      model.load_state_dict(torch.load(args.pretrain_net))
      for param_tensor in model.state_dict():
       print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    lr=args.learning_rate
    optimizer = key2opt[args.optim](model.parameters(), lr=args.learning_rate)
    print('===> Training model')
    test_iou=-0.1
    #trainloader_iter = iter(training_data_loader)
    for iteration, batch1 in enumerate(training_data_loader1):
        optimizer.zero_grad()
        input1 = Variable(batch1[0])[:,0:3,:,:]
        target1 = Variable(batch1[1])[:,0,:,:]
        input2 = Variable(batch1[0])[:,3:6,:,:]
        target2 = Variable(batch1[1])[:,1,:,:]
        criterion = nn.NLLLoss2d()
        if args.cuda:
         input1 = input1.cuda()
         target1 = target1.cuda()
         input2 = input2.cuda() 
         target2 = target2.cuda()
         criterion = criterion.cuda()
        model.train()                                                                                                                
        target1 =target1.squeeze(1)
        target2 =target2.squeeze(1)
        output1,output2 = model(input1,input2)
        loss = criterion(output1, target1.long())+0.00001*criterion(output2, target2.long())
        loss.backward()
        optimizer.step()
        print(iteration)
        if (iteration % 100 ==0):
          avg_test_loss,iou1,acc_meter1,iou2,acc_meter2=test(args,model,testing_data_loader1,optimizer)
          if iou1[1]>test_iou:
             test_iou=iou1[1]
             checkpoint(args,model,iteration)
          ResultPath=args.root_result+'/accuracy.txt'
          f = open(ResultPath, 'a+')
          new_content = '%d' % (iteration) + '\t' + '%.4f' % (loss)+ '\t' + '%.4f' % (avg_test_loss) + '\t' + '%.4f' % (iou1[1])  + '\t' +'%.4f' % (iou2[1])+ '\t' +'\n'
          f.write(new_content)
          f.close()

# Training settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--trainbatchsize', default=4,type=int,
                        help="input batch size per gpu for training")
    parser.add_argument('--validationbatchsize', default=1,type=int,
                        help="input batch size per gpu for validation")
    parser.add_argument('--num_steps', default=100000,type=int,
                        help="epochs to train for")
    parser.add_argument('--learning_rate', default=0.01,type=float,
                        help="learning rate")
    parser.add_argument('--threads', default=10,type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--img_size', default=256,type=int,
                        help="image size of the input")
    parser.add_argument('--seed', default=123,type=int,
                        help="random seed to use")
    parser.add_argument('--colordim', default=3,type=int,
                        help="color dimension of the input image") 
    parser.add_argument('--pretrained', default=False,
                        help='whether to load saved trained model')
    parser.add_argument('--pretrain_net', default='./a/best_model.pth',
                        help='path of saved pretrained model')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')                            
    parser.add_argument('--root_dataset1', default='./vdata',
                        help='path of datasets')
    parser.add_argument('--optim', default='sgd',
                        help='optimizer')
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--checkpoint', default='./checkpoint',
                        help='folder to output checkpoints')
    parser.add_argument('--target_mode', default='seg',
                        help='folder (mode) of target label')
    parser.add_argument('--root_result', default='./result',
                        help='path of result')
    args = parser.parse_args()
    args.checkpoint += '-batchsize' + str(args.trainbatchsize)
    args.checkpoint += '-learning_rate' + str(args.learning_rate)
    args.checkpoint += '-optimizer' + str(args.optim)
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    if not os.path.isdir(args.root_result):
        os.makedirs(args.root_result)
    main(args)
