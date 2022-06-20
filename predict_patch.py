import torch
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#import sys
#sys.path.insert(0, '/work/shi/DeepLearning')
import argparse
#from tiramisu_de import FCDenseNet67
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
#from PIL import Image
#import cv2
#import h5py
from sklearn.metrics import confusion_matrix
from skimage import io
from models.docs import DOCSNeteunet 
class generateDataset(Dataset):

        def __init__(self, dirFiles,img_size,colordim,isTrain=True):
                self.isTrain = isTrain
                self.dirFiles = dirFiles
                self.nameFiles = [name for name in os.listdir(dirFiles) if os.path.isfile(os.path.join(dirFiles, name))]
                self.numFiles = len(self.nameFiles)
                self.img_size = img_size
                self.colordim = colordim
                print('number of files : ' + str(self.numFiles))
                
        def __getitem__(self, index):
                filename = self.dirFiles + self.nameFiles[index]
                imgf=io.imread(filename)
                imgf=imgf/255.0
                imga=imgf[:,0:256,:]
                imgb=imgf[:,256:512,:]
                img=np.concatenate((imga,imgb),axis=2)
                img = torch.from_numpy(img).float()
                img = img.transpose(0, 1).transpose(0, 2)
                imgName, imgSuf = os.path.splitext(self.nameFiles[index])
                return img, imgName
        
        def __len__(self):
                return int(self.numFiles)
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
def main(args):
    if args.cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")
    num_class=args.num_class
    if args.id==0:
      model = DOCSNeteunet(in_channels=3,out_channels=2)
      #FCDenseNet67(in_channels=args.colordim,n_classes=num_class)   
    if args.cuda:
      model=model.cuda()
    model.load_state_dict(torch.load(args.pretrain_net))
    model.eval()
    predDataset = generateDataset(args.pre_root_dir, args.img_size, args.colordim, isTrain=False)
    predLoader = DataLoader(dataset=predDataset, batch_size=args.predictbatchsize, num_workers=args.threads)
    with torch.no_grad():
      cm_w = np.zeros((2,2))
      for batch_idx, (batch_x, batch_name) in enumerate(predLoader):
        batch_x1 = batch_x[:,0:3,:,:]
        batch_x2 = batch_x[:,3:6,:,:]
        #batch_x=map01(batch_x)
        if args.cuda:
            batch_x1 = batch_x1.cuda()
            batch_x2 = batch_x2.cuda()
        
        out_a,out_b= model(batch_x1,batch_x2)
        pred_prop, pred_label = torch.max(out_a, 1)
        pred_prop_np = pred_prop.cpu().numpy()
        pred_label_np = pred_label.cpu().numpy() 
        print(len(batch_name))       
        for id in range(len(batch_name)):
                pred_label_single = pred_label_np[id, :, :]
                predLabel_filename = args.preDir +  batch_name[id] + '.png'
                io.imsave(predLabel_filename, pred_label_single.astype(np.uint8))
                label_filename= args.label_root_dir +  batch_name[id] + '.png'
                label = io.imread(label_filename)
                label =label[:,0:256]
                cm = confusion_matrix(label.ravel(), pred_label_single.ravel())
                print(cm)
                cm_w = cm_w + cm
                #OA_s, F1_s, IoU_s = evaluate(cm)
                #print('OA_s = ' + str(OA_s) + ', F1_s = ' + str(F1_s) + ', IoU = ' + str(IoU_s))
        
      print(cm_w)      
      OA_w, F1_w, IoU_w = evaluate(cm_w)
      print('OA_w = ' + str(OA_w) + ', F1_w = ' + str(F1_w) + ', IoU = ' + str(IoU_w))

def evaluate(cm):

        UAur=float(cm[1][1])/float(cm[1][0]+cm[1][1])
        UAnonur=float(cm[0][0])/float(cm[0][0]+cm[0][1])
        PAur=float(cm[1][1])/float(cm[0][1]+cm[1][1])
        PAnonur=float(cm[0][0])/float(cm[1][0]+cm[0][0])
        OA=float(cm[1][1]+cm[0][0])/float(cm[1][0]+cm[1][1]+cm[0][0]+cm[0][1])
        F1=2*UAur*PAur/(UAur+PAur)
        IoU=float(cm[1][1])/float(cm[1][0]+cm[1][1]+cm[0][1])
        
        return OA, F1, IoU


# Prediction settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0,type=int,
                        help="a name for identifying the model")
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--predictbatchsize', default=1,type=int,
                        help="input batch size per gpu for prediction")
    parser.add_argument('--threads', default=1,type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--img_size', default=256,type=int,
                        help="image size of the input")
    parser.add_argument('--seed', default=123,type=int,
                        help="random seed to use")
    parser.add_argument('--colordim', default=3,type=int,
                        help="color dimension of the input image") 
    parser.add_argument('--pretrain_net', default='./checkpoint-batchsize4-learning_rate0.01-optimizersgd/best_model.pth',
                        help='path of saved pretrained model')                       
    parser.add_argument('--pre_root_dir', default='./vdata/val/data/',
                        help='path of input datasets for predict')
    parser.add_argument('--label_root_dir', default='./vdata/val/seg/',
                        help='path of label of input datasets')
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--preDir', default='./predictionv/',
                        help='path of result')
    args = parser.parse_args()

    if not os.path.isdir(args.preDir):
        os.makedirs(args.preDir)
    main(args)
