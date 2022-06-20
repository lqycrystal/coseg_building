import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .layers import *
from .efficientnet import EfficientNet
#from .correlation_package.correlation import Correlation
from collections import OrderedDict

#############################################################
# DOCS Encoder
#
class DOCSEncoderNet(nn.Module):
    def __init__(self, features):
        super(DOCSEncoderNet, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)

encoder_archs = {
    'vgg16-based-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 1024, 1024]
}

def make_encoder_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    output_scale = 1.0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            output_scale /= 2.0
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2,inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.2,inplace=True)]
            in_channels = v
    return nn.Sequential(*layers), in_channels, output_scale

#############################################################

#############################################################
# DOCS Decoder
#
class DOCSDecoderNet(nn.Module):
    def __init__(self, features):
        super(DOCSDecoderNet, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)

decoder_archs = {
    'd16': [1024, 'd512', 512, 512, 'd512', 512, 512, 'd256', 256, 256, 'd128', 128, 128, 'd64', 64, 64, 'c2']
}

def make_decoder_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if type(v) is str:
            if v[0] == 'd':
                v = int(v[1:])
                convtrans2d = nn.ConvTranspose2d(in_channels, v, kernel_size=4, stride=2, padding=1)
                if batch_norm:
                    layers += [convtrans2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                else:
                    layers += [convtrans2d, nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                in_channels = v
            elif v[0] == 'c':
                v = int(v[1:])
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

#############################################################


#############################################################
# DOCS Network
#
class DOCSNet(nn.Module):
    '''DOCSNet a Siamese Encoder-Decoder for Object Co-segmentation.'''
    def __init__(self, input_size=512, init_weights=True, batch_norm=False, 
                 en_arch='vgg16-based-16', de_arch='d16',
                 has_squeez=True, squeezed_out_channels=512):
        super(DOCSNet, self).__init__()

        self.en_arch = en_arch
        self.de_arch = de_arch

        en_layers, en_out_channels, en_output_scale = make_encoder_layers(encoder_archs[en_arch], batch_norm)
        self.encoder = DOCSEncoderNet(en_layers)
        en_output_size = round(input_size * en_output_scale)

        disp = en_output_size-1
        self.corr = Correlation(pad_size=disp, kernel_size=1, max_displacement=disp, stride1=1, stride2=1)
        corr_out_channels = self.corr.out_channels

        self.has_squeez = has_squeez
        if has_squeez:
            self.conv_squeezed = nn.Conv2d(en_out_channels, squeezed_out_channels, 1, padding=0)
            de_in_channels = int(squeezed_out_channels + corr_out_channels)
        else:
            de_in_channels = int(en_out_channels + corr_out_channels)

        de_layers = make_decoder_layers(decoder_archs[de_arch], de_in_channels, batch_norm)
        self.decoder = DOCSDecoderNet(de_layers)

        if init_weights:
            self._initialize_weights()
        self.ml = nn.LogSoftmax(dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_a, img_b, softmax_out=True):
        self.en_a = self.encoder(img_a)
        self.en_b = self.encoder(img_b)

        self.corr_ab = self.corr(self.en_a, self.en_b)
        self.corr_ba = self.corr(self.en_b, self.en_a)

        if self.has_squeez:
            cat_a = torch.cat((self.conv_squeezed(self.en_a), self.corr_ab),dim=1)
            cat_b = torch.cat((self.conv_squeezed(self.en_b), self.corr_ba),dim=1)
        else:
            cat_a = torch.cat((self.en_a, self.corr_ab),dim=1)
            cat_b = torch.cat((self.en_b, self.corr_ba),dim=1)

        self.out_a = self.decoder(cat_a)
        self.out_b = self.decoder(cat_b)

        if softmax_out:
            self.out_a =self.ml(self.out_a)
            self.out_b = self.ml(self.out_b)

        return self.out_a, self.out_b
def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True,in_channels=3):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, self.size[4])
            self.double_conv_input = double_conv(self.size[4]+in_channels, self.size[5])
        else:
            self.up_conv_input = up_conv(64, self.size[4])
            self.double_conv_input = double_conv(self.size[4], self.size[5])       
        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)
        

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x0,x):
        input_ = x0

        blocks = get_blocks_to_be_concat(self.encoder, x0)
        _, x1 = blocks.popitem()

        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)
        else:
            x = self.up_conv_input(x)
            x = self.double_conv_input(x)
        x = self.final_conv(x)
        x = self.softmax(x)
        return x
#############################################################

def featureL2Norm(feature):
        epsilon = 1e-6
            #        print(feature.size())
                #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
class DOCSNeteunet(nn.Module):
    '''DOCSNet a Siamese Encoder-Decoder for Object Co-segmentation.'''
    def __init__(self,pretrained=True,in_channels=3,out_channels=2,has_squeez=False, squeezed_out_channels=512,concat_input=True):
        super(DOCSNeteunet, self).__init__()

        self.encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained,in_channels=in_channels)
        en_output_size = 8
        #disp = en_output_size-1
        #self.corr = Correlation(pad_size=disp, kernel_size=1, max_displacement=disp, stride1=1, stride2=1)
        corr_out_channels =  en_output_size* en_output_size
        en_out_channels=1280
        self.has_squeez = has_squeez
        if has_squeez:
            self.conv_squeezed = nn.Conv2d(en_out_channels, squeezed_out_channels, 1, padding=0)
            de_in_channels = int(squeezed_out_channels + corr_out_channels)
        else:
            de_in_channels = int(en_out_channels + corr_out_channels)

        
        self.decoder = EfficientUnet(self.encoder, out_channels=out_channels, concat_input=concat_input,in_channels=in_channels)
        self.ReLU=nn.ReLU()
        self.linear_e = nn.Linear(en_out_channels,en_out_channels,bias = False)
        self.linear_e2 = nn.Linear(en_out_channels,en_out_channels,bias = False)


    def forward(self, img_a, img_b):
        self.en_a = self.encoder(img_a)
        self.en_b = self.encoder(img_b)
        b,c,h,w=self.en_a.size()
        feature_a = self.en_a.view(b,c,h*w) # size [b,c,h*w]
        feature_b = self.en_b.view(b,c,h*w).transpose(1,2)
        feature_mul = torch.bmm(self.linear_e(feature_b),feature_a)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        self.corr_ab = featureL2Norm(self.ReLU(correlation_tensor))
        

        feature_a2 = self.en_a.view(b,c,h*w).transpose(1,2)# size [b,c,h*w]
        feature_b2 = self.en_b.view(b,c,h*w)
        feature_mul2 = torch.bmm(self.linear_e2(feature_a2),feature_b2)
        #print(feature_mul2.size())
        correlation_tensor2 = feature_mul2.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        self.corr_ba = featureL2Norm(self.ReLU(correlation_tensor2))

        #print(self.corr_ab.size())
        #self.corr_ba = self.corr(self.en_b, self.en_a)

        if self.has_squeez:
            cat_a = torch.cat((self.conv_squeezed(self.en_a), self.corr_ba),dim=1)
            cat_b = torch.cat((self.conv_squeezed(self.en_b), self.corr_ab),dim=1)
        
        else:
            cat_a=torch.bmm(self.corr_ab.view(b,h*w,h*w),feature_a2)
            cat_b=torch.bmm(self.corr_ba.view(b,h*w,h*w),feature_b)
            
        self.out_a = self.decoder(img_a,cat_a.view(b,c,h,w))
        self.out_b = self.decoder(img_b,cat_b.view(b,c,h,w))

        return self.out_a, self.out_b
