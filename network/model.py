import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from utils  import *

class model_resnet18(nn.Module):
    def __init__(self):
        super(model_resnet18,self).__init__()

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        #self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        #output = self.fc(feature)
        return feature#,output

class model_resnet50(nn.Module):
    def __init__(self):
        super(model_resnet50,self).__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        #self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        feature = self.convnet(x)
        feature = feature.view(x.size(0), -1)
        #output = self.fc(feature)
        return feature#,output



class DoubleConv(nn.Module):
    def __init__(self,  in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
       
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



# directly modified from model_resnet50
class Model(nn.Module):
    def __init__(self, batch_size, n_channels=3, bilinear=True):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.bilinear = bilinear
        # the input size should be (batch*4, 3, 224, 224)
        # the output is (batch*4, 64, 224, 224)
        self.inc = DoubleConv(n_channels, 64)
        # the output is (batch*4, 128, 112, 112)
        self.down1 = Down(64, 128)
        # the output is (batch*4, 256, 56, 56)
        self.down2 = Down(128, 256)
        # the output is (batch*4, 512, 28, 28)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        # the output is (batch*4, 512, 14, 14)
        self.down4 = Down(512, 1024 // factor)
        # the output is (batch*4, 256, 28, 28)
        self.up1 = Up(1024, 512 // factor, bilinear)
        # the output is (batch*4, 128, 56, 56)
        self.up2 = Up(512, 256 // factor, bilinear)
        # the output is (batch*4, 64, 112, 112)
        self.up3 = Up(256, 128 // factor, bilinear)
        # the final output is (batch*4, 64, 224, 224)
        self.up4 = Up(128, 64, bilinear)
        
        
        self.conv3d = nn.Conv3d(64, 64, (3,3,3), padding=(0,1,1))
        

    

    
    def forward(self, x, reference_color):
        # the input size should be (batch_size*4, 1, 224, 224)
        # while the reference_color is (batch_size, 3, 2, 224, 224)
        #down_feature = self.convnet(x)
        #print('x', x.size())
        #print('refer', reference_color.size())
        row = torch.linspace(0,1,224)
        row = row.unsqueeze(1)
        row = row.expand(224,224)
        col = row.permute(1, 0)
        row = row.unsqueeze(0)
        col = col.unsqueeze(0)

        pos = torch.cat((row, col), dim=0)
        pos = pos.expand(x.size(0), 2, x.size(2), x.size(3)).cuda()
        # the concated input should be (batch_size*4, 3, 224, 224)
        x_pos = torch.cat((x, pos), dim=1)
        #print('x_pos', x_pos.size())
        
        # the final output size is (batch*4, 64, 224, 224)
        x1 = self.inc(x_pos)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        #print('U-net',x.size())        
        # resize the downfeature to (batch_size, 4, 64, 224, 224)
        down_feature = x.view(self.batch_size, -1, x.size(1), x.size(2), x.size(3))
        # permute the channel
        # after permutation, the size should be (batch_size, 64, 4, 224, 224)
        down_feature = down_feature.permute(0, 2, 1, 3, 4)
        #print('down feature', down_feature.size())i


        # implement the 3d conv on reference frames
        # the size of refer is (batch_size, 64, 3, 224, 224)
        # the size of targe is (batch_size, 64, 224, 224)
        reference_down_feature = down_feature[:,:,0:3,:,:]
        target_embedding = down_feature[:,:,3,:,:]
   
        # after conv3d, the size should be (batch_size, 64, 1, 224, 224)
        conv3d_feature = self.conv3d(reference_down_feature)
        
        # after permutation and reshape, the size should be (batch, 64, 224, 224), the embedding size is 32
        reference_embedding = conv3d_feature.permute(0, 2, 1, 3, 4).squeeze(1)
        #print('conv3d',conv3d_feature.size())
       
        #print('refer embed', reference_embedding.size())
        #print('targe embed', target_embedding.size())
       

        # the formula to calculate predicted color of target frame is :
        # y_i = \sum_{i} A_ij * c_i
        # while A_ij = exp(fi * fj) / \sum_{k}  exp(fk * fj)
   

        # the size should be (batch, 1, 224, 224)
        reference_color_a = reference_color[:,2,0,:,:].unsqueeze(1)
        reference_color_b = reference_color[:,2,1,:,:].unsqueeze(1)
        
        reference_color_a = F.pad(reference_color_a, (1,1,1,1), mode="replicate")
        reference_color_b = F.pad(reference_color_b, (1,1,1,1), mode="replicate")

        
        target_a = torch.zeros(self.batch_size, 1, 224, 224)
        target_b = torch.zeros(self.batch_size, 1, 224, 224)

              
        return target_a, target_b


        
        

        

