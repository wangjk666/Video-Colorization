import  numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from utils import *
from dataloader import *
from model import *
<<<<<<< HEAD
import skimage
from skimage import io
from skimage import img_as_ubyte

def NormMinandMax(npdarr, min=0, max=1):
    arr = npdarr.flatten()
    Ymax = np.max(arr) 
    Ymin = np.min(arr) 
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (arr - Ymin)
    last = np.reshape(last, npdarr.shape)
    return last
 
=======

>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
class TrainNetwork():
    def __init__(self, ckp_path, epoch_nums, batch_size, lr, lr_step_size, reference_frames=3):
        # get params
        self.ckp_path = ckp_path
        self.epoch_nums = epoch_nums
        self.batch_size = batch_size
        self.lr= lr
        self.lr_step_size = lr_step_size
        self.reference_frames = reference_frames
        

        # mkdir ckg_path
        if not os.path.exists(self.ckp_path):
            os.makedirs(self.ckp_path)

        # load model
        self.mymodel = Model(batch_size = self.batch_size)
        self.mymodel.train()
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
<<<<<<< HEAD
        #self.mymodel = torch.nn.DataParallel(self.mymodel)
=======
        self.mymodel = torch.nn.DataParallel(self.mymodel)
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
        self.mymodel.cuda()
        print('model loaded.')

        # define video_dataloader
        self.myDataset = VideoDataset(TRAIN_LIST)
        self.myDataloader = DataLoader(self.myDataset, batch_size= self.batch_size, shuffle=True, num_workers=0)

    def finetune_model(self):
        # define params
<<<<<<< HEAD
        #params_list = list(self.mymodel.parameters()) 
        optimizer = optim.Adam(self.mymodel.parameters(), lr= self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.lr_step_size, gamma=0.1)
        
        
        criterion = nn.MSELoss().cuda()
=======
        params_list = list(self.mymodel.parameters()) 
        optimizer = optim.SGD(params_list, lr= self.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.lr_step_size, gamma=0.1)
        
        
        criterion = nn.MSELoss()
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
	
	# training process
        for epoch in range(self.epoch_nums):
            #scheduler_1.step()
            #scheduler_2.step()
            running_loss=0.0
            for i_batch, sample_batched in enumerate(self.myDataloader):
                # get the inputs
                # the size is:
                # reference_l: (batch_size, 3, 1, 224, 224)
                # reference_ab: (batch_size, 3, 2, 224, 224)
                # target_l: (batch_size, 1, 1, 224, 224)
                # target_ab: (batch_size, 1, 2, 224, 224)
                reference_l, reference_ab, target_l, target_ab = sample_batched['reference_l'], sample_batched['reference_ab'], sample_batched['target_l'], sample_batched['target_ab']
                
                reference_l = reference_l.view(self.batch_size*self.reference_frames, reference_l.size(2), reference_l.size(3), reference_l.size(4))
                #reference_ab = reference_ab.view(self.batch_size*self.reference_frames, reference_ab.size(2), reference_ab.size(3), reference_ab.size(4))

                target_l = target_l.view(self.batch_size, target_l.size(2), target_l.size(3), target_l.size(4))
                #target_ab = target_ab.view(self.batch_size, target_ab.size(2), target_ab.size(3), target_ab.size(4))

                # the input to the model should be (batch_size*4, 1, 224, 224)
                x = torch.cat((reference_l, target_l), dim=0)
                x = Variable(x).cuda()
                
                # the size of reference color is (batch_size, 3, 2, 224, 224)
                reference_color = Variable(reference_ab).cuda()
                # the size of the target color is (batch_size, 1, 2, 224, 224)
<<<<<<< HEAD
                target_ab = target_ab.squeeze(1)
=======
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
                target_color = Variable(target_ab).cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
               
                
                # forward   
                # the size of target_a / target_b is (batch_size, 1, 224, 224)
                target_a, target_b = self.mymodel(x, reference_color)
<<<<<<< HEAD
                 
                #print(target_a.size())
                #print(target_b.size())
                
                
                #max_a = torch.max(target_a).item()
                #max_b = torch.max(target_b).item()
                #min_a = torch.min(target_a).item()
                #min_b = torch.min(target_b).item()
                #print(max_a, min_a)
                #print(max_b, min_b)
                #print(torch.max(target_ab[:,0,:,:]).item())
                #print(torch.min(target_ab[:,0,:,:]).item())
                #print(torch.max(target_ab[:,1,:,:]).item())
                #print(torch.min(target_ab[:,1,:,:]).item())
                '''
                k1 = 200/(max_a - min_a)
                k2 = 200/(max_b - min_b)
                #print(k1, k2)
                #max_a = max_a*torch.ones(self.batch_size, 1, 224, 224).cuda()
                #max_b = max_b*torch.ones(self.batch_size, 1, 224, 224).cuda()
                min_a = min_a*torch.ones(self.batch_size, 1, 224, 224).cuda()
                min_b = min_b*torch.ones(self.batch_size, 1, 224, 224).cuda()

                low = -100*torch.ones(self.batch_size, 1, 224, 224).cuda()
             
                target_a = k1*(target_a - min_a)+low
                target_b = k2*(target_b - min_b)+low 
                target_a = target_a.unsqueeze(1)
                target_b = target_b.unsqueeze(1)
                #print(target_a.device)
                '''

                predicted_color = torch.cat((target_a, target_b), dim=1).cuda()
                
                #predicted_image_lab = torch.cat((target_l, predicted_color), dim=2).squeeze(0).squeeze(0)
                
                #print(predicted_color.size())
                #print(target_color.size())
                # get loss anf backward
                #print("predicted max\n",torch.max(predicted_color))
                #print("predicted min\n",torch.min(predicted_color))
                #print("target max\n",torch.max(target_color))
                #print("target min\n",torch.min(target_color))
=======
                target_a = target_a.unsqueeze(1)
                target_b = target_b.unsqueeze(1)

                predicted_color = torch.cat((target_a, target_b), dim=2)

 

                # get loss anf backward
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
                loss = criterion(predicted_color, target_color)
                loss.backward()
                optimizer.step()
                
<<<<<<< HEAD
                predicted_color = predicted_color.cpu()
                target_color = target_color.cpu()
                
                
                predicted_image_lab = torch.cat((target_l, predicted_color), dim=1).squeeze(0)
                #print(predicted_image_lab.size())
                gt_image_lab = torch.cat((target_l,target_color),dim=1).squeeze(0)
                #print(gt_image_lab.size())

                image_true_file = os.path.join(self.ckp_path, "true")
                image_l_file = os.path.join(self.ckp_path,"l")
                image_pred_file = os.path.join(self.ckp_path, "pred")
                if not os.path.exists(image_true_file):
                     os.makedirs(image_true_file)
                if not os.path.exists(image_l_file):
                     os.makedirs(image_l_file)
                if not os.path.exists(image_pred_file):
                     os.makedirs(image_pred_file)

                image_path = os.path.join(image_true_file, "image_"+str(epoch)+".jpg")
                image_true_path = os.path.join(image_l_file, "true_"+str(epoch)+".jpg")                  
                image_l_path = os.path.join(image_pred_file, "l_"+str(epoch)+".jpg")
                input_l = target_l.squeeze(0)
                #print(input_l.size())

    
                predicted_image_lab = predicted_image_lab.permute(1,2,0)
                predicted_image_lab = predicted_image_lab.detach().numpy()
                predicted_image_rgb = skimage.color.lab2rgb(predicted_image_lab)
                #predicted_image_rgb = NormMinandMax(predicted_image_rgb, min=0,max=0.99)

                gt_image_lab = gt_image_lab.permute(1,2,0)
                gt_image_lab = gt_image_lab.detach().numpy()
                gt_image_rgb = skimage.color.lab2rgb(gt_image_lab)
                 
                print(np.amax(predicted_image_rgb))
                print(np.amax(gt_image_rgb))
                print(np.amin(predicted_image_rgb))
                print(np.amin(gt_image_rgb)) 
                input_l = input_l.permute(1,2,0)
                input_l = input_l.detach().numpy()
                
                #print("pred:\n",predicted_image_lab)
                #print("gt:\n",gt_image_lab)
                io.imsave(image_l_path, input_l)
                io.imsave(image_path, img_as_ubyte(predicted_image_rgb))
                io.imsave(image_true_path, img_as_ubyte(gt_image_rgb))
 
=======

                
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
                # print statistics
                running_loss = running_loss+loss.item()
                #if i_batch % 50 == 49:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i_batch + 1, running_loss))
                #print ('[%d, %5d] loss: %.3f' %(epoch + 1, i_batch + 1, running_loss / 50),file=file)
                running_loss = 0.0

            scheduler.step()
            

<<<<<<< HEAD
            save_model_path= os.path.join(self.ckp_path,'model'+str(epoch+1)+'.pkl')
            #torch.save(self.mymodel.state_dict(),save_model_path)
=======
            save_model_path= self.ckp_path +'model'+str(epoch+1)+'.pkl'
            torch.save(self.mymodel.module.state_dict(),save_model_path)
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)

<<<<<<< HEAD
    ckp_path, epoch_nums, batch_size, lr, lr_step_size = "test_1", 100, 1, 0.001, 50
=======
    ckp_path, epoch_nums, batch_size, lr, lr_step_size = "test_1", 20, 1, 0.0001, 10
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
    mytrain = TrainNetwork(ckp_path, epoch_nums, batch_size, lr, lr_step_size)
    mytrain.finetune_model()
