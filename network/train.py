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
        self.mymodel = torch.nn.DataParallel(self.mymodel)
        self.mymodel.cuda()
        print('model loaded.')

        # define video_dataloader
        self.myDataset = VideoDataset(TRAIN_LIST)
        self.myDataloader = DataLoader(self.myDataset, batch_size= self.batch_size, shuffle=True, num_workers=0)

    def finetune_model(self):
        # define params
        params_list = list(self.mymodel.parameters()) 
        optimizer = optim.SGD(params_list, lr= self.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.lr_step_size, gamma=0.1)
        
        
        criterion = nn.MSELoss()
	
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
                target_color = Variable(target_ab).cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
               
                
                # forward   
                # the size of target_a / target_b is (batch_size, 1, 224, 224)
                target_a, target_b = self.mymodel(x, reference_color)
                target_a = target_a.unsqueeze(1)
                target_b = target_b.unsqueeze(1)

                predicted_color = torch.cat((target_a, target_b), dim=2)

 

                # get loss anf backward
                loss = criterion(predicted_color, target_color)
                loss.backward()
                optimizer.step()
                

                
                # print statistics
                running_loss = running_loss+loss.item()
                #if i_batch % 50 == 49:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i_batch + 1, running_loss))
                #print ('[%d, %5d] loss: %.3f' %(epoch + 1, i_batch + 1, running_loss / 50),file=file)
                running_loss = 0.0

            scheduler.step()
            

            save_model_path= self.ckp_path +'model'+str(epoch+1)+'.pkl'
            torch.save(self.mymodel.module.state_dict(),save_model_path)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)

    ckp_path, epoch_nums, batch_size, lr, lr_step_size = "test_1", 20, 1, 0.0001, 10
    mytrain = TrainNetwork(ckp_path, epoch_nums, batch_size, lr, lr_step_size)
    mytrain.finetune_model()
