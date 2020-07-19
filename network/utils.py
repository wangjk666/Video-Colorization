import os
import cv2
import copy
import torch
import pickle
import random
import numpy as np
import subprocess
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch.autograd import Variable
<<<<<<< HEAD
from skimage import io
from skimage.transform import resize
import skimage

=======
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563

VIDEO_FRAMES = 4
VIDEO_ROOT_DIR = '../video_process'

TRAIN_LIST =  './train.list'
TEST_LIST =  './test.list'
                 

def transforms():
    toTensor = torchvision.transforms.ToTensor()
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return torchvision.transforms.Compose([toTensor, normalize])
    
def rgb2lab(image_path):
<<<<<<< HEAD
    '''
=======
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    # the size of image in lab space is (224, 224, 3)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    #img_ls = img_lab[..., 0]
    #img_as = img_lab[..., 1]
    #img_bs = img_lab[..., 2] 
    mytransform = transforms()
    # the size of tensor is [3, 224, 224]
<<<<<<< HEAD
    '''
    mytransform = transforms()
    img_rgb = io.imread(image_path)
    #print(img_rgb.shape)
    img_rgb = resize(img_rgb,(56,56))
    #print(img_rgb.shape)
    img_lab = skimage.color.rgb2lab(img_rgb)
    #print(img_lab.shape)
    img_lab = img_lab.transpose(2,0,1)
    #print(img_lab.shape)
    #img_lab_tensor = mytransform(img_lab)
    #print(img_lab_tensor.size())
    img_lab_tensor = torch.from_numpy(img_lab)
=======
    img_lab_tensor = mytransform(img_lab)
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
    return img_lab_tensor



def get_frames_from_path(video_info, mode, root_path=VIDEO_ROOT_DIR):
    video_frame_path = os.path.join(root_path, video_info)

    all_frame_count = len(os.listdir(video_frame_path))

    if mode == 'train':
<<<<<<< HEAD
        #start_index = random.randint(0, all_frame_count - VIDEO_FRAMES)
        start_index = 9
=======
        start_index = random.randint(0, all_frame_count - VIDEO_FRAMES)
>>>>>>> 27145d2521bc8b4508f2f1d60a25af7c537fe563
    else:
        start_index = 0
    
    myTransform = transforms()

    reference_l_list = []
    reference_ab_list = []
    target_l_list = []
    target_ab_list = []
    for i in range(VIDEO_FRAMES):
        s = "%05d"%start_index
        image_path = 'image_' + s + '.jpg'
        image_path = os.path.join(video_frame_path, image_path)
        '''
        rgb = Image.open(image_path)
        # transform rgb to tensor with size of 3 * 256 * 256
        rgb_tensor = myTransform(rgb)
        # transform rgb to gray with size of 1 * 256 * 256
        # use the equation of gray = 0.2989*r + 0.5870*g + 0.1140*b
        gray = rgb2gray(rgb)
        '''
        # the size should be (3, 224, 224)
        img_lab_tensor = rgb2lab(image_path)
        l_channel = img_lab_tensor[0,:,:]
        # the size of l_channel is (1, 224, 224)
        l_channel = l_channel.unsqueeze(0)
        # the size of ab_channel is (2, 224, 224)
        ab_channel = img_lab_tensor[1:,:,:]
        if i == VIDEO_FRAMES-1:
            l_channel = np.array(l_channel)
            target_l_list.append(l_channel)
            ab_channel = np.array(ab_channel)
            target_ab_list.append(ab_channel)
        else:
            l_channel = np.array(l_channel)
            reference_l_list.append(l_channel)
            ab_channel = np.array(ab_channel)
            reference_ab_list.append(ab_channel)
    
    reference_l_array = np.array(reference_l_list)
    # the size of the reference_l_tensor shoule be (video_frames -1, 1, 224, 224)
    # in the common case, it should be (3, 1, 256, 256)
    reference_l_tensor =  torch.from_numpy(reference_l_array)

    # the size of the reference_ab_tensor shoule be (video_frames-1, 2, 224, 224)
    reference_ab_array = np.array(reference_ab_list)
    reference_ab_tensor = torch.from_numpy(reference_ab_array)

    target_l_array = np.array(target_l_list)
    # the size of the target_l_tensor should be (1, 1, 224, 224)
    target_l_tensor = torch.from_numpy(target_l_array)

    target_ab_array = np.array(target_ab_list)
    # the size of the target_l_tensor should be (1, 2, 224, 224)
    target_ab_tensor = torch.from_numpy(target_ab_array)

    return reference_l_tensor, reference_ab_tensor, target_l_tensor, target_ab_tensor



        
       

        


