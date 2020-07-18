import torch
from torch.utils.data import Dataset, DataLoader
from utils import *

class VideoDataset(Dataset):
    def __init__(self, info_txt=TRAIN_LIST, root_path=VIDEO_ROOT_DIR, mode='train'):
        # set params
        self.info_txt = info_txt
        self.root_path = root_path
        self.mode=mode
        # read info_list
        self.info_list=open(self.info_txt).readlines()

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        info_line = self.info_list[idx]
        video_info = info_line.strip('\n')
        
        reference_l_tensor, reference_ab_tensor, target_l_tensor, target_ab_tensor = get_frames_from_path(video_info, self.mode, self.root_path)

        sample = {'reference_l': reference_l_tensor, 'reference_ab': reference_ab_tensor, 'target_l': target_l_tensor, 'target_ab': target_ab_tensor}

        sample['reference_l'] = sample['reference_l'].float()
        sample['reference_ab'] = sample['reference_ab'].float()
        sample['target_l'] =  sample['target_l'].float()
        sample['target_ab'] = sample['target_ab'].float()

        return sample

