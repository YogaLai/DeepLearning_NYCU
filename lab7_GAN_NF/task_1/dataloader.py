import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
        
    elif mode =='new_test':
        data = json.load(open(os.path.join(root_folder,'new_test.json')))
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))

    obj = json.load(open(os.path.join(root_folder,'objects.json')))
    label = data
    for i in range(len(label)):
        for j in range(len(label[i])):
            label[i][j] = obj[label[i][j]]
        tmp = np.zeros(len(obj))
        tmp[label[i]] = 1
        label[i] = tmp
    return None, label


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False, mode='train'):
        self.root_folder = root_folder
        self.mode = mode
        self.img_list, self.label_list = get_iCLEVR_data(root_folder,mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
        
        self.cond = cond
        self.num_classes = 24
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])
        
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        condition = self.label_list[index]
        if self.mode == 'train':
            img_pth = self.img_list[index]
            img = Image.open(self.root_folder + 'images/' + img_pth).convert('RGB')
            # img = img.resize((64,64))
            img = self.transform(img)
            # tmp = img.permute(1,2,0)
            # tmp = np.asarray(tmp)
            # plt.imsave('test.png', tmp)

            return img, condition.astype(np.float32)
        
        else:
            return condition.astype(np.float32)