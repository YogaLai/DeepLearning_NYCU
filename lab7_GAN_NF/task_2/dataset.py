import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def get_CelebA_data(root_folder):
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list


class CelebALoader(data.Dataset):
    def __init__(self, root_folder, img_size, trans=None, cond=False):
        self.root_folder = root_folder
        assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)
        
        self.cond = cond
        self.img_list, self.label_list = get_CelebA_data(self.root_folder)
        self.num_classes = 40
        self.transfrom = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
        print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        path = '/media/jojorge/NTFS/yoga/109_2/DeepLearning/DeepLearning_NYCU/lab7_GAN_NF/task_2/CelebA-HQ-img/'
        img = Image.open(path + self.img_list[index])
        img = self.transfrom(img)
        label = torch.tensor(self.label_list[index])

        return img, label

