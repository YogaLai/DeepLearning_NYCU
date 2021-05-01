import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
    else:
        img = pd.read_csv('test_img.csv')
        label= pd.read_csv('test_label.csv')
    return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(Dataset):
    def __init__(self, img_path, mode):
        """
        Args:
            img_path: Root path of the dataset.
            mode: training/testing
            
            self.img_names (string list): String list that store all image names.
            self.labels (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_path = img_path
        self.img_names, self.labels = getData(mode)        
        
        self.transformations=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        print(f'>> Found {len(self.labels)} images...')
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        single_img_name=os.path.join(self.img_path,self.img_names[index]+'.jpeg')
        single_img=Image.open(single_img_name)  # read an PIL image
        img=self.transformations(single_img)
        label=self.labels[index]
        
        return img, label