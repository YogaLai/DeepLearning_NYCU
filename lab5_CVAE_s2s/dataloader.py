import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch

def getDataset(mode):
    words=[]
    tense=[]
    if mode=='train':
        f=open('dataset/train.txt')
        for line in f:
            words.extend(line.split('\n')[0].split(' '))
            tense.extend([0,1,2,3])
    return words, tense
            

class WordTransoformer():
    def __init__(self):
        self.char2idx = {'SOS': 0, 'EOS': 1,}
        self.idx2char = {0: 'SOS', 1: 'EOS'}
        for c in range(ord('a'), ord('z') + 1):
          self.char2idx[chr(c)]=c-95
          self.idx2char[c-95]=chr(c)

    def words2tensor(self, word):
        indices=[]
        for c in word:
            indices.append(self.char2idx[c])
        indices.append(self.char2idx['EOS'])
        return torch.tensor(indices).view(-1,1)
    
    def tensor2words(self, tensor):
        word=[]
        for idx in tensor:
            if idx.item() == self.char2idx['EOS']:
                break
            word.append(self.idx2char[idx.item()])
        return ''.join(word)

class WordDataset(Dataset):
    def __init__(self, mode):
        self.transformer = WordTransoformer()
        self.tense2idx={'sp':0,'tp':1,'pg':2,'p':3}
        self.idx2tense={0:'sp',1:'tp',2:'pg',3:'p'}
        self.words, self.tense = getDataset(mode)
        self.max_length = 0
        for word in self.words:
            if len(word) > self.max_length:
                self.max_length = len(word)

    def __len__(self):
        return len(self.tense)

    def __getitem__(self, index):
        tense = torch.tensor(self.tense[index]).view(-1,1)
        return self.transformer.words2tensor(self.words[index]), tense
