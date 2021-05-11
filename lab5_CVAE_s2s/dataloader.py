import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch

def getDataset(mode):
    words=[]
    tense=[]
    target=[]
    if mode=='train':
        f=open('dataset/train.txt')
        for line in f:
            words.extend(line.split('\n')[0].split(' '))
            tense.extend([0,1,2,3])
        return words, tense
    else:
        f=open('dataset/test.txt')
        for line in f:
            word_pair = line.split('\n')[0].split(' ')
            words.append(word_pair[0])
            target.append(word_pair[1])
        return words, target
            

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
        self.mode = mode
        self.transformer = WordTransoformer()
        self.tense2idx={'sp':0,'tp':1,'pg':2,'p':3}
        self.idx2tense={0:'sp',1:'tp',2:'pg',3:'p'}
        self.words, self.tense = getDataset(mode)
        self.max_length = 0
        for word in self.words:
            if len(word) > self.max_length:
                self.max_length = len(word)
        if mode == 'train':
            self.words, self.tense = getDataset(mode)
        else:
            self.words, self.target = getDataset(mode)
            self.test_tense=['p','pg','tp','tp','tp','pg','sp','sp','p','tp']
            for i in range(len(self.test_tense)):
                self.test_tense[i] = self.tense2idx[self.test_tense[i]]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        word = self.transformer.words2tensor(self.words[index])
        if self.mode == 'train':
            tense = torch.tensor(self.tense[index]).view(-1,1)
            return word, tense
        else:
            tense = torch.tensor(self.test_tense[index]).view(-1,1)
            target = self.target[index] 
            return word, tense, target  
