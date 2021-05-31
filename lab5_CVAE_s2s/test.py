import torch
from cvae import CVAE
from dataloader import WordDataset, WordTransoformer
from torch.utils.data import DataLoader
import sys
import numpy as np
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_size = 32

def evaluate(model, dataloader, tense_list):
    with torch.no_grad():
        model.eval()
        total_BLEU_score = 0
        predict_list = []
        for times, (word_tensor, tense_tensor, target) in enumerate(dataloader):
            word_tensor = word_tensor[0]
            tense_tensor = tense_tensor[0]
            target = target[0]
            word_tensor, tense_tensor = word_tensor.to(device), tense_tensor.to(device)
            # inference without teacher forcing => teacher forcing ratio sets -1
            output, predict_distribution, mean, log_var = model(word_tensor, tense_tensor, -1, inference=True)

            predict = transformer.tensor2words(output)
            total_BLEU_score += compute_bleu(predict, target)
            
            predict_list.append(predict)
        
        # Gaussian score
        words = []
        for i in range(100):
            latent = torch.randn(1, 1, latent_size).to(device)
            words.append(model.generate_words(latent, tense_list))
        for word_list in words:
            for i in range(len(word_list)):
                word_list[i] = transformer.tensor2words(word_list[i])
        gaussian_score = Gaussian_score(words)
      
    return total_BLEU_score/len(dataloader.dataset), predict_list, gaussian_score, words

def record_score(bleu_score, gaussian_score, predict_list, generate_words, dataloader, transformer):
    bleu_record = open('test/bleu_record.txt', 'w')
    for i, (word_tensor, tense_tensor, target) in enumerate(dataloader):
        word_tensor = word_tensor[0]
        target = target[0]
        input_word = transformer.tensor2words(word_tensor)
        print('----------------', file=bleu_record)
        print('Input: ', input_word, file=bleu_record)
        print('Target: ', target, file=bleu_record)
        print('Prediction: ', predict_list[i], file=bleu_record) 
        print('----------------\n', file=bleu_record)

    print('Average BLEU-4 score: ', bleu_score, file=bleu_record)
    bleu_record.close()
    
    gaussian_record = open('test/gaussian_record.txt', 'w')
    for word_list in generate_words:
        for i, word in enumerate(word_list):
            if (i+1) % 4 != 0:
                word += ', '
            else:
                word +='\n'
            print(word, file=gaussian_record, end='')
    print('Gaussian score: ', gaussian_score, file=gaussian_record)
    gaussian_record.close()


test_dataset = WordDataset('test')
max_length = test_dataset.max_length
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
tense_list = dataloader.dataset.tense2idx.values()
transformer = WordTransoformer()

# Epoch 57 is the best
loadmodel = 'model/cycle_500/checkpoint57.pkl'
model = CVAE(max_length)
model = model.cuda()
state_dict = torch.load(loadmodel)
model.load_state_dict(state_dict)

average_bleu_score, predict_list, gaussian_score, generate_words = evaluate(model, dataloader, tense_list)
record_score(average_bleu_score, gaussian_score, predict_list, generate_words, dataloader, transformer)
