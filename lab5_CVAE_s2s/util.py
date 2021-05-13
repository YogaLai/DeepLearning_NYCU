import numpy as np
import math
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import time

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""

def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = 'dataset/train.txt'#should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_kl_weight(epoch,epochs,kl_annealing_type,time):
    """
    :param epoch: i-th epoch
    :param kl_annealing_type: 'monotonic' or 'cycle'
    :param time:
        if('monotonic'): # of epoch for kl_weight from 0.0 to reach 1.0
        if('cycle'):     # of cycle
    """
    assert kl_annealing_type=='monotonic' or kl_annealing_type=='cycle','kl_annealing_type not exist!'

    if kl_annealing_type == 'monotonic':
        return (1./(time-1))*(epoch-1) if epoch<time else 1.

    else: #cycle
        period = epochs//time
        epoch %= period
        kl_weight = sigmoid((epoch - period // 2) / (period // 10)) / 2

        # if epoch % time == 0:
        #     kl_weight = args.kl_start
        # else:
        #     kl_weight = min(1, (epoch%time) * 0.015)

        return kl_weight