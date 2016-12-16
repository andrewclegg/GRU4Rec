# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from itertools import izip

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import gru4rec
import evaluation

PATH_TO_TRAIN = '/home/aclegg/data/recsys_challenge_2015/gru4rec/rsc15_train_full.txt'
PATH_TO_TEST = '/home/aclegg/data/recsys_challenge_2015/gru4rec/rsc15_test.txt'

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
    
    print('Training GRU4Rec with 100 hidden units')    
    
    gru = gru4rec.GRU4Rec(layers=[100], loss='top1', batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0)
    gru.fit(data)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    
