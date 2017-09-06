# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import utils.data_utils as data_utils
import utils.read_data as read_data
# import seq2seq_model
from Flags import *


def loadGloVe(filename):
    vocab = []
    embd = []
    embd_dic={}
    file = open(filename, encoding='utf8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
        embd_dic[vocab[-1]]=embd[-1]
    print('Loaded GloVe!')
    file.close()
    embedding_dim = len(embd[0])
    return embd_dic, embedding_dim


def read_embedding(filename):
    # filename = 'E:/downloads/glove.6B/glove.6B.50d.txt'
    vocab, embd = loadGloVe(filename)
    vocab_size = len(vocab)

    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    return embedding, vocab, vocab_size, embedding_dim