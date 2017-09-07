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
#import seq2seq_model
from Flags import *


def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename, encoding='utf8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab, embd


def read_embedding(filename):
    #filename = 'E:/downloads/glove.6B/glove.6B.50d.txt'
    vocab, embd = loadGloVe(filename)
    vocab_size = len(vocab)

    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)

    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    with tf.Session() as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})


    return embedding,vocab,vocab_size,embedding_dim

def test_embedding(config):
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  filename = 'E:/downloads/glove.6B/glove.6B.50d.txt'
  #read_embedding(filename)

  #print("Preparing WMT data in %s" % config.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ ,embd_mat_en,embd_mat_fr= data_utils.prepare_wmt_data(
      config.data_dir, config.en_vocab_size, config.fr_vocab_size, config.load_embeddings)




