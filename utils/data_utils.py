# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "r") as gz_file:
    with open(new_path, "w") as new_file:
      for line in gz_file:
        new_file.write(line)


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def loadGloVe(filename,normalize_digits=True):
    vocab = []
    embd = []
    embd_dic={}
    file = open(filename, encoding='utf8')
    for line in file.readlines():
        row = line.strip().split(' ')
        w=row[0]
        # make it consistent
        word = _DIGIT_RE.sub("0", w) if normalize_digits else w
        vocab.append(word)
        embd.append(row[1:])
        embd_dic[vocab[-1]]=embd[-1]
    print('Loaded GloVe!')
    file.close()
    return vocab, embd,embd_dic



def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, embedding_path=None, W,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path) :
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub("0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

      id2embd_dic=None
      if embedding_path !=None:
        embd_dic,embedding_dim=loadGloVe(embedding_path)
        embd_dic['_PAD']=[PAD_ID] * embedding_dim
        embd_dic['_GO'] = [GO_ID] * embedding_dim
        embd_dic['_EOS'] = [EOS_ID] * embedding_dim
        embd_dic['_UNK'] = [UNK_ID] * embedding_dim

        id2embd_dic=[]

      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
          for id, w in enumerate(vocab_list):
            vocab_file.write(w + "\n")

            if embedding_path!=None:
              if w in embd_dic:
                id2embd_dic.append(embd_dic[w])
              else:  #in vocab but no embdding
                id2embd_dic.append(embd_dic['_UNK'])
          if embedding_path!=None:
            embedding = np.asarray(id2embd_dic)

            W_ = tf.Variable(tf.constant(0.0, shape=[len(vocab_list), embedding_dim]),
                             trainable=False, name="W")

            embedding_placeholder = tf.placeholder(tf.float32, [len(vocab_list), embedding_dim])
            embedding_init = W.assign(embedding_placeholder)
            tf.session.run(embedding_init, feed_dict={embedding_placeholder: embedding})

  #W is the actual tensor of embeddings for each vocabulary










def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size,
        load_embeddings=False, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
      (7) embedding dictionary(id, embedding)

  """
  # Get wmt data to the specified directory.
  train_path = os.path.join(data_dir, "train.txt")
  dev_path = os.path.join(data_dir, "dev.txt")

  # Create vocabularies of the appropriate sizes.
  fr_vocab_path = os.path.join(data_dir, "vocab%d.out" % fr_vocabulary_size)
  en_vocab_path = os.path.join(data_dir, "vocab%d.in" % en_vocabulary_size)

  w_fr=None

  create_vocabulary(fr_vocab_path, train_path + ".out", fr_vocabulary_size,
          #os.path.join(data_dir, "dec_embedding{0}.tsv".format(fr_vocabulary_size)),
          w_fr ,
          tokenizer)
  create_vocabulary(en_vocab_path, train_path + ".in", en_vocabulary_size,
          #os.path.join(data_dir, "enc_embedding{0}.tsv".format(en_vocabulary_size)),
          tokenizer)


  #if load_embeddings:
    #embed_utils.save_embeddings(fr_vocab_path, "embed5000.txt")
    #embed_utils.save_embeddings(en_vocab_path, "embed5000.txt")
    

  # Create token ids for the training data.
  fr_train_ids_path = train_path + (".ids%d.out" % fr_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.in" % en_vocabulary_size)
  data_to_token_ids(train_path + ".out", fr_train_ids_path, fr_vocab_path, tokenizer)
  data_to_token_ids(train_path + ".in", en_train_ids_path, en_vocab_path, tokenizer)

  # Create token ids for the development data.
  fr_dev_ids_path = dev_path + (".ids%d.out" % fr_vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.in" % en_vocabulary_size)
  data_to_token_ids(dev_path + ".out", fr_dev_ids_path, fr_vocab_path, tokenizer)
  data_to_token_ids(dev_path + ".in", en_dev_ids_path, en_vocab_path, tokenizer)

  return (en_train_ids_path, fr_train_ids_path,
          en_dev_ids_path, fr_dev_ids_path,
          en_vocab_path, fr_vocab_path)
