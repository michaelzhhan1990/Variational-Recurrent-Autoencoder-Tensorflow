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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils.data_utils as data_utils
import seq2seq
from tensorflow.python.ops import variable_scope

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               latent_dim,
               max_gradient_norm,
               batch_size,
               learning_rate,
               kl_min=2,
               word_dropout_keep_prob=1.0,
               anneal=False,
               kl_rate_rise_factor=None,
               use_lstm=False,
               num_samples=512,
               optimizer=None,
               activation=tf.nn.relu,
               forward_only=False,
               feed_previous=True,
               bidirectional=False,
               weight_initializer=None,
               bias_initializer=None,
               iaf=False,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.latent_dim = latent_dim
    self.buckets = buckets
    self.batch_size = batch_size
    self.word_dropout_keep_prob = word_dropout_keep_prob
    self.kl_min = kl_min
    feed_previous = feed_previous or forward_only

    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)

    self.enc_embedding = tf.get_variable("enc_embedding", [source_vocab_size, size], dtype=dtype, initializer=weight_initializer())

    self.dec_embedding = tf.get_variable("dec_embedding", [target_vocab_size, size], dtype=dtype, initializer=weight_initializer())

    self.kl_rate = tf.Variable(
       0.0, trainable=False, dtype=dtype)
    self.new_kl_rate = tf.placeholder(tf.float32, shape=[], name="new_kl_rate")
    self.kl_rate_update = tf.assign(self.kl_rate, self.new_kl_rate)

    self.replace_input = tf.placeholder(tf.int32, shape=[None], name="replace_input")
    replace_input = tf.nn.embedding_lookup(self.dec_embedding, self.replace_input)

    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype, initializer=weight_initializer())
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype, initializer=bias_initializer)
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       num_samples, self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss
    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.contrib.rnn.GRUCell(size)
    if use_lstm:
      single_cell = tf.contrib.rnn.BasicLSTMCell(size)

    cell = single_cell

    def encoder_f(encoder_inputs):
      return seq2seq.embedding_encoder(
          encoder_inputs,
          cell,
          self.enc_embedding,
          num_symbols=source_vocab_size,
          embedding_size=size,
          bidirectional=bidirectional,
          weight_initializer=weight_initializer,
          dtype=dtype)

    def decoder_f(encoder_state, decoder_inputs):
      return seq2seq.embedding_rnn_decoder(
          decoder_inputs,
          encoder_state,
          cell,
          embedding=self.dec_embedding,
          word_dropout_keep_prob=word_dropout_keep_prob,
          replace_input=replace_input,
          num_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          weight_initializer=weight_initializer)

    def enc_latent_f(encoder_state):
      return seq2seq.encoder_to_latent(
                     encoder_state,
                     embedding_size=size,
                     latent_dim=latent_dim,
                     num_layers=num_layers,
                     activation=activation,
                     use_lstm=use_lstm,
                     enc_state_bidirectional=bidirectional,
                     dtype=dtype)
