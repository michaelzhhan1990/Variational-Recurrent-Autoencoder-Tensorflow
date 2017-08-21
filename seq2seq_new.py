from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np
from utils.distributions import DiagonalGaussian

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access


def embedding_encoder(encoder_inputs,
                      cell,
                      embedding,
                      num_symbols,
                      embedding_size,
                      bidirectional=False,
                      dtype=None,
                      weight_initializer=None,
                      scope=None):

  with variable_scope.variable_scope(
      scope or "embedding_encoder", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    if not embedding:
      embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size],
              initializer=weight_initializer())
    emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in encoder_inputs]
    if bidirectional:
      _, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell, cell, emb_inp,
              dtype=dtype)
      encoder_state = tf.concat(1, [output_state_fw, output_state_bw])
    else:
      _, encoder_state = tf.contrib.rnn.static_rnn(
        cell, emb_inp, dtype=dtype)

    return encoder_state
