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

def prelu(_x):
  with tf.variable_scope("prelu"):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
      initializer=tf.constant_initializer(0.0),
      dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


def rnn_decoder(decoder_inputs, initial_state, cell, word_dropout_keep_prob=1, replace_inp=None,
                loop_function=None, scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    seq_len = len(decoder_inputs)
    keep = tf.select(tf.random_uniform([seq_len]) < word_dropout_keep_prob,
            tf.fill([seq_len], True), tf.fill([seq_len], False))
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          if word_dropout_keep_prob < 1:
            inp = tf.cond(keep[i], lambda: loop_function(prev, i), lambda: replace_inp)
          else:
            inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


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

    emb_inp = [tf.nn.embedding_lookup(embedding, i) for i in encoder_inputs]
    if bidirectional:
      _, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell, cell, emb_inp,
              dtype=dtype)
      encoder_state = tf.concat(1, [output_state_fw, output_state_bw])
    else:
      _, encoder_state = tf.contrib.rnn.static_rnn(
        cell, emb_inp, dtype=dtype)

    return encoder_state


def beam_rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None,output_projection=None, beam_size=1):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    log_beam_probs, beam_path, beam_symbols = [],[],[]
    state_size = int(initial_state.get_shape().with_rank(2)[1])

    #inp is not set any valule yet, just a place holder
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i,log_beam_probs, beam_path, beam_symbols)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      input_size = inp.get_shape().with_rank(2)[1]
      x = inp
      output, state = cell(x, state)

      if loop_function is not None:
        prev = output
      if  i ==0:  #only initlilize it in the beginning
          states =[]
          for kk in range(beam_size):
                states.append(state)
          state = tf.reshape(tf.concat(0, states), [-1, state_size])

      outputs.append(tf.argmax(nn_ops.xw_plus_b(
          output, output_projection[0], output_projection[1]), dimension=1))
  return outputs, state, tf.reshape(tf.concat(0, beam_path),[-1,beam_size]), tf.reshape(tf.concat(0, beam_symbols),[-1,beam_size])


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          embedding,
                          num_symbols,
                          embedding_size,
                          word_dropout_keep_prob=1,
                          replace_input=None,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          weight_initializer=None,
                          beam_size=1,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    if not embedding:
      embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size],
              initializer=weight_initializer())

    '''
    if beam_size > 1:
        loop_function = _extract_beam_search(
        embedding, beam_size,num_symbols,embedding_size,  output_projection,
        update_embedding_for_previous)
    else:
    '''

    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None

    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    if beam_size > 1:
        return beam_rnn_decoder(emb_inp, initial_state, cell,loop_function=loop_function,
                output_projection=output_projection, beam_size=beam_size)

    return rnn_decoder(emb_inp, initial_state, cell, word_dropout_keep_prob, replace_input,
                       loop_function=loop_function)

