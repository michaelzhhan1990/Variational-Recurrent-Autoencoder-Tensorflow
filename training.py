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
import seq2seq_model
from Flags import *

def create_model(session, config, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float32
  optimizer = None
  if not forward_only:
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
  if config.activation == "elu":
    activation = tf.nn.elu
  elif config.activation == "prelu":
    activation = prelu
  else:
    activation = tf.identity

  weight_initializer = tf.orthogonal_initializer if config.orthogonal_initializer else tf.uniform_unit_scaling_initializer
  bias_initializer = tf.zeros_initializer

  model = seq2seq_model.Seq2SeqModel(
      config.en_vocab_size,
      config.fr_vocab_size,
      config.buckets,
      config.size,
      config.num_layers,
      config.latent_dim,
      config.max_gradient_norm,
      config.batch_size,
      config.learning_rate,
      config.kl_min,
      config.word_dropout_keep_prob,
      config.anneal,
      config.use_lstm,
      optimizer=optimizer,
      activation=activation,
      forward_only=forward_only,
      feed_previous=config.feed_previous,
      bidirectional=config.bidirectional,
      weight_initializer=weight_initializer,
      bias_initializer=bias_initializer,
      iaf=config.iaf,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if not FLAGS.new and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train(config):
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  print("Preparing WMT data in %s" % config.data_dir)
  en_embd_name=""
  en_train, fr_train, en_dev, fr_dev, _, _, embd_mat_en, embd_mat_fr = data_utils.prepare_wmt_data(
      config.data_dir, config.en_vocab_size, config.fr_vocab_size)
  #config.embedding_en_path, config.embedding_fr_path, 'enc_embedding', 'dec_embedding')

  with tf.Session() as sess:
    if not os.path.exists(FLAGS.model_dir):
      os.makedirs(FLAGS.model_dir)

    # Create model.
    print("Creating %d layers of %d units." % (config.num_layers, config.size))
    model = create_model(sess, config, False)

    #if not config.probabilistic:
     # self.kl_rate_update(0.0)

    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir,"train"), graph=sess.graph)
    dev_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, "test"), graph=sess.graph)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % config.max_train_data_size)

    dev_set = read_data.read_data(en_dev, fr_dev, config)
    train_set = read_data.read_data(en_train, fr_train, config, config.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(config.buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    KL_loss = 0.0
    current_step = model.global_step.eval()
    step_loss_summaries = []
    step_KL_loss_summaries = []
    overall_start_time = time.time()
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, step_KL_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False, config.probabilistic)

      if config.anneal and model.global_step.eval() > config.kl_rate_rise_time and model.kl_rate < 1:
        new_kl_rate = model.kl_rate.eval() + config.kl_rate_rise_factor
        sess.run(model.kl_rate_update, feed_dict={'new_kl_rate': new_kl_rate})

      step_time += (time.time() - start_time) / config.steps_per_checkpoint
      step_loss_summaries.append(tf.Summary(value=[tf.Summary.Value(tag="step loss", simple_value=float(step_loss))]))
      step_KL_loss_summaries.append(tf.Summary(value=[tf.Summary.Value(tag="KL step loss", simple_value=float(step_KL_loss))]))
      loss += step_loss / config.steps_per_checkpoint
      KL_loss += step_KL_loss / config.steps_per_checkpoint
      current_step = model.global_step.eval()

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % config.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

        print ("global step %d learning rate %.4f step-time %.2f KL divergence "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, KL_loss))
        wall_time = time.time() - overall_start_time
        print("time passed: {0}".format(wall_time))

        # Add perplexity, KL divergence to summary and stats.
        perp_summary = tf.Summary(value=[tf.Summary.Value(tag="train perplexity", simple_value=perplexity)])
        train_writer.add_summary(perp_summary, current_step)
        KL_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="KL divergence", simple_value=KL_loss)])
        train_writer.add_summary(KL_loss_summary, current_step)
        for i, summary in enumerate(step_loss_summaries):
          train_writer.add_summary(summary, current_step - 200 + i)
        step_loss_summaries = []
        for i, summary in enumerate(step_KL_loss_summaries):
          train_writer.add_summary(summary, current_step - 200 + i)
        step_KL_loss_summaries = []

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name + ".ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss, KL_loss = 0.0, 0.0, 0.0

        # Run evals on development set and print their perplexity.
        eval_losses = []
        eval_KL_losses = []
        eval_bucket_num = 0
        for bucket_id in xrange(len(config.buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          eval_bucket_num += 1
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, eval_KL_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True, config.probabilistic)
          eval_losses.append(float(eval_loss))
          eval_KL_losses.append(float(eval_KL_loss))
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

          eval_perp_summary = tf.Summary(value=[tf.Summary.Value(tag="eval perplexity for bucket {0}".format(bucket_id), simple_value=eval_ppx)])
          dev_writer.add_summary(eval_perp_summary, current_step)

        mean_eval_loss = sum(eval_losses) / float(eval_bucket_num)
        mean_eval_KL_loss = sum(eval_KL_losses) / float(eval_bucket_num)
        mean_eval_ppx = math.exp(float(mean_eval_loss))
        print("  eval: mean perplexity {0}".format(mean_eval_ppx))

        eval_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean eval loss", simple_value=float(mean_eval_ppx))])
        dev_writer.add_summary(eval_loss_summary, current_step)
        eval_KL_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean eval loss", simple_value=float(mean_eval_KL_loss))])
        dev_writer.add_summary(eval_KL_loss_summary, current_step)