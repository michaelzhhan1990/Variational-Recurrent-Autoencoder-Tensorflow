from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import sys
#sys.path.insert(0, '../')



import json
from Configuration import *
from test_embedding import *
from training import *


def main(_):

  with open(os.path.join(FLAGS.model_dir, "config.json")) as config_file:
    configs = json.load(config_file)

  FLAGS.model_name = os.path.basename(os.path.normpath(FLAGS.model_dir))
  behavior = ["train", "interpolate", "reconstruct", "sample","test_embedding"]
  if FLAGS.do not in behavior:
    raise ValueError("argument \"do\" is not one of the following: train, interpolate, decode or sample.")

  if FLAGS.do != "train" and FLAGS.do != "test_embedding":
    FLAGS.new = False

  config = Config(**configs["model"])
  config.update(**configs[FLAGS.do])
  interp_config = Config(**configs["model"])
  interp_config.update(**configs["interpolate"])
  enc_dec_config = Config(**configs["model"])
  enc_dec_config.update(**configs["reconstruct"])
  sample_config = Config(**configs["model"])
  sample_config.update(**configs["sample"])
  embedding_config =Config(**configs["test_embedding"])
  embedding_config.update(**configs["test_embedding"])

  if FLAGS.do == "reconstruct":
    with tf.Session() as sess:
      model = create_model(sess, enc_dec_config, True)
      #reconstruct(sess, model, enc_dec_config)
  elif FLAGS.do == "interpolate":
    with tf.Session() as sess:
      model = create_model(sess, interp_config, True)
      #encode_interpolate(sess, model, interp_config)
  elif FLAGS.do == "sample":
    with tf.Session() as sess:
      model = create_model(sess, sample_config, True)
      #n_sample(sess, model, config)
  elif FLAGS.do == "train":
    train(config)
  elif FLAGS.do == 'test_embedding':
    test_embedding(config)


if __name__ == "__main__":
  tf.app.run()
