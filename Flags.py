import tensorflow as tf

tf.app.flags.DEFINE_string("model_dir", "models", "directory of the model.")
tf.app.flags.DEFINE_boolean("new", True, "whether this is a new model or not.")
tf.app.flags.DEFINE_string("do", "train", "what to do. accepts train, interpolate, sample, and decode.")
tf.app.flags.DEFINE_string("input", None, "input filename for reconstruct sample, and interpolate.")
tf.app.flags.DEFINE_string("output", None, "output filename for reconstruct sample, and interpolate.")

FLAGS = tf.app.flags.FLAGS

def prelu(x):
  with tf.variable_scope("prelu") as scope:
    alphas = tf.get_variable("alphas", [], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    return tf.nn.relu(x) - tf.mul(alphas, tf.nn.relu(-x))
