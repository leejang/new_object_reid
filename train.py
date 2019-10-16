from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math

slim = tf.contrib.slim
from ops import *

# dataset Veri-776
import veri_776 
# inception v3 and mobilenet v1 use the same preprocessing procedcure
import inception_preprocessing

# inception v3
from inception_v3 import inception_v3, inception_v3_arg_scope
#image_size = inception_v3.default_image_size
# mobilenet v1
from mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
#image_size = mobilenet_v1.default_image_size
# mobilenet v1 with self-attention (sa)
from mobilenet_v1_w_sa import mobilenet_v1_w_sa, mobilenet_v1_w_sa_arg_scope
image_size = mobilenet_v1_w_sa.default_image_size


LIB_NAME = 'extra_losses'

def load_op_module(lib_name):
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf.extra_losses/build/lib{0}.so'.format(lib_name))
  oplib = tf.load_op_library(lib_path)
  return oplib

op_module = load_op_module(LIB_NAME)


tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate')

tf.flags.DEFINE_string('log_dir', './mobilenet_v1_w_sa_two_self_sn_n_cos_loss_logs', 
                        'The directory to save the model files in')
tf.flags.DEFINE_string('dataset_dir', './tfrecords/train',
                        'The directory where the dataset files are stored')
tf.flags.DEFINE_string('checkpoint', './logs',
                        'The directory where the pretrained model is stored')
tf.flags.DEFINE_integer('num_classes', 576,
                        'Number of classes')


FLAGS = tf.app.flags.FLAGS

def get_init_fn(checkpoint_dir):
    checkpoint_exclude_scopes = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(os.path.join(checkpoint_dir, 'inception_v3.ckpt'),
            variables_to_restore)


def main(_):
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Select the dataset
        # veri-776
        dataset = veri_776.get_split('train', FLAGS.dataset_dir)

        data_provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset, 
                        num_readers=2,
                        common_queue_capacity=20 * FLAGS.batch_size, 
                        common_queue_min=10 * FLAGS.batch_size)

        image, label = data_provider.get(['image', 'label'])

        label = tf.decode_raw(label, tf.float32)
        label = tf.reshape(label, [FLAGS.num_classes])

        # Preprocess images
        image = inception_preprocessing.preprocess_image(image, image_size, image_size,
                is_training=True)

        # Training bathes and queue
        images, labels = tf.train.batch(
                [image, label],
                batch_size = FLAGS.batch_size,
                num_threads = 1,
                capacity = 5 * FLAGS.batch_size)

        # Create the model
        # inception v3
        #with slim.arg_scope(inception_v3_arg_scope()):
        #    logits, _ = inception_v3(images, num_classes = FLAGS.num_classes, is_training=True)

        # mobilenet v1
        #with slim.arg_scope(mobilenet_v1_arg_scope()):
        #    logits, _ = mobilenet_v1(images, num_classes = FLAGS.num_classes, is_training=True)

        # mobilenet v1 with self-attention (sa)
        with slim.arg_scope(mobilenet_v1_w_sa_arg_scope()):
            logits, _ = mobilenet_v1_w_sa(images, num_classes = FLAGS.num_classes, is_training=True)

        predictions = tf.nn.softmax(logits, name='prediction')

        """
        # original softmax loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)
        loss = tf.reduce_mean(cross_entropy)
        """

        labels = tf.argmax(labels, axis=1)
        labels = tf.cast(labels, tf.int64)
        loss = cos_loss(logits, labels, 576)

        """
        angular_softmax = op_module.angular_softmax
        #var_weights = tf.Variable(initial_value, trainable=True, name='asoftmax_weights')
        var_weights = tf.Variable(constant_xavier_initializer([FLAGS.num_classes, 1024]), name='asoftmax_weights')
        normed_var_weights = tf.nn.l2_normalize(var_weights, 1, 1e-10, name='weights_normed')
        result = angular_softmax(logits, normed_var_weights, labels, 1, 4, 1000., 0.000025, 35., 0.)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=result[0]))
        """

        # Add summaries
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        train_op = slim.learning.create_train_op(loss, optimizer)

        num_batches = math.ceil(data_provider.num_samples()/float(FLAGS.batch_size)) 
        num_steps = FLAGS.epochs * int(num_batches)
        print('num_steps: {}, num_batches: {}'.format(num_steps,num_batches))

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)

        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        session_config.gpu_options.visible_device_list = "0"
 
        slim.learning.train(
            train_op,
            logdir=FLAGS.log_dir,
            #init_fn=get_init_fn(FLAGS.checkpoint),
            session_config=session_config,
            number_of_steps=num_steps,
            # save summery every 5 min
            save_summaries_secs=300,
            # save checkpoints every 10 min
            save_interval_secs=600
        )
        print ('done!')


if __name__ == '__main__':
    tf.app.run()
