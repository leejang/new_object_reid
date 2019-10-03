from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math

# dataset Veri-776
import veri_776 
import inception_preprocessing

from inception_v3 import inception_v3, inception_v3_arg_scope

slim = tf.contrib.slim
image_size = inception_v3.default_image_size


tf.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.flags.DEFINE_integer('epochs', 10, 'Number of training epochs')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate')

tf.flags.DEFINE_string('log_dir', './logs', 
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
        with slim.arg_scope(inception_v3_arg_scope()):
            logits, _ = inception_v3(images, num_classes = FLAGS.num_classes, is_training=True)

        predictions = tf.nn.softmax(logits, name='prediction')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)
        loss = tf.reduce_mean(cross_entropy)

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
            save_summaries_secs=50,
            save_interval_secs=50
        )
        print ('done!')


if __name__ == '__main__':
    tf.app.run()
