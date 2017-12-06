#!/usr/bin/env python

import os
import sys
import tensorflow as tf
import logging

#TARGET_WIDTH = 128
#TARGET_HEIGH = 192
TARGET_WIDTH = 64
TARGET_HEIGH = 64

NOISE_SIZE = 128
TRUE_IMAGES_BATCH_SIZE = 64
NOISE_BATCH_SIZE = TRUE_IMAGES_BATCH_SIZE
NUM_EPOCHS = 100

GENERATOR_PARAMS = 'generator_params'
DISCRIMINATOR_PARAMS = 'discriminator_params'
MODEL_DIR = 'model'

def add_batch_normalization(prev_layer_out):
  mean, variance = tf.nn.moments(prev_layer_out, axes=[0])
  return tf.nn.batch_normalization(
      prev_layer_out,
      mean,
      variance,
      offset=None,
      scale=None,
      variance_epsilon=10 ** -9)


class Generator(object):
  def _build_dense_layer(self, prev_layer_out, num_units):
    prev_layer_num_units = int(prev_layer_out.shape[-1])
    with tf.variable_scope('Dense'):
      w = tf.get_variable(
        'weights',
        shape=(prev_layer_num_units, num_units),
        initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
      b = tf.get_variable(
        'biases',
        shape=(1, num_units),
        initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
      tf.add_to_collection(GENERATOR_PARAMS, w)
      tf.add_to_collection(GENERATOR_PARAMS, b)
      dense_normalized = add_batch_normalization(tf.matmul(prev_layer_out, w) + b)
      return tf.nn.relu(dense_normalized)

  def _build_output_layer(self, prev_layer_out):
    with tf.variable_scope('Output'):
      prev_layer_num_channels = int(prev_layer_out.shape[-1])
      kernels = tf.get_variable(
         'kernels',
         shape=(3, 3, prev_layer_num_channels, 3),
         initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
      tf.add_to_collection(GENERATOR_PARAMS, kernels)
      conv_out = tf.nn.conv2d(
         prev_layer_out,
         kernels,
         strides=[1,1,1,1],
         padding='SAME')
      assert conv_out.shape.as_list()[1:] == [TARGET_HEIGH, TARGET_WIDTH, 3]
      return tf.nn.sigmoid(conv_out)

  def _build_residual_block(self, prev_layer_out):
    prev_layer_num_channels = int(prev_layer_out.shape[-1])
    kernels = tf.get_variable(
       'kernels',
       shape=(3, 3, prev_layer_num_channels, 16),
       initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
    tf.add_to_collection(GENERATOR_PARAMS, kernels)
    conv_out = tf.nn.conv2d(
       prev_layer_out,
       kernels,
       strides=[1,1,1,1],
       padding='SAME')
    conv_out = add_batch_normalization(conv_out)
    return tf.nn.relu(conv_out)

  def build_generator(self, noise_input):
    with tf.variable_scope('Generator'):
      dense_out = self._build_dense_layer(
          noise_input, num_units=TARGET_WIDTH * TARGET_HEIGH)
      cur_layer_out = tf.reshape(
          dense_out, shape=[-1, TARGET_HEIGH, TARGET_WIDTH, 1])
      for i in xrange(16):
        with tf.variable_scope('Residual_' + str(i)):
          cur_layer_out = self._build_residual_block(
              cur_layer_out)
        assert(cur_layer_out.shape.as_list()[1:] ==
            [TARGET_HEIGH, TARGET_WIDTH, 16])
      return self._build_output_layer(cur_layer_out)


class Discriminator(object):
  def _build_residual_block(self, prev_layer_out, spatial_stride):
    prev_layer_num_channels = int(prev_layer_out.shape[-1])
    kernels = tf.get_variable(
       'kernels',
       shape=(3, 3, prev_layer_num_channels, 16),
       initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
    tf.add_to_collection(DISCRIMINATOR_PARAMS, kernels)
    conv_out = tf.nn.conv2d(
       prev_layer_out,
       kernels,
       strides=[1, spatial_stride, spatial_stride, 1],
       padding='SAME')
    return tf.nn.relu(conv_out)

  def build_discriminator(self, image_input, reuse):
    with tf.variable_scope('Discriminator', reuse=reuse):
      cur_input = image_input
      for i in xrange(10):
        with tf.variable_scope('DiscriminatorResidual_' + str(i), reuse=reuse):
          cur_input = self._build_residual_block(
              cur_input,
              1 if i < 8 else 2)
      num_elements = 1
      for dim in cur_input.shape[1:]:
        num_elements *= int(dim)
      cur_input = tf.reshape(cur_input, shape=[-1, num_elements])
      w = tf.get_variable(
        'weights',
        shape=(cur_input.shape[1], 1),
        initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
      b = tf.get_variable(
        'biases',
        shape=(1, 1),
        initializer=tf.random_normal_initializer(mean=0, stddev=0.2))
      tf.add_to_collection(DISCRIMINATOR_PARAMS, w)
      tf.add_to_collection(DISCRIMINATOR_PARAMS, b)
      return tf.matmul(cur_input, w) + b


def true_images_input_pipeline(file_name):
  filenames_queue = tf.train.string_input_producer(
      [file_name], num_epochs=NUM_EPOCHS)
  reader = tf.TFRecordReader()
  key, value = reader.read(filenames_queue)
  features = tf.parse_single_example(
      value,
      features={
        'image_bytes': tf.FixedLenFeature(
            shape=[],
            dtype=tf.string),
      })
  image = tf.decode_raw(features['image_bytes'], tf.uint8)
  image = tf.reshape(image, shape=(TARGET_HEIGH, TARGET_WIDTH, 3))
  image = tf.cast(image, tf.float32) / 256.
  return tf.train.shuffle_batch(
      [image],
      batch_size=TRUE_IMAGES_BATCH_SIZE,
      capacity=2000,
      min_after_dequeue=1000)


def noise_input_pipeline():
  return tf.random_uniform(shape=[NOISE_BATCH_SIZE, NOISE_SIZE])


def process(input_file_name, should_continue):
  tf.set_random_seed(12345)

  in_noise_pipeline = noise_input_pipeline()
  in_true_image_pipeline = true_images_input_pipeline(input_file_name)

  # Explicity dequeue next data from pilelines and use it during iteration
  # to ensure data we're operating will not change on each access.
  in_noise = tf.get_variable(
      'in_noise',
      shape=in_noise_pipeline.shape,
      dtype=in_noise_pipeline.dtype)
  in_true_image = tf.get_variable(
      'in_true_image',
      shape=in_true_image_pipeline.shape,
      dtype=in_true_image_pipeline.dtype)

  generator_output = Generator().build_generator(in_noise)
  discriminator = Discriminator()
  generated_discriminator_output_without_sigmoid = \
      discriminator.build_discriminator(generator_output, reuse=False)
  true_discriminator_output_without_sigmoid  = \
      discriminator.build_discriminator(in_true_image, reuse=True)

  discriminator_loss = -0.5 * tf.reduce_mean(
      -tf.nn.softplus(-true_discriminator_output_without_sigmoid) +
          (-generated_discriminator_output_without_sigmoid -
            tf.nn.softplus(-generated_discriminator_output_without_sigmoid)))
  generator_loss = -0.5 * tf.reduce_mean(
      -tf.nn.softplus(-generated_discriminator_output_without_sigmoid))

  optimizer = tf.train.AdamOptimizer()
  discriminator_step = optimizer.minimize(
      discriminator_loss, var_list=tf.get_collection(DISCRIMINATOR_PARAMS))
  generator_step = optimizer.minimize(
      generator_loss, var_list=tf.get_collection(GENERATOR_PARAMS))
  global_step_counter = tf.get_variable(
      'global_step_couner', trainable=False, initializer=0)

  tf.summary.scalar('discriminator_loss', discriminator_loss)
  tf.summary.scalar('generator_loss', generator_loss)
  tf.summary.image('generated_images', generator_output)

  next_step_noise = in_noise.assign(in_noise_pipeline)
  next_step_true_image = in_true_image.assign(in_true_image_pipeline)
  increment_global_step_counter = global_step_counter.assign_add(1)

  model_saver = tf.train.Saver(max_to_keep=None)
  merged_summaries = tf.summary.merge_all()
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  model_file_prefix = os.path.join(MODEL_DIR, 'GAN')
  with tf.Session() as session:
    with session.as_default():
      session.run(tf.global_variables_initializer())
      if should_continue:
        latest_checkpoint = tf.train.latest_checkpoint(MODEL_DIR)
        logging.info(
            'NOTE: Restoring saved session from {}....'.format(latest_checkpoint))
        model_saver.restore(session, latest_checkpoint)
      else:
        logging.info('NOTE: Starting new learning session....')
        model_saver.save(session, model_file_prefix, latest_filename='checkpoint')
      session.run(tf.local_variables_initializer())
      coordinator = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=session, coord=coordinator)
      sw = tf.summary.FileWriter('summary_dir', session.graph)
      try:
        while not coordinator.should_stop():
          session.run([next_step_noise, next_step_true_image])
          for _ in xrange(15):
            session.run([next_step_noise, next_step_true_image])
            session.run(discriminator_step)
          session.run([next_step_noise, next_step_true_image])
          session.run(generator_step)
          counter_val = session.run(increment_global_step_counter)
          summary = session.run(merged_summaries)
          sw.add_summary(summary, counter_val)
          logging.info('{} iterations done'.format(counter_val))
          model_saver.save(
              session,
              model_file_prefix,
              global_step=counter_val,
              latest_filename='checkpoint',
              write_meta_graph=False)
      finally:
        sw.close()
        coordinator.request_stop()


def main():
  if len(sys.argv) < 2:
    sys.stderr.write('Trains GAN\n');
    sys.stderr.write(
        'Usage: {} <prepared_tensorflow_file> [--continue]\n'.format(sys.argv[0]))
    sys.exit(1)
  should_continue = False
  if len(sys.argv) == 3 and sys.argv[2] == '--continue':
    should_continue = True
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  process(sys.argv[1], should_continue)


if __name__ == '__main__':
  main()
