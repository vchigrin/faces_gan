#!/usr/bin/env python

import itertools
import logging
import os
import sys
import tensorflow as tf

#TARGET_WIDTH = 128
#TARGET_HEIGH = 192
TARGET_WIDTH = 64
TARGET_HEIGHT = 64
NUM_UPSCALING_BLOCKS = 2
GENERATOR_RES_BLOCK_NUM_CHANNELS = 64
UPSCALING_BLOCK_NUM_CHANNELS = 256
NUM_RES_BLOCKS = 16

NOISE_SIZE = 128
TRUE_IMAGES_BATCH_SIZE = 64
NOISE_BATCH_SIZE = TRUE_IMAGES_BATCH_SIZE
NUM_EPOCHS = 100
DRAGAN_COEF = 10

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


class Network(object):
  def _collection_name(self):
    raise NotImplemented();

  def _get_variable(self, name, shape):
    result = tf.get_variable(
      name,
      shape=shape,
      initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
    tf.add_to_collection(self._collection_name(), result)
    return result


class Generator(Network):
  def __init__(self):
    super(Generator, self).__init__()

  def _collection_name(self):
    return GENERATOR_PARAMS

  def _build_dense_layer(self, prev_layer_out, num_units):
    prev_layer_num_units = int(prev_layer_out.shape[-1])
    with tf.variable_scope('Dense'):
      w = self._get_variable(
        'weights',
        shape=(prev_layer_num_units, num_units))
      b = self._get_variable(
        'biases',
        shape=(1, num_units))
      dense_normalized = add_batch_normalization(
          tf.matmul(prev_layer_out, w) + b)
      return tf.nn.leaky_relu(dense_normalized)

  def _build_output_layer(self, prev_layer_out):
    with tf.variable_scope('Output'):
      prev_layer_num_channels = int(prev_layer_out.shape[-1])
      kernels = self._get_variable(
         'kernels',
         shape=(9, 9, prev_layer_num_channels, 3))
      biases = self._get_variable(
          'biases',
          shape=(1, 1, 3))
      conv_out = tf.nn.conv2d(
         prev_layer_out,
         kernels,
         strides=[1, 1, 1, 1],
         padding='SAME')
      conv_out = conv_out + biases
      assert conv_out.shape.as_list()[1:] == [TARGET_HEIGHT, TARGET_WIDTH, 3]
      return tf.nn.sigmoid(conv_out)

  def _build_residual_block(self, prev_layer_out):
    prev_layer_num_channels = int(prev_layer_out.shape[-1])
    kernels1 = self._get_variable(
       'kernels1',
       shape=(3, 3, prev_layer_num_channels, GENERATOR_RES_BLOCK_NUM_CHANNELS))

    kernels2 = self._get_variable(
       'kernels2',
       shape=(3, 3, GENERATOR_RES_BLOCK_NUM_CHANNELS, GENERATOR_RES_BLOCK_NUM_CHANNELS))

    biases1 = self._get_variable(
        'biases1',
        shape=(1, 1, GENERATOR_RES_BLOCK_NUM_CHANNELS))

    biases2 = self._get_variable(
        'biases2',
        shape=(1, 1, GENERATOR_RES_BLOCK_NUM_CHANNELS))

    cur_out = tf.nn.conv2d(
       prev_layer_out,
       kernels1,
       strides=[1, 1, 1, 1],
       padding='SAME')
    cur_out = cur_out + biases1
    cur_out = add_batch_normalization(cur_out)
    cur_out = tf.nn.leaky_relu(cur_out)
    cur_out = tf.nn.conv2d(
       cur_out,
       kernels2,
       strides=[1,1,1,1],
       padding='SAME')
    cur_out = cur_out + biases2
    cur_out = add_batch_normalization(cur_out)
    assert cur_out.shape == prev_layer_out.shape
    cur_out = cur_out + prev_layer_out
    return cur_out

  def _build_upscaling_block(self, prev_layer_out):
    prev_layer_shape = [int(x) for x in prev_layer_out.shape]
    kernels = self._get_variable(
       'kernels',
       shape=(3, 3, prev_layer_shape[-1], UPSCALING_BLOCK_NUM_CHANNELS))
    biases = self._get_variable(
       'biases',
       shape=(1, 1, UPSCALING_BLOCK_NUM_CHANNELS))

    cur_out = tf.nn.conv2d(
        prev_layer_out,
        kernels,
        strides=[1, 1, 1, 1],
        padding='SAME')
    cur_out = cur_out + biases
    cur_out = tf.contrib.periodic_resample.periodic_resample(
        cur_out,
        [prev_layer_shape[0],
         prev_layer_shape[1] * 2,
         prev_layer_shape[2] * 2,
         None])
    cur_out = add_batch_normalization(cur_out)
    return tf.nn.leaky_relu(cur_out)

  def build_generator(self, noise_input):
    with tf.variable_scope('Generator'):
      res_block_width = TARGET_WIDTH / (2 ** NUM_UPSCALING_BLOCKS)
      res_block_height = TARGET_HEIGHT / (2 ** NUM_UPSCALING_BLOCKS)
      dense_out = self._build_dense_layer(
          noise_input,
          num_units=res_block_width * res_block_height * GENERATOR_RES_BLOCK_NUM_CHANNELS)
      cur_layer_out = tf.reshape(
          dense_out,
          shape=[-1, res_block_width, res_block_height, GENERATOR_RES_BLOCK_NUM_CHANNELS])
      for i in xrange(NUM_RES_BLOCKS):
        with tf.variable_scope('Residual_' + str(i)):
          cur_layer_out = self._build_residual_block(
              cur_layer_out)
        assert(cur_layer_out.shape.as_list()[1:] ==
            [res_block_width, res_block_height, GENERATOR_RES_BLOCK_NUM_CHANNELS])
      expected_width = res_block_width
      expected_height = res_block_height
      for i in xrange(NUM_UPSCALING_BLOCKS):
        with tf.variable_scope('Upscaling_' + str(i)):
          cur_layer_out = self._build_upscaling_block(cur_layer_out)
          expected_width *= 2
          expected_height *= 2

        assert(cur_layer_out.shape.as_list()[1:] ==
            [expected_width, expected_height, UPSCALING_BLOCK_NUM_CHANNELS / 4])
      return self._build_output_layer(cur_layer_out)


class Discriminator(Network):
  def __init__(self):
    super(Discriminator, self).__init__()

  def _collection_name(self):
    return DISCRIMINATOR_PARAMS

  def _build_residual_block(self, prev_layer_out, num_channels):
    prev_layer_num_channels = int(prev_layer_out.shape[-1])
    kernels1 = self._get_variable(
       'kernels1',
       shape=(3, 3, prev_layer_num_channels, num_channels))

    kernels2 = self._get_variable(
       'kernels2',
       shape=(3, 3, prev_layer_num_channels, num_channels))

    biases1 = self._get_variable(
        'biases1',
        shape=(1, 1, num_channels))

    biases2 = self._get_variable(
        'biases2',
        shape=(1, 1, num_channels))

    cur_out = tf.nn.conv2d(
       prev_layer_out,
       kernels1,
       strides=[1, 1, 1, 1],
       padding='SAME')
    cur_out = cur_out + biases1
    cur_out = tf.nn.leaky_relu(cur_out)
    cur_out = tf.nn.conv2d(
       cur_out,
       kernels2,
       strides=[1, 1, 1, 1],
       padding='SAME')
    cur_out = cur_out + biases2
    cur_out = tf.nn.leaky_relu(cur_out)

    cur_out = cur_out + prev_layer_out

    return tf.nn.leaky_relu(cur_out)

  def _build_conv_block(self, prev_layer_out, kernel_size, num_channels, stride):
    prev_layer_num_channels = int(prev_layer_out.shape[-1])
    kernels = self._get_variable(
       'kernels',
       shape=(kernel_size, kernel_size, prev_layer_num_channels, num_channels))
    biases = self._get_variable(
       'biases',
       shape=(1, 1, num_channels))

    cur_out = tf.nn.conv2d(
       prev_layer_out,
       kernels,
       strides=[1, stride, stride, 1],
       padding='SAME')
    cur_out = cur_out + biases
    return tf.nn.leaky_relu(cur_out)

  def build_discriminator(self, image_input, reuse):
    with tf.variable_scope('Discriminator', reuse=reuse):
      cur_layer_out = image_input
      layer_index = 0
      cur_num_channels = 32
      for _ in xrange(5):
        with tf.variable_scope('DiscriminatorConv' + str(layer_index), reuse=reuse):
          layer_index += 1
          cur_layer_out = self._build_conv_block(cur_layer_out,
              kernel_size=4 if cur_num_channels < 256 else 3,
              num_channels=cur_num_channels,
              stride=2)
        for i in xrange(2):
          with tf.variable_scope('DiscriminatorResidual_' + str(layer_index), reuse=reuse):
            layer_index += 1
            cur_layer_out = self._build_residual_block(
                cur_layer_out,
                num_channels=cur_num_channels)
        cur_num_channels *= 2
      assert cur_num_channels == 1024
      with tf.variable_scope('DiscriminatorConv' + str(layer_index), reuse=reuse):
        layer_index += 1
        cur_layer_out = self._build_conv_block(cur_layer_out,
            kernel_size=3, num_channels=cur_num_channels, stride=2)

      num_elements = 1
      for dim in cur_layer_out.shape[1:]:
        num_elements *= int(dim)
      cur_layer_out = tf.reshape(cur_layer_out, shape=[-1, num_elements])
      w = self._get_variable(
        'weights',
        shape=(cur_layer_out.shape[1], 1))
      b = self._get_variable(
        'biases',
        shape=(1, 1))
      return tf.matmul(cur_layer_out, w) + b


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
  image = tf.reshape(image, shape=(TARGET_HEIGHT, TARGET_WIDTH, 3))
  image = tf.cast(image, tf.float32) / 256.
  return tf.train.shuffle_batch(
      [image],
      batch_size=TRUE_IMAGES_BATCH_SIZE,
      capacity=2000,
      min_after_dequeue=1000)


def noise_input_pipeline():
  return tf.random_uniform(shape=[NOISE_BATCH_SIZE, NOISE_SIZE])


def compute_perturbed_images(in_true_images):
  mean, variance = tf.nn.moments(in_true_images, axes=[0])
  C = tf.random_uniform(shape=(1, 1))
  std_dev = tf.sqrt(variance)
  return in_true_images + C * 0.5 * std_dev


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

  discriminator_on_true_images_output = (
      tf.reduce_mean(tf.nn.sigmoid(true_discriminator_output_without_sigmoid)))
  discriminator_on_generated_images_output = (
      tf.reduce_mean(tf.nn.sigmoid(generated_discriminator_output_without_sigmoid)))

  perturbed_true_images = compute_perturbed_images(in_true_image)
  discriminator_on_perturbed_images_without_sigmoid = (
          discriminator.build_discriminator(perturbed_true_images, reuse=True))

  # Actually DRAGAN requires gradient of discriminator output with sigmoid,
  # but try to use logits gradient instead to overcome division by zero error.
  gradients = tf.gradients(
      discriminator_on_perturbed_images_without_sigmoid,
          [perturbed_true_images])[0]
  l2_gradients_minus_one = (1 - tf.sqrt(
      tf.reduce_sum(tf.square(gradients), axis=(1, 2, 3))))
  dragan_loss = DRAGAN_COEF * tf.square(l2_gradients_minus_one)

  discriminator_loss = 0.5 * tf.reduce_mean(
      tf.nn.softplus(-true_discriminator_output_without_sigmoid) +
          generated_discriminator_output_without_sigmoid +
            tf.nn.softplus(-generated_discriminator_output_without_sigmoid) +
            dragan_loss)

  generator_loss = 0.5 * tf.reduce_mean(
      tf.nn.softplus(-generated_discriminator_output_without_sigmoid))

  optimizer = tf.train.AdamOptimizer(
      learning_rate=0.0002,
      beta1=0.5,
      epsilon=1e-04)
  discriminator_step = optimizer.minimize(
      discriminator_loss, var_list=tf.get_collection(DISCRIMINATOR_PARAMS))
  generator_step = optimizer.minimize(
      generator_loss, var_list=tf.get_collection(GENERATOR_PARAMS))

  global_step_counter = tf.get_variable(
      'global_step_couner', trainable=False, initializer=0)

  tf.summary.scalar('discriminator_loss', discriminator_loss)
  tf.summary.scalar('generator_loss', generator_loss)
  tf.summary.image('generated_images', generator_output)

  tf.summary.scalar('discriminator_on_true_images_output',
      discriminator_on_true_images_output);
  tf.summary.scalar('discriminator_on_generated_images_output',
      discriminator_on_generated_images_output);

  next_step_noise = in_noise.assign(in_noise_pipeline)
  next_step_true_image = in_true_image.assign(in_true_image_pipeline)
  increment_global_step_counter = global_step_counter.assign_add(1)

  model_saver = tf.train.Saver(max_to_keep=4)
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
          for _ in xrange(5):
            session.run([next_step_noise, next_step_true_image])
            session.run(discriminator_step)
          session.run([next_step_noise, next_step_true_image])
          session.run(generator_step)
          counter_val = session.run(increment_global_step_counter)
          summary = session.run(merged_summaries)
          sw.add_summary(summary, counter_val)
          logging.info('{} iterations done'.format(counter_val))
          if counter_val % 5 == 0:
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
