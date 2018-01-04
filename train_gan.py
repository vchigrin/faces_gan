#!/usr/bin/env python

import argparse
import itertools
import logging
import os
import sys
import tensorflow as tf
from tensorflow.python.client import timeline

#TARGET_WIDTH = 128
#TARGET_HEIGH = 192
TARGET_WIDTH = 64
TARGET_HEIGHT = 64
NUM_UPSCALING_BLOCKS = 2
GENERATOR_RES_BLOCK_NUM_CHANNELS = 64
UPSCALING_BLOCK_NUM_CHANNELS = 256
NUM_RES_BLOCKS = 16

DISCRIMINATOR_GRADIENT_CLIP_NORM = 10

NOISE_SIZE = 128
TRUE_IMAGES_BATCH_SIZE = 64
NOISE_BATCH_SIZE = TRUE_IMAGES_BATCH_SIZE
NUM_EPOCHS = 100
DRAGAN_COEF = 10

GENERATOR_PARAMS = 'generator_params'
DISCRIMINATOR_PARAMS = 'discriminator_params'
MODEL_DIR = 'model'
TRACE_DIR = 'trace'

def add_batch_normalization(prev_layer_out, offset=None, scale=None):
  mean, variance = tf.nn.moments(prev_layer_out, axes=[0])
  if offset is not None:
    assert offset.shape == mean.shape
  if scale is not None:
    assert scale.shape == mean.shape
  return tf.nn.batch_normalization(
      prev_layer_out,
      mean,
      variance,
      offset=offset,
      scale=scale,
      variance_epsilon=10 ** -9)


class Network(object):
  def _collection_name(self):
    raise NotImplemented();

  def _get_variable(self, name, shape, mean=0):
    result = tf.get_variable(
      name,
      shape=shape,
      initializer=tf.random_normal_initializer(mean=mean, stddev=0.02))
    tf.add_to_collection(self._collection_name(), result)
    return result

  def _make_batch_normalization_block(self, prev_layer_out):
    offset_and_scale_shape = prev_layer_out.shape.as_list()
    del offset_and_scale_shape[0]  # Batch dimension

    offset = self._get_variable(
        'offset',
        shape=offset_and_scale_shape)
    scale = self._get_variable(
        'scale',
        shape=offset_and_scale_shape,
        mean=1.)
    return add_batch_normalization(
        prev_layer_out, offset=offset, scale=scale)


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

    cur_out = tf.nn.conv2d(
       prev_layer_out,
       kernels1,
       strides=[1, 1, 1, 1],
       padding='SAME')
    with tf.variable_scope('BatchNormalization1'):
      cur_out = self._make_batch_normalization_block(cur_out)
    cur_out = tf.nn.leaky_relu(cur_out)
    cur_out = tf.nn.conv2d(
       cur_out,
       kernels2,
       strides=[1,1,1,1],
       padding='SAME')
    with tf.variable_scope('BatchNormalization2'):
      cur_out = self._make_batch_normalization_block(cur_out)
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
       shape=(3, 3, num_channels, num_channels))

    biases_shape = prev_layer_out.shape.as_list()
    del biases_shape[0]  # Batch dimension
    biases_shape[-1] = num_channels

    biases1 = self._get_variable('biases1', shape=biases_shape)
    biases2 = self._get_variable('biases2', shape=biases_shape)

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

    cur_out = tf.nn.conv2d(
       prev_layer_out,
       kernels,
       strides=[1, stride, stride, 1],
       padding='SAME')

    biases_shape = cur_out.shape.as_list()
    del biases_shape[0]  # Batch dimension

    biases = self._get_variable(
       'biases',
       shape=biases_shape)

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


def run_with_trace_if_need(session, steps_to_run, dump_traces, file_name):
  if not dump_traces:
    return session.run(steps_to_run)
  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  results = session.run(
      steps_to_run, options=run_options, run_metadata=run_metadata)
  tl = timeline.Timeline(run_metadata.step_stats)
  if not os.path.exists(TRACE_DIR):
    os.makedirs(TRACE_DIR)
  with open(os.path.join(TRACE_DIR, file_name), 'w') as f:
    f.write(tl.generate_chrome_trace_format())
  return results


def process(parsed_args):
  tf.set_random_seed(12345)

  in_noise_pipeline = noise_input_pipeline()
  in_true_image_pipeline = true_images_input_pipeline(
      parsed_args.prepared_file_name)

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

  with tf.name_scope('losses'):
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

  optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
  grads_and_vars = optimizer.compute_gradients(
      discriminator_loss, var_list=tf.get_collection(DISCRIMINATOR_PARAMS))

  gradients, variables = zip(*grads_and_vars)
  gradients, original_norm = tf.clip_by_global_norm(
      gradients, DISCRIMINATOR_GRADIENT_CLIP_NORM)
  grads_and_vars = list(zip(gradients, variables))
  discriminator_step = optimizer.apply_gradients(grads_and_vars)

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

  model_saver = tf.train.Saver(max_to_keep=5)
  merged_summaries = tf.summary.merge_all()
  fixed_noise_output = tf.summary.image('fixed_noise_image', generator_output)
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  model_file_prefix = os.path.join(MODEL_DIR, 'GAN')
  with tf.Session() as session:
    with session.as_default():
      fixed_noise = session.run(
          tf.random_uniform(shape=[NOISE_BATCH_SIZE, NOISE_SIZE]))

      session.run(tf.global_variables_initializer())
      if parsed_args.continue_from_checkpoint:
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
        counter_val = 0
        while not coordinator.should_stop():
          for i in xrange(5):
            session.run([next_step_noise, next_step_true_image])
            _, original_norm_value = run_with_trace_if_need(
                session, [discriminator_step, original_norm],
                parsed_args.dump_traces,
                'Discriminator_{}_{}.json'.format(counter_val, i))
            logging.info('Original Discriminator gradient norm {}'.format(original_norm_value))
          session.run([next_step_noise, next_step_true_image])
          _, dl, gl, summary, counter_val = run_with_trace_if_need(
              session,
              [generator_step,
                discriminator_loss,
                generator_loss,
                merged_summaries,
                increment_global_step_counter],
               parsed_args.dump_traces,
               'Generator_{}.json'.format(counter_val))
          sw.add_summary(summary, counter_val)
          if parsed_args.show_fixed_noise_output:
            session.run(in_noise.assign(fixed_noise))
            sw.add_summary(session.run(fixed_noise_output), counter_val)
          logging.info('{} iterations done, Discriminator loss {}, generator loss {}'.format(
              counter_val, dl, gl))
          if dl > 500 or gl > 500:
            logging.info('Model blowed up! stopping...')
            model_saver.save(
                session,
                model_file_prefix,
                global_step=counter_val,
                latest_filename='checkpoint',
                write_meta_graph=False)
            break
          if counter_val % 100 == 0:
            model_saver.save(
                session,
                model_file_prefix,
                global_step=counter_val,
                latest_filename='checkpoint',
                write_meta_graph=False)
      finally:
        sw.close()
        coordinator.request_stop()

def parse_args():
  parser = argparse.ArgumentParser('Trains GAN neural network')
  parser.add_argument(
      'prepared_file_name',
      help='File with source images, in Tensorflow format')
  parser.add_argument(
      '--continue-from-checkpoint', action='store_true',
      help='Continue training from the latest checkpoint')
  parser.add_argument(
      '--show-fixed-noise-output', action='store_true',
      help='Displays generator perfromance on some fixed noise vector')
  parser.add_argument(
      '--dump-traces', action='store_true',
      help='Dumps performance traces in Chrome trace JSON format')
  return parser.parse_args()

def main():
  logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
  process(parse_args())


if __name__ == '__main__':
  main()
