#!/usr/bin/python

import os
from collections import OrderedDict
from glob import glob

import numpy as np
import tensorflow.compat.v1 as tf
import utils.io.image
import utils.sitk_image
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.tensorflow_util import create_placeholders
from utils.segmentation.segmentation_test import SegmentationTest

from dataset import Dataset
from network import network_scn


class MainLoop(MainLoopBase):
    def __init__(self, modality, image_folder, image_postfix, image_extension, landmarks_file, load_model_filename, output_folder):
        super().__init__()
        self.modality = modality
        self.image_folder = image_folder
        self.image_postfix = image_postfix
        self.image_extension = image_extension
        self.load_model_filename = load_model_filename
        self.landmarks_file = landmarks_file
        self.output_folder = output_folder

        self.num_labels = 8
        self.data_format = 'channels_first'
        self.channel_axis = 1
        self.save_debug_images = False

        self.image_size = [64, 64, 64]
        if modality == 'ct':
            self.image_spacing = [3, 3, 3]
        else:
            self.image_spacing = [4, 4, 4]
        self.input_gaussian_sigma = 1.0
        self.label_gaussian_sigma = 1.0
        self.use_landmarks = landmarks_file is not None

        dataset_parameters = dict(image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  base_folder='',
                                  image_folder=self.image_folder,
                                  cv=0,
                                  modality=self.modality,
                                  input_gaussian_sigma=self.input_gaussian_sigma,
                                  label_gaussian_sigma=self.label_gaussian_sigma,
                                  use_landmarks=self.use_landmarks,
                                  num_labels=self.num_labels,
                                  image_filename_postfix=self.image_postfix,
                                  image_filename_extension=self.image_extension,
                                  landmarks_file=self.landmarks_file,
                                  data_format=self.data_format,
                                  save_debug_images=self.save_debug_images)

        self.dataset = Dataset(**dataset_parameters)
        self.dataset_inference = self.dataset.dataset_inference()
        self.network = network_scn

    def init_networks(self):
        network_image_size = self.image_size

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1])])

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        val_placeholders = create_placeholders(data_generator_entries, shape_prefix=[1])
        self.data_val = val_placeholders['image']
        self.prediction_val, self.local_prediction_val, self.spatial_prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, data_format=self.data_format)
        self.prediction_softmax_val = tf.nn.softmax(self.prediction_val, axis=1 if self.data_format == 'channels_first' else 4)

    def test(self):
        print('Testing...')
        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        labels = list(range(self.num_labels))
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator='cubic',
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)
        filenames = glob(os.path.join(self.image_folder, '*' + self.image_postfix + self.image_extension))
        for filename in sorted(filenames):
            current_id = os.path.split(filename)[-1]
            current_id = current_id[:current_id.find(self.image_postfix + self.image_extension)]
            dataset_entry = self.dataset_inference.get({'image_id': current_id})
            datasources = dataset_entry['datasources']
            generators = dataset_entry['generators']
            transformations = dataset_entry['transformations']
            feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0)}
            run_tuple = self.sess.run((self.prediction_softmax_val, self.local_prediction_val, self.spatial_prediction_val), feed_dict=feed_dict)
            prediction = np.squeeze(run_tuple[0], axis=0)
            #local_prediction = np.squeeze(run_tuple[1], axis=0)
            #spatial_prediction = np.squeeze(run_tuple[2], axis=0)

            input = datasources['image']
            transformation = transformations['image']

            prediction_labels = segmentation_test.get_label_image(prediction, input, self.image_spacing, transformation)
            utils.io.image.write(prediction_labels, self.output_file_for_current_iteration(current_id + '.mha'))
            origin = transformation.TransformPoint(np.zeros(3, np.float64))
            utils.io.image.write_multichannel_np(generators['image'], self.output_file_for_current_iteration(current_id + '_input_image.mha'), data_format=self.data_format, image_type=np.float32, spacing=self.image_spacing, origin=origin)
            utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), data_format=self.data_format, image_type=np.float32, spacing=self.image_spacing, origin=origin)
            #utils.io.image.write_multichannel_np(local_prediction, self.output_file_for_current_iteration(current_id + '_local_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            #utils.io.image.write_multichannel_np(spatial_prediction, self.output_file_for_current_iteration(current_id + '_spatial_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)


if __name__ == '__main__':
    modality = 'ct'  # important for intensity preprocessing
    image_folder = 'TODO'  # e.g., 'mmwhs_dataset/ct_mha/'
    image_postfix = '_image'
    image_extension = '.mha'
    landmarks_file = 'TODO'  # if None, center images; e.g., 'mmwhs_dataset/setup/ct_seg_center_rai_w_spacing.csv'
    load_model_filename = 'TODO'
    output_folder = 'inference'

    loop = MainLoop(modality,
                    image_folder,
                    image_postfix,
                    image_extension,
                    landmarks_file,
                    load_model_filename,
                    output_folder)
    loop.run_test()
