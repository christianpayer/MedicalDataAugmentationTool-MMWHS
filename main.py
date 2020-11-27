#!/usr/bin/python

from collections import OrderedDict
import numpy as np
import tensorflow.compat.v1 as tf
import utils.io.image
import tensorflow_train.utils.tensorflow_util
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.losses.semantic_segmentation_losses import softmax_cross_entropy_with_logits
from tensorflow_train.train_loop import MainLoopBase
import utils.sitk_image
from utils.segmentation.segmentation_test import SegmentationTest
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from utils.segmentation.metrics import DiceMetric
from dataset import Dataset
from network import network_scn
from tensorflow_train.utils.summary_handler import create_summary_placeholder


class MainLoop(MainLoopBase):
    def __init__(self, modality, cv):
        super().__init__()
        self.modality = modality
        self.cv = cv
        self.batch_size = 1
        self.learning_rates = [0.00001, 0.000001]
        self.learning_rate_boundaries = [20000]
        self.max_iter = 40000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = self.test_iter
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.0001
        self.num_labels = 8
        self.data_format = 'channels_first'
        self.channel_axis = 1
        self.save_debug_images = False

        self.has_validation_groundtruth = cv != 0
        self.base_folder = 'mmwhs_dataset'
        self.image_size = [64, 64, 64]
        if modality == 'ct':
            self.image_spacing = [3, 3, 3]
        else:
            self.image_spacing = [4, 4, 4]
        self.input_gaussian_sigma = 1.0
        self.label_gaussian_sigma = 1.0
        self.use_landmarks = True

        self.output_folder = './output/scn_' + modality + '_' + str(cv) + '/' + self.output_folder_timestamp()

        dataset_parameters = dict(image_size=self.image_size,
                                  image_spacing=self.image_spacing,
                                  base_folder=self.base_folder,
                                  cv=self.cv,
                                  modality=self.modality,
                                  input_gaussian_sigma=self.input_gaussian_sigma,
                                  label_gaussian_sigma=self.label_gaussian_sigma,
                                  use_landmarks=self.use_landmarks,
                                  num_labels=self.num_labels,
                                  data_format=self.data_format,
                                  save_debug_images=self.save_debug_images)

        self.dataset = Dataset(**dataset_parameters)

        self.dataset_train = self.dataset.dataset_train()
        self.dataset_val = self.dataset.dataset_val()
        self.dataset_val = self.dataset.dataset_val()
        self.files_to_copy = ['main.py', 'network.py', 'dataset.py']
        self.dice_names = list(map(lambda x: 'dice_{}'.format(x), range(self.num_labels)))
        self.additional_summaries_placeholders_val = dict([(name, create_summary_placeholder(name)) for name in self.dice_names])
        self.loss_function = softmax_cross_entropy_with_logits
        self.network = network_scn

    def init_networks(self):
        network_image_size = self.image_size

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('labels', [self.num_labels] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('labels', network_image_size + [self.num_labels])])

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build train graph
        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size)
        data, mask = self.train_queue.dequeue()
        prediction, _, _ = training_net(data, num_labels=self.num_labels, is_training=True, data_format=self.data_format)
        # losses
        self.loss_net = self.loss_function(labels=mask, logits=prediction, data_format=self.data_format)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.reg_constant > 0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.loss_reg = self.reg_constant * tf.add_n(reg_losses)
                self.loss = self.loss_net + self.loss_reg
            else:
                self.loss = self.loss_net

        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg)])

        # solver
        global_step = tf.Variable(self.current_iter)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

        # build val graph
        val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries, shape_prefix=[1])
        self.data_val = val_placeholders['image']
        self.prediction_val, self.local_prediction_val, self.spatial_prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, data_format=self.data_format)
        self.prediction_softmax_val = tf.nn.softmax(self.prediction_val, axis=1 if self.data_format == 'channels_first' else 4)

        if self.has_validation_groundtruth:
            self.mask_val = val_placeholders['labels']
            # losses
            self.loss_val = self.loss_function(labels=self.mask_val, logits=self.prediction_val, data_format=self.data_format)
            self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg)])

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
        segmentation_statistics = SegmentationStatistics(labels,
                                                         self.output_folder_for_current_iteration(),
                                                         metrics={'dice': DiceMetric()})
        num_entries = self.dataset_val.num_entries()
        for i in range(num_entries):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            generators = dataset_entry['generators']
            transformations = dataset_entry['transformations']
            if self.has_validation_groundtruth:
                feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0),
                             self.mask_val: np.expand_dims(generators['labels'], axis=0)}
                # run loss and update loss accumulators
                run_tuple = self.sess.run((self.prediction_softmax_val, self.local_prediction_val, self.spatial_prediction_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(),
                                          feed_dict=feed_dict)
            else:
                feed_dict = {self.data_val: np.expand_dims(generators['data'], axis=0)}
                # run loss and update loss accumulators
                run_tuple = self.sess.run((self.prediction_softmax_val,), feed_dict=feed_dict)

            prediction = np.squeeze(run_tuple[0], axis=0)
            #local_prediction = np.squeeze(run_tuple[1], axis=0)
            #spatial_prediction = np.squeeze(run_tuple[2], axis=0)
            input = datasources['image']
            transformation = transformations['image']
            prediction_labels, prediction_sitk = segmentation_test.get_label_image(prediction, input, self.image_spacing, transformation, return_transformed_sitk=True)
            utils.io.image.write(prediction_labels, self.output_file_for_current_iteration(current_id + '.mha'))
            origin = transformation.TransformPoint(np.zeros(3, np.float64))
            utils.io.image.write_multichannel_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            #utils.io.image.write_multichannel_np(local_prediction, self.output_file_for_current_iteration(current_id + '_local_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            #utils.io.image.write_multichannel_np(spatial_prediction, self.output_file_for_current_iteration(current_id + '_spatial_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            if self.has_validation_groundtruth:
                groundtruth = datasources['labels']
                segmentation_statistics.add_labels(current_id, prediction_labels, groundtruth)
            tensorflow_train.utils.tensorflow_util.print_progress_bar(i, num_entries, prefix='Testing ', suffix=' complete')

        # finalize loss values
        if self.has_validation_groundtruth:
            segmentation_statistics.finalize()
            dice_list = segmentation_statistics.get_metric_mean_list('dice')
            dice_dict = OrderedDict(list(zip(self.dice_names, dice_list)))
            self.val_loss_aggregator.finalize(self.current_iter, summary_values=dice_dict)


if __name__ == '__main__':
    # cv 1, 2, 3 for cross validation
    # cv 0 for training on full training set and testing on test set
    for i in [1, 2, 3, 0]:
        loop = MainLoop('ct', i)
        loop.run()
