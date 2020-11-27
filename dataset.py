import os

import numpy as np
import SimpleITK as sitk
from datasets.graph_dataset import GraphDataset

from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from graph.node import LambdaNode
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation
from utils.np_image import split_label_image
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from transformations.intensity.np.normalize import normalize_robust


class Dataset(object):
    """
    The dataset that processes files from the MMWHS challenge.
    """
    def __init__(self,
                 image_size,
                 image_spacing,
                 base_folder,
                 cv,
                 modality,
                 input_gaussian_sigma=1.0,
                 label_gaussian_sigma=1.0,
                 use_landmarks=True,
                 num_labels=8,
                 image_folder=None,
                 setup_folder=None,
                 image_filename_postfix='_image',
                 image_filename_extension='.mha',
                 labels_filename_postfix='_label_sorted',
                 labels_filename_extension='.mha',
                 landmarks_file=None,
                 data_format='channels_first',
                 save_debug_images=False):
        """
        Initializer.
        :param image_size: Network input image size.
        :param image_spacing: Network input image spacing.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (1, 2, 3). Or 0 if full training/testing.
        :param modality: Either 'ct' or 'mr'.
        :param input_gaussian_sigma: Sigma value for input smoothing.
        :param label_gaussian_sigma: Sigma value for label smoothing.
        :param use_landmarks: If True, center on loaded landmarks, otherwise use image center.
        :param num_labels: The number of output labels.
        :param image_folder: If set, use this folder for loading the images, otherwise use MMWHS default.
        :param setup_folder: If set, use this folder for loading the setup files, otherwise use MMWHS default.
        :param image_filename_postfix: The image filename postfix.
        :param image_filename_extension: The image filename extension.
        :param labels_filename_postfix: The labels filename postfix.
        :param labels_filename_extension: The labels filename extension.
        :param landmarks_file: If set, use this file for loading image landmarks, otherwise us MMWHS default.
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.cv = cv
        self.modality = modality
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.use_landmarks = use_landmarks
        self.num_labels = num_labels
        self.image_filename_postfix = image_filename_postfix
        self.image_filename_extension = image_filename_extension
        self.labels_filename_postfix = labels_filename_postfix
        self.labels_filename_extension = labels_filename_extension
        self.data_format = data_format
        self.save_debug_images = save_debug_images
        self.dim = 3
        self.image_base_folder = image_folder or os.path.join(self.base_folder, modality + '_mha')
        self.setup_base_folder = setup_folder or os.path.join(self.base_folder, 'setup')
        self.landmarks_file = landmarks_file or os.path.join(self.setup_base_folder, modality + '_seg_center_rai_w_spacing.csv')

        if modality == 'ct':
            self.postprocessing_random = self.intensity_postprocessing_ct_random
            self.postprocessing = self.intensity_postprocessing_ct
        else:  # if modality == 'mr':
            self.postprocessing_random = self.intensity_postprocessing_mr_random
            self.postprocessing = self.intensity_postprocessing_mr
        if cv > 0:
            self.cv_folder = os.path.join(self.setup_base_folder, os.path.join(modality + '_cv', str(cv)))
            self.train_file = os.path.join(self.cv_folder, 'train.txt')
            self.test_file = os.path.join(self.cv_folder, 'test.txt')
        else:
            self.train_file = os.path.join(self.setup_base_folder, modality + '_train_all.txt')
            self.test_file = os.path.join(self.setup_base_folder, modality + '_test_all.txt')

    def datasources(self, iterator, cached):
        """
        Returns the data sources that load data.
        {
        'image:' (Cached)ImageDataSource that loads the image files.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        'mask:' (Cached)ImageDataSource that loads the groundtruth labels.
        }
        :param iterator: The iterator.
        :param cached: If True, use CachedImageDataSource.
        :return: A dict of data sources.
        """
        preprocessing = lambda image: gaussian_sitk(image, self.input_gaussian_sigma)
        image_data_source = CachedImageDataSource if cached else ImageDataSource
        image_datasource = image_data_source(self.image_base_folder, '', self.image_filename_postfix, self.image_filename_extension, preprocessing=preprocessing, name='image', parents=[iterator])
        landmark_datasource = LandmarkDataSource(self.landmarks_file, 1, self.dim, name='landmarks', parents=[iterator])
        mask_datasource = image_data_source(self.image_base_folder, '', self.labels_filename_postfix, self.labels_filename_extension, sitk_pixel_type=sitk.sitkUInt8, name='labels', parents=[iterator])
        return {'image': image_datasource,
                'landmarks': landmark_datasource,
                'labels': mask_datasource}

    def data_generators(self, datasources, transformation, image_post_processing, mask_post_processing):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param datasources: The datasources dictionary (see self.datasources()).
        :param transformation: The spatial transformation.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param mask_post_processing: The np postprocessing function fo the mask data generator
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='linear', post_processing_np=image_post_processing, data_format=self.data_format, name='image', parents=[datasources['image'], transformation])
        mask_image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='nearest', post_processing_np=mask_post_processing, data_format=self.data_format, name='labels', parents=[datasources['labels'], transformation])
        return {'data': image_generator,
                'labels': mask_image_generator}

    def split_labels(self, image):
        """
        Splits a groundtruth label image into a stack of one-hot encoded images.
        :param image: The groundtruth label image.
        :return: The one-hot encoded image.
        """
        split = split_label_image(np.squeeze(image, 0), list(range(self.num_labels)), np.uint8)
        split_smoothed = [gaussian(i, self.label_gaussian_sigma) for i in split]
        smoothed = np.stack(split_smoothed, 0)
        image_smoothed = np.argmax(smoothed, axis=0)
        split = split_label_image(image_smoothed, list(range(self.num_labels)), np.uint8)
        return np.stack(split, 0)

    def intensity_postprocessing_ct_random(self, image):
        """
        Intensity postprocessing for CT input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               random_shift=0.2,
                               random_scale=0.2,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def intensity_postprocessing_ct(self, image):
        """
        Intensity postprocessing for CT input.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)

    def intensity_postprocessing_mr_random(self, image):
        """
        Intensity postprocessing for MR input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(random_shift=0.2,
                               random_scale=0.4,
                               clamp_min=-1.0)(image)

    def intensity_postprocessing_mr(self, image):
        """
        Intensity postprocessing for MR input.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(clamp_min=-1.0)(image)

    def spatial_transformation_augmented(self, datasources):
        """
        The spatial image transformation with random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        if self.use_landmarks:
            transformation_list.append(landmark.Center(self.dim, True))
            kwparents['landmarks'] = datasources['landmarks']
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.extend([translation.Random(self.dim, [20, 20, 20]),
                                    rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                    deformation.Output(self.dim, [8, 8, 8], 15, self.image_size, self.image_spacing)])
        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return LambdaNode(lambda comp: sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, size=self.image_size, outputSpacing=self.image_spacing)), name='image', kwparents={'comp': comp})

    def spatial_transformation(self, datasources):
        """
        The spatial image transformation without random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        if self.use_landmarks:
            transformation_list.append(landmark.Center(self.dim, True))
            kwparents['landmarks'] = datasources['landmarks']
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.append(translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing))
        return composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_file, random=True, use_shuffle=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, True)
        reference_transformation = self.spatial_transformation_augmented(sources)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing_random, self.split_labels)

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.test_file, random=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, False)
        reference_transformation = self.spatial_transformation(sources)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing, self.split_labels)

        if self.cv == 0:
            del sources['labels']
            del generators['labels']

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)

    def dataset_inference(self):
        """
        Returns the inference dataset. No random augmentation is performed.
        :return: The inference dataset.
        """
        iterator = 'iterator'
        sources = self.datasources(iterator, False)
        reference_transformation = self.spatial_transformation(sources)
        generators = self.data_generators(sources, reference_transformation, self.postprocessing, None)

        del sources['labels']
        del generators['labels']

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)
