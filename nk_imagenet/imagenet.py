''' Classes for performing image clustering '''

import logging
import os
import numpy as np
from keras.applications import inception_v3

from .image_utils import image_array_from_path, image_array_from_url, load_image_url


class ImagenetModel:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, target_size=(299, 299)):
        # TODO allow for other keras imagenet models
        self.target_size = target_size
        self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        self.preprocess = inception_v3.preprocess_input
        self.decode = inception_v3.decode_predictions

    def save_url_images(self, image_urls, write_dir='images'):
        if not os.path.isdir(write_dir):
            logging.info(f'creating write directory for downloaded images: {write_dir}')
            os.makedirs(write_dir)

        for url in image_urls:
            img = load_image_url(url)
            filename = ''.join([ch for ch in url if str.isalnum(ch)]) + '.png'
            filepath = os.path.join(write_dir, filename)
            img.save(filepath)

    # TODO cache url/path -> feature funcs?
    def get_features_from_paths(self, image_paths, n_channels=None):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        images_array = np.array([image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths])
        return self.get_features(images_array, n_channels=n_channels)

    def get_features_from_urls(self, image_urls, n_channels=None):
        ''' takes a list of image urls and returns the features resulting from applying the imagenet model to
        successfully downloaded images along with the urls that were successful.
        '''
        logging.info(f'getting {len(image_urls)} images from urls')
        images_array = [image_array_from_url(url, target_size=self.target_size) for url in image_urls]
        # filter out unsuccessful image urls which output None in the list of
        url_to_image = {url: img for url, img in zip(image_urls, images_array) if img is not None}
        images_array = np.array(list(url_to_image.values()))

        logging.info(f'getting features from image arrays')
        features = self.get_features(images_array, n_channels=n_channels)
        return features, list(url_to_image.keys())

    def get_features(self, images_array, n_channels=None):
        ''' takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array '''
        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')
        logging.info(f'preprocessing {images_array.shape[0]} images')
        images_array = self.preprocess(images_array)
        logging.info(f'computing image features')
        image_features = self.model.predict(images_array)
        if n_channels:
            logging.info(f'truncated to first {n_channels} channels')
            # if n_channels is specified, only keep that number of channels
            image_features = image_features[:, :, :, :n_channels]

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))
