''' Classes for performing image clustering '''

import logging
import os
from urllib.parse import urlsplit
import pickle

import numpy as np
from cachetools import LRUCache
from keras.applications import inception_v3, xception, mobilenetv2

from .utils import image_array_from_path, image_array_from_url, load_image_url, partition


class ImagenetModel:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, target_size=(299, 299), pooling=None, n_channels=None, cache_size=1e4, model='xception', cache_path='imagenet-cache.pkl'):
        self.target_size = target_size
        self.n_channels = n_channels
        self.cache_path = cache_path  # can be set to None to not load cache even if default file is present

        if self.cache_path and os.path.isfile(self.cache_path):
            self.load_cache()
        else:
            self.cache = LRUCache(cache_size)

        if model == 'xception':
            self.model = xception.Xception(weights='imagenet', include_top=False, pooling=pooling)
            self.preprocess = xception.preprocess_input
            self.decode = xception.decode_predictions
        elif model == 'inception_v3':
            self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
            self.preprocess = inception_v3.preprocess_input
            self.decode = inception_v3.decode_predictions
        elif model == 'mobilenet_v2':
            self.model = mobilenetv2.MobileNetV2(weights='imagenet', include_top=False, pooling=pooling)
            self.preprocess = mobilenetv2.preprocess_input
            self.decode = mobilenetv2.decode_predictions
        else:
            raise Exception('model option not implemented')

    def save_cache(self, cache_path=None):
        cache_path = cache_path if cache_path else self.cache_path
        with open(cache_path, 'wb') as pkl_file:
            pickle.dump(self.cache, pkl_file)

    def load_cache(self, cache_path=None):
        cache_path = cache_path if cache_path else self.cache_path
        with open(cache_path, 'rb') as pkl_file:
            self.cache = pickle.load(pkl_file)

    def save_url_images(self, image_urls, write_dir='images'):
        ''' takes a list of urls then downloads and saves the image files to write_dir '''
        if not os.path.isdir(write_dir):
            logging.info(f'creating directory for downloaded images: {write_dir}')
            os.makedirs(write_dir)

        for url in image_urls:
            try:
                img = load_image_url(url)
                if img:
                    filename = os.path.split(urlsplit(url)[2])[-1]
                    filepath = os.path.join(write_dir, filename)
                    if not os.path.isfile(filepath):
                        img.save(filepath)
                    else:
                        logging.warning(f'file {filepath} already present')
            except OSError as err:
                logging.error(f'error requesting url: {url}')
                logging.error(err)

    def get_features_from_paths(self, image_paths):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        # TODO add caching for paths like urls
        images_array = np.array((image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths))
        return self.get_features(images_array)

    def get_features_from_urls(self, image_urls):
        ''' takes a list of image urls and returns the features resulting from applying the imagenet model to
        successfully downloaded images along with the urls that were successful.
        '''
        new_urls, cached_urls = partition(lambda x: x in self.cache, image_urls, as_list=True)

        logging.info(f'loading features for {len(cached_urls)} images from cache')
        logging.info(f'computing features for {len(new_urls)} images from urls')

        logging.info(f'getting image arrays from urls')

        if cached_urls:
            cached_image_features = np.array([self.cache[url] for url in cached_urls])

        if new_urls:
            # attempt to download images and convert to constant-size arrays
            new_image_arrays = (image_array_from_url(url, target_size=self.target_size) for url in new_urls)
            # filter out unsuccessful image urls which output None in the list of  # TODO this could probably be optimized
            url_to_image = {url: img for url, img in zip(new_urls, new_image_arrays) if img is not None}
            new_image_arrays = np.array(list(url_to_image.values()))
            new_urls = list(url_to_image.keys())

            print(new_image_arrays.shape)
            logging.info(f'getting features from image arrays')
            new_image_features = self.get_features(new_image_arrays)

            # add new image features to cache
            self.cache.update(zip(new_urls, new_image_features))

        if cached_urls and new_urls:
            # combine results
            image_features = np.vstack((cached_image_features, new_image_features))
            image_urls = cached_urls + new_urls
        elif cached_urls:
            image_features = cached_image_features
            image_urls = cached_urls
        elif new_urls:
            image_features = new_image_features
            image_urls = new_urls
        else:
            logging.warning('no new or cached urls')
            return [], []

        return image_features, image_urls

    def get_features(self, images_array):
        ''' takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array '''
        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')
        logging.info(f'preprocessing {images_array.shape[0]} images')
        images_array = self.preprocess(images_array)
        logging.info(f'computing image features')
        image_features = self.model.predict(images_array)
        if self.n_channels:
            logging.info(f'truncating to first {self.n_channels} channels')
            # if n_channels is specified, only keep that number of channels
            image_features = image_features.T[:self.n_channels].T

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))
