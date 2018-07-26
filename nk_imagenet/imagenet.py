''' Classes for performing image clustering '''

import logging
import os
import numpy as np
from keras.applications import inception_v3
from urllib.parse import urlsplit
from cachetools import LRUCache

from .image_utils import image_array_from_path, image_array_from_url, load_image_url


class ImagenetModel:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, target_size=(299, 299)):
        # TODO allow for other keras imagenet models
        self.target_size = target_size
        self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        self.preprocess = inception_v3.preprocess_input
        self.decode = inception_v3.decode_predictions
        self.cache = LRUCache(1e4)

    def save_url_images(self, image_urls, write_dir='images'):
        ''' takes a list of urls then downloads and saves the image files to write_dir '''
        if not os.path.isdir(write_dir):
            logging.info(f'creating directory for downloaded images: {write_dir}')
            os.makedirs(write_dir)

        for url in image_urls:
            try:
                img = load_image_url(url)
                if img:
                    # filename = ''.join([ch for ch in url if str.isalnum(ch)]) + '.png'
                    filename = os.path.split(urlsplit(url)[2])[-1]
                    filepath = os.path.join(write_dir, filename)
                    if not os.path.isfile(filepath):
                        img.save(filepath)
                    else:
                        # TODO guarantee unique filename?
                        logging.warning(f'file {filepath} already present')
            except OSError as err:
                logging.error(f'error requesting url: {url}')
                logging.error(err)

    def get_features_from_paths(self, image_paths, n_channels=None):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        images_array = np.array([image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths])
        return self.get_features(images_array, n_channels=n_channels)

    def get_features_from_urls(self, image_urls, n_channels=None):
        ''' takes a list of image urls and returns the features resulting from applying the imagenet model to
        successfully downloaded images along with the urls that were successful.
        '''

        cached_urls = [url for url in image_urls if url in self.cache]
        cached_image_features = np.array([self.cache[url] for url in cached_urls])

        new_urls = set(image_urls).difference(cached_urls)

        logging.info(f'getting {len(cached_urls)} images from cache')
        logging.info(f'getting {len(image_urls)} images from urls')

        new_images_array = [image_array_from_url(url, target_size=self.target_size) for url in new_urls]
        # filter out unsuccessful image urls which output None in the list of
        url_to_image = {url: img for url, img in zip(new_urls, new_images_array) if img is not None}
        new_images_array = np.array(list(url_to_image.values()))
        new_urls = list(url_to_image.keys())

        logging.info(f'getting features from image arrays')
        new_image_features = self.get_features(new_images_array, n_channels=n_channels)

        # add new image features to cache
        self.cache.update(dict(zip(new_urls, new_image_features)))

        # combine results
        image_features = np.vstack((cached_image_features, new_image_features))
        image_urls = cached_urls + new_urls

        return image_features, image_urls

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
