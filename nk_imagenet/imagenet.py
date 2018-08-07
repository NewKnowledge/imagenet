''' Classes for performing image clustering '''

import logging
import os
import pickle

import numpy as np
from cachetools import LRUCache
from keras.applications import inception_v3, mobilenetv2, xception

from .utils import image_array_from_path, image_array_from_url, partition

logging.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
        # logging.FileHandler(f"{logPath}/{fileName}.log"),
    ])


class ImagenetModel:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, pooling=None, n_channels=None, cache_size=int(1e4), model='inception_v3', cache_dir='.'):

        self.n_channels = n_channels
        self.failed_urls = set()
        # NOTE: set cache_dir to None to turn off caching
        if cache_dir:
            # create default cache path in the current file dir w/ filename specifying config
            config = [str(cache_size), model, str(pooling) if pooling else '', str(n_channels) if n_channels else '']
            config_str = '-'.join([c for c in config if c])  # filter out empty strings and join w/ -
            cache_fname = f'imagenet-cache-{config_str}.pkl'
            self.cache_path = os.path.join(cache_dir, cache_fname)
            # TODO allow larger cache_size to still load from previous smaller caches
        else:
            self.cache_path = None

        if self.cache_path and os.path.isfile(self.cache_path):
            self.load_cache()
        else:
            self.cache = LRUCache(cache_size)

        if model == 'xception':
            self.model = xception.Xception(weights='imagenet', include_top=False, pooling=pooling)
            self.preprocess = xception.preprocess_input
            self.decode = xception.decode_predictions
            self.target_size = (299, 299)
            self.output_dim = (n_channels if n_channels else 2048) * (1 if pooling else 10**2)
        elif model == 'inception_v3':
            self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
            self.preprocess = inception_v3.preprocess_input
            self.decode = inception_v3.decode_predictions
            self.target_size = (299, 299)
            self.output_dim = (n_channels if n_channels else 2048) * (1 if pooling else 8**2)
        elif model == 'mobilenet_v2':
            self.model = mobilenetv2.MobileNetV2(weights='imagenet', include_top=False, pooling=pooling)
            self.preprocess = mobilenetv2.preprocess_input
            self.decode = mobilenetv2.decode_predictions
            self.target_size = (244, 244)
            self.output_dim = (n_channels if n_channels else 1280) * (1 if pooling else 7**2)
        else:
            raise Exception('model option not implemented')

        # NOTE: we force the imagenet model to load in the same scope as the functions using it to avoid tensorflow weirdness
        self.model.predict(np.zeros((1, *self.target_size, 3)))
        logging.info('imagenet loaded')

    def save_cache(self, cache_path=None):
        ''' saves cache of image identifier (url or path) to image features at the given cache path '''
        logging.info('saving cache')
        cache_path = cache_path if cache_path else self.cache_path
        with open(cache_path, 'wb') as pkl_file:
            pickle.dump({'cache': self.cache, 'failed_urls': self.failed_urls}, pkl_file)

    def load_cache(self, cache_path=None):
        ''' loads cache of image identifier (url or path) to image features '''
        cache_path = cache_path if cache_path else self.cache_path
        logging.info(f'loading cache from {cache_path}')
        if not os.path.isfile(cache_path):
            logging.error(f'cache file not present at: {cache_path}')
        else:
            with open(cache_path, 'rb') as pkl_file:
                pkl_data = pickle.load(pkl_file)
                self.cache = pkl_data['cache']
                self.failed_urls = pkl_data['failed_urls']

            logging.info(f'successfully loaded cache with {len(self.cache)} entries \
                         and failed urls with {len(self.failed_urls)} entries')

    def get_features_from_paths(self, image_paths):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        # TODO add caching for paths like urls
        images_array = np.array((image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths))
        return self.get_features(images_array)

    def get_features_from_url(self, image_url):
        ''' attempt to download the image at the given url, then return the imagenet features if successful, and None if not '''
        if image_url in self.cache:
            return self.cache[image_url]
        else:
            image_array = image_array_from_url(image_url, target_size=self.target_size)
            if image_array is not None:
                if image_array.ndim == 3:
                    image_array = image_array[None, :, :, :]
                return self.get_features(image_array)

    def get_features_from_url_batch(self, image_urls, ignore_failed=True):
        ''' takes a list of image urls and returns the features resulting from applying the imagenet model to
        successfully downloaded images along with the urls that were successful. Cached values are used when available
        '''
        # split urls into new ones and ones that have cached results
        new_urls, cached_urls = partition(lambda x: x in self.cache, image_urls, as_list=True)
        logging.info(f'getting image arrays from {len(image_urls)} urls')

        if cached_urls:
            logging.info(f'loading features for {len(cached_urls)} images from cache')
            cached_image_features = np.array([self.cache[url] for url in cached_urls])

        # remove new urls known to fail
        if new_urls and ignore_failed:
            new_urls = list(filter(lambda x: x not in self.failed_urls, new_urls))

        if new_urls:
            logging.info(f'computing features for {len(new_urls)} images from urls')
            # attempt to download images and convert to constant-size arrays  # TODO what to do with failed urls, try again, cache failure?
            new_image_arrays = (image_array_from_url(url, target_size=self.target_size) for url in new_urls)

            # filter out unsuccessful image urls which output None
            failed_images, downloaded_images = partition(
                lambda x: x[1] is not None, zip(new_urls, new_image_arrays), as_list=True)
            # add failed urls to list
            self.failed_urls.update(pair[0] for pair in failed_images)
            # downloaded_images = [(url, img) for url, img in zip(new_urls, new_image_arrays) if img is not None]

            if downloaded_images:
                # unzip any successful url, img pairs and convert data types
                new_urls, new_image_arrays = zip(*downloaded_images)
                new_urls = list(new_urls)
                print('new image arrays:', new_image_arrays[:2])
                new_image_arrays = np.array(new_image_arrays)

                logging.debug('getting features from image arrays')
                new_image_features = self.get_features(new_image_arrays)
                # add new image features to cache
                logging.debug('saving features to cache')
                self.cache.update(zip(new_urls, new_image_features))

        if cached_urls and new_urls and downloaded_images:
            logging.debug('cached and new')
            # combine results
            image_features = np.vstack((cached_image_features, new_image_features))
            image_urls = cached_urls + new_urls
        elif cached_urls:
            logging.debug('cached')
            image_features = cached_image_features
            image_urls = cached_urls
        elif new_urls and downloaded_images:
            logging.debug('new')
            image_features = new_image_features
            image_urls = new_urls
        else:
            logging.warning('no new or cached urls')
            return np.array([]), []

        return image_features, image_urls

    def get_features(self, images_array):
        ''' takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array '''
        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')
        logging.debug(f'preprocessing {images_array.shape[0]} images')
        images_array = self.preprocess(images_array)
        logging.debug(f'computing image features')
        image_features = self.model.predict(images_array)
        if self.n_channels:
            logging.debug(f'truncating to first {self.n_channels} channels')
            # if n_channels is specified, only keep that number of channels
            image_features = image_features.T[: self.n_channels].T

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))

    def predict(self, images_array):
        ''' alias for get_features to more closely match scikit-learn interface '''
        return self.get_features(images_array)
