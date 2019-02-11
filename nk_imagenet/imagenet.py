''' Classes for performing image clustering '''
import logging
import os
import pickle

import numpy as np
from cachetools import LRUCache
from keras.applications import inception_v3, mobilenetv2, xception, vgg19
from keras.preprocessing.image import img_to_array

from .utils import image_array_from_path, image_array_from_url, partition

logger = logging.getLogger(__name__)

# TODO set target size indep of model?

MODEL_CLASSES = {
    'xception':  xception,
    'inception_v3': inception_v3,
    'mobilenet_v2': mobilenetv2,
    'vgg19': vgg19,
}


class ImagenetModel:

    def __init__(self, model='inception_v3', include_top=False, pooling=None):
        # set self.model and target_size based on input model string
        if model == 'xception':
            self.model = xception.Xception(weights='imagenet', include_top=include_top, pooling=pooling)
            self.target_size = (299, 299)
        elif model == 'inception_v3':
            self.model = inception_v3.InceptionV3(weights='imagenet', include_top=include_top, pooling=pooling)
            self.target_size = (299, 299)
        elif model == 'mobilenet_v2':
            self.model = mobilenetv2.MobileNetV2(weights='imagenet', include_top=include_top, pooling=pooling)
            self.target_size = (244, 244)
        elif model == 'vgg19':
            self.model = vgg19.VGG19(weights='imagenet', include_top=include_top, pooling=pooling)
            self.target_size = (244, 244)
        else:
            raise Exception('model option not implemented')

        self.preprocess = MODEL_CLASSES[model].preprocess_input

    def predict(self, images_array):
        ''' preprocesses the image and computes the prediction. this function works for both object recognition and featurization '''
        input_dim = images_array.ndim
        if input_dim == 3:
            images_array = images_array[None, :, :, :]  # add a dimension, making batch size 1
        elif input_dim != 4:
            raise Exception(
                'invalid input shape for images_array, expects a 3d (single sample) or 4d (first dim is batch size) array')

        # preprocess
        logger.debug(f'preprocessing {images_array.shape[0]} images')
        images_array = self.preprocess(images_array)

        # compute prediction
        logger.debug(f'computing prediction')
        prediction = self.model.predict(images_array)

        return prediction[0] if input_dim == 3 else prediction


class ImagenetRecognizer(ImagenetModel):

    def __init__(self, model='inception_v3', n_objects=10):
        super().__init__(model=model, include_top=True)

        self.n_objects = n_objects
        self.decode = MODEL_CLASSES[model].decode_predictions

        # NOTE: we force the imagenet model to load in the same scope as the functions using it to avoid tensorflow weirdness
        output = self.model.predict(np.zeros((1, *self.target_size, 3)))
        logger.debug(f'object prediction output: {output}')

    def get_objects(self, images_array):
        ''' provided as an array '''
        ''' takes a batch of images as a 4-d array and returns the detected objects as a list of {object: score} dicts '''
        logger.debug(f'recognizing objects')
        predictions = super().predict(images_array)
        predictions = self.decode(predictions, top=self.n_objects)

        # return list of {object: score} dicts, one dict per image (n_objects total items)
        return [{obj[1]: obj[2] for obj in objects} for objects in predictions]


class ImagenetFeaturizer(ImagenetModel):

    def __init__(self, model='vgg19', include_top=False, pooling='avg', n_channels=None):
        super().__init__(model=model, include_top=include_top, pooling=pooling)
        self.n_channels = n_channels

        # NOTE: we force the imagenet model to load in the same scope as the functions using it to avoid tensorflow weirdness
        output = self.model.predict(np.zeros((1, *self.target_size, 3)))
        # set output shape param for convenient inspection
        self.output_shape = (*output.shape[1:3], n_channels) if n_channels else output.shape[1:]

    @property
    def output_dim(self):
        return np.prod(self.output_shape)

    def get_features(self, images_array, flatten=True):
        ''' takes a batch of images as a 4-d array and returns the imagenet features for those images as a numpy array '''

        # preprocess and compute image features
        image_features = super().predict(images_array)

        # if n_channels is specified, only keep that number of channels
        if self.n_channels is not None:
            logger.debug(f'truncating last dimension of array to only use the first {self.n_channels} channels')
            # transpose to slice the last dim (regardless of input dim), then transpose back
            image_features = image_features.T[:self.n_channels].T

        if flatten:
            # reshape output array by flattening each image into a vector of features
            shape = image_features.shape
            if image_features.ndim == 4:
                image_features = image_features.reshape(shape[0], np.prod(shape[1:]))
            elif image_features.ndim == 3:
                image_features = image_features.reshape(np.prod(shape))

        return image_features


class ImagenetEverything:

    ''' A class for featurizing images using pre-trained neural nets, includes obj recognition and featurization for image arrays, files, or urls. There is an optional url->array cache that is persistent if saved manually. '''

    def __init__(self, model='inception_v3', include_top=False, pooling=None, n_channels=None, n_objects=None, cache_size=int(1e4), cache_dir=None):

        self.include_top = include_top  # determines if used for classification or featurization, TODO separate into two classes?
        self.n_channels = n_channels
        self.n_objects = n_objects

        self.failed_urls = set()

        # NOTE: set cache_dir to None to turn off caching (off by default)
        if cache_dir:
            # create default cache path in the current file dir w/ filename specifying config
            config = [f'objects-{self.n_objects}' if include_top else 'features', str(cache_size), model,
                      pooling if pooling else '', str(n_channels) if n_channels else '']
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
            self.model = xception.Xception(weights='imagenet', include_top=include_top, pooling=pooling)
            self.preprocess = xception.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = xception.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (1 if pooling else 10**2)
        elif model == 'inception_v3':
            self.model = inception_v3.InceptionV3(weights='imagenet', include_top=include_top, pooling=pooling)
            self.preprocess = inception_v3.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = inception_v3.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (1 if pooling else 8**2)
        elif model == 'mobilenet_v2':
            self.model = mobilenetv2.MobileNetV2(weights='imagenet', include_top=include_top, pooling=pooling)
            self.preprocess = mobilenetv2.preprocess_input
            self.target_size = (244, 244)
            if include_top:
                self.decode = mobilenetv2.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 1280) * (1 if pooling else 7**2)
        else:
            raise Exception('model option not implemented')

        # NOTE: we force the imagenet model to load in the same scope as the functions using it to avoid tensorflow weirdness
        self.model.predict(np.zeros((1, *self.target_size, 3)))
        logger.info('imagenet loaded')

    def save_cache(self, cache_path=None):
        ''' saves cache of image identifier (url or path) to image features at the given cache path '''
        logger.info('saving cache')
        cache_path = cache_path if cache_path else self.cache_path
        with open(cache_path, 'wb') as pkl_file:
            pickle.dump({'cache': self.cache, 'failed_urls': self.failed_urls}, pkl_file)

    def load_cache(self, cache_path=None):
        ''' loads cache of image identifier (url or path) to image features '''
        cache_path = cache_path if cache_path else self.cache_path
        logger.info(f'loading cache from {cache_path}')
        if not os.path.isfile(cache_path):
            logger.error(f'cache file not present at: {cache_path}')
        else:
            with open(cache_path, 'rb') as pkl_file:
                pkl_data = pickle.load(pkl_file)
                self.cache = pkl_data['cache']
                self.failed_urls = pkl_data['failed_urls']

            logger.info(f'successfully loaded cache with {len(self.cache)} entries \
                         and failed urls with {len(self.failed_urls)} entries')

    def get_objects_from_url(self, image_url, ignore_failed=True):
        ''' detects objects from image in a url, returns None if url download failed '''
        if image_url not in self.cache:
            # skip if we're ignoring previously failed urls
            if ignore_failed and image_url in self.failed_urls:
                return

            # download image and convert into numpy array
            image_array = image_array_from_url(image_url, target_size=self.target_size)
            if image_array is None:
                # if url request failed, add to failed set
                self.failed_urls.add(image_url)
                return

            # add a dim if needed
            if image_array.ndim == 3:
                image_array = image_array[None, :, :, :]

            # use the imagenet model to detect the objects in the image and add result to cache
            self.cache[image_url] = self.get_objects(image_array)

        # returned cached result
        return self.cache[image_url]

    def get_objects(self, image_array):
        ''' detects objects in an image provided as an array '''
        logger.debug(f'recognizing objects')
        image_array = self.preprocess(image_array)
        objects = self.model.predict(image_array)
        objects = self.decode(objects, top=self.n_objects)[0]
        return {obj[1]: obj[2] for obj in objects}  # {concept: score}

    def get_features_from_paths(self, image_paths):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        if self.include_top:
            raise Exception('getting features from a classification model with include_top=True is currently not supported')
        # TODO add caching for paths like urls
        images_array = np.array([image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths])
        return self.get_features(images_array)

    def get_features_from_url(self, image_url):
        ''' attempt to download the image at the given url, then return the imagenet features if successful, and None if not '''
        if self.include_top:
            raise Exception('getting features from a classification model with include_top=True is currently not supported')

        if image_url not in self.cache:
            image_array = image_array_from_url(image_url, target_size=self.target_size)
            if image_array is None:
                self.failed_urls.add(image_url)
                return
            else:
                if image_array.ndim == 3:
                    image_array = image_array[None, :, :, :]
                self.cache[image_url] = self.get_features(image_array)

        return self.cache.get(image_url)

    def get_features_from_url_batch(self, image_urls, ignore_failed=True):
        ''' takes a list of image urls and returns the features resulting from applying the imagenet model to
        successfully downloaded images along with the urls that were successful. Cached values are used when available
        '''
        if self.include_top:
            raise Exception('getting features from a classification model with include_top=True is currently not supported')
        # split urls into new ones and ones that have cached results
        new_urls, cached_urls = partition(lambda x: x in self.cache, image_urls, as_list=True)
        logger.info(f'getting image arrays from {len(image_urls)} urls; \
                     {len(new_urls)} new urls and {len(cached_urls)} cached urls')
        if cached_urls:
            logger.debug(f'loading features for {len(cached_urls)} images from cache')
            cached_image_features = np.array([self.cache[url] for url in cached_urls])
            assert cached_image_features.ndim == 2

        # remove new urls known to fail
        if new_urls and ignore_failed:
            logger.debug(f'num new urls before dopping fails: {len(new_urls)}')
            new_urls = list(filter(lambda x: x not in self.failed_urls, new_urls))

        if new_urls:
            logger.debug(f'computing features for {len(new_urls)} images from urls')
            # attempt to download images and convert to constant-size arrays  # TODO what to do with failed urls, try again, cache failure?
            new_image_arrays = (image_array_from_url(url, target_size=self.target_size) for url in new_urls)

            # filter out unsuccessful image urls which output None
            failed_images, downloaded_images = partition(
                lambda x: x[1] is not None, zip(new_urls, new_image_arrays), as_list=True)

            logger.debug(f'found {len(failed_images)} failed url images')
            logger.info(f'successfully downloaded {len(downloaded_images)} url images')
            # add failed urls to list
            logger.debug('saving failed urls to failed set')
            self.failed_urls.update(pair[0] for pair in failed_images)
            # downloaded_images = [(url, img) for url, img in zip(new_urls, new_image_arrays) if img is not None]

            if downloaded_images:
                # unzip any successful url, img pairs and convert data types
                new_urls, new_image_arrays = zip(*downloaded_images)
                new_urls = list(new_urls)
                new_image_arrays = np.array(new_image_arrays)

                logger.debug(f'getting features from image arrays with shape {new_image_arrays.shape}')
                new_image_features = self.get_features(new_image_arrays)
                if new_image_features.ndim == 1:
                    # add a dimension to work with cache update zip
                    new_image_features = new_image_features[None, :]
                logger.debug(f'got features array with shape {new_image_features.shape}')
                # add new image features to cache
                logger.info('saving features to cache')
                self.cache.update(zip(new_urls, new_image_features))

        if cached_urls and new_urls and downloaded_images:
            logger.debug('cached and new')
            # combine results
            image_features = np.vstack((cached_image_features, new_image_features))
            image_urls = cached_urls + new_urls
        elif cached_urls:
            logger.debug('cached')
            image_features = cached_image_features
            image_urls = cached_urls
        elif new_urls and downloaded_images:
            logger.debug('new')
            image_features = new_image_features
            image_urls = new_urls
        else:
            logger.debug('no new or cached urls')
            return np.array([[]]), []

        return image_features, image_urls

    def get_features_from_image(self, image_obj, flatten=False):
        image_obj = image_obj.resize(self.target_size)
        img_array = img_to_array(image_obj)
        if img_array.ndim == 3:
            img_array = img_array[None, :, :, :]
        return self.get_features(img_array, flatten=flatten)

    def get_features(self, images_array, flatten=True):
        ''' takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array '''
        if self.include_top:
            raise Exception('getting features from a classification model with include_top=True is currently not supported')

        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')

        # preprocess and compute image features
        logger.debug(f'preprocessing {images_array.shape[0]} images')
        images_array = self.preprocess(images_array)
        logger.debug(f'computing image features')
        image_features = self.model.predict(images_array)

        # if n_channels is specified, only keep that number of channels
        if self.n_channels:
            logger.debug(f'truncating to first {self.n_channels} channels')
            image_features = image_features.T[: self.n_channels].T

        if flatten:
            # reshape output array by flattening each image into a vector of features
            shape = image_features.shape
            if shape[0] == 1:
                return image_features.reshape(np.prod(shape[1:]))
            image_features = image_features.reshape(shape[0], np.prod(shape[1:]))
        return image_features

    def predict(self, images_array):
        ''' alias for get_features or get_objects to more closely match scikit-learn interface '''
        if images_array.ndim == 3:
            images_array = images_array[None, :, :, :]

        if self.include_top:
            return [self.get_objects(img) for img in images_array]

        return self.get_features(images_array)
