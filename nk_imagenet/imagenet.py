''' Classes for performing image clustering '''

import os
import pickle
import math
import numpy as np
from cachetools import LRUCache
import tensorflow as tf
from tensorflow.keras.applications import inception_v3, mobilenet_v2, xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical, Sequence

from .utils import image_array_from_path, image_array_from_url, partition
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# NUM_OBJECTS = int(os.ggitenv('NUM_OBJECTS', '5'))

# logging.basicConfig(
#     format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
#     handlers=[
#         logging.StreamHandler()
#     ])
# logging.FileHandler(f"{logPath}/{fileName}.log"),


class ImagenetModel:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, 
        include_top=False, 
        pooling=None, 
        n_channels=None, 
        cache_size=int(1e4), 
        model='inception_v3', 
        weights = 'imagenet', 
        cache_dir=None, 
        n_objects=None):

        self.include_top = include_top  # determines if used for classification or featurization, TODO separate into two classes?
        self.n_channels = n_channels
        self.n_objects = n_objects
        self.pooling = pooling
        self.finetune_model = None # so we know whether finetune model has been created or not

        self.failed_urls = set()

        # NOTE: set cache_dir to None to turn off caching
        if cache_dir:
            # create default cache path in the current file dir w/ filename specifying config
            config = [f'objects-{NUM_OBJECTS}' if include_top else 'features', str(cache_size), model,
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
            self.model = xception.Xception(weights=weights, include_top=include_top, pooling=pooling)
            self.preprocess = xception.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = xception.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (1 if pooling else 10**2)
        elif model == 'inception_v3':
            self.model = inception_v3.InceptionV3(weights=weights, include_top=include_top, pooling=pooling)
            self.preprocess = inception_v3.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = inception_v3.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (1 if pooling else 8**2)
        elif model == 'mobilenet_v2':
            self.model = mobilenetv2.MobileNetV2(weights=weights, include_top=include_top, pooling=pooling)
            self.preprocess = mobilenetv2.preprocess_input
            self.target_size = (244, 244)
            if include_top:
                self.decode = mobilenetv2.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 1280) * (1 if pooling else 7**2)
        else:
            raise Exception('model option not implemented')

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
        ''' detects objects in image provided as an array '''
        logger.debug(f'recognizing objects')
        image_array = self.preprocess(image_array)
        objects = self.model.predict(image_array)
        objects = self.decode(objects, top=self.n_objects)[0]
        return {obj[1]: obj[2] for obj in objects}  # objects = [{'object': obj[1], 'score': obj[2]} for obj in objects]

    def get_features_from_paths(self, image_paths):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        if self.include_top:
            raise Exception('getting features from a classification model with include_top=True is currently not supported')
        # TODO add caching for paths like urls
        images_array = np.array((image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths))
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
        new_urls = image_urls
        cached_urls = []
        # new_urls, cached_urls = partition(lambda x: x in self.cache, image_urls, as_list=True)
        logger.info(f'getting image arrays from {len(image_urls)} urls; \
                     {len(new_urls)} new urls and {len(cached_urls)} cached urls')
        if cached_urls:
            logger.debug(f'loading features for {len(cached_urls)} images from cache')
            if len(cached_urls) == 1:
                cached_image_features = self.cache[cached_urls[0]]
                # print('pre cached dim:', cached_image_features.ndim)
                # if cached_image_features.ndim == 1:
                #     cached_image_features = cached_image_features[None, :]
                # elif cached_image_features.ndim == 3:
                #     assert cached_image_features.shape[:1] == (1, 1)
                #     cached_image_features = cached_image_features[0, :, :]
                # print('post cached dim:', cached_image_features.ndim)
                assert cached_image_features.ndim == 2
            else:
                cached_image_features = np.array([self.cache[url] for url in cached_urls])
                # print('pre cached dim:', cached_image_features.ndim)
                # if cached_image_features.ndim == 1:
                #     cached_image_features = cached_image_features[None, :]
                # elif cached_image_features.ndim == 3:
                #     assert cached_image_features.shape[:1] == (1, 1)
                #     cached_image_features = cached_image_features[0, :, :]
                # print('cached dim:', cached_image_features.ndim)
                assert cached_image_features.ndim == 2
            # print('cached dim:', cached_image_features.ndim)

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
                assert new_image_features.ndim == 2
                logger.debug(f'got features array with shape {new_image_features.shape}')
                # add new image features to cache
                logger.info('saving features to cache')

                self.cache.update(zip(new_urls, new_image_features))

        if cached_urls and new_urls and downloaded_images:
            # print('cached:', cached_image_features.shape)
            # print('new: ', new_image_features.shape)
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

    def get_features(self, images_array):
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

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))

    def predict(self, images_array):
        ''' alias for get_features to more closely match scikit-learn interface '''
        return self.get_features(images_array)

    def create_finetune_model(self, 
                pooling = 'avg',
                dense_dim = 1024,
                nclasses = 2,
        ):
        ''' Creates finetuning Imagenet model with single layer dense classification head 
                :param pooling: pooling to do after ImageNet feature generation, before classification head
                :param dense_dim: dimension of classification head (1 single dense layer)
                :param nclasses: 
        '''
        
        # create new model for finetuned classification
        out = self.model.output
        if self.pooling is None:
            out = GlobalAveragePooling2D()(out) if pooling == 'avg' else GlobalMaxPooling2D()(out)
        dense = Dense(dense_dim, activation='relu')(out)
        preds = Dense(nclasses, activation='softmax')(dense)
        self.finetune_model = Model(inputs = self.model.input, outputs = preds)

    def finetune(self, 
                train_dataset, 
                val_dataset = None, 
                top_layer_epochs = 1,
                unfreeze_proportions = [0.5],
                all_layer_epochs = 5, 
                class_weight = None,
                optimizer_top = 'rmsprop',
                optimizer_full = 'sgd',
                callbacks = None, 
                num_workers = 8
        ):
        ''' Finetunes the Imagenet model iteratively on a smaller set of images with (potentially) a smaller set of classes.
            First finetunes last layer then freezes bottom N layers and retrains the rest
                :param train_dataset: (X, y) pair of tf.constant tensors for training 
                :param val_dataset: (X, y) pair of tf.constant tensors for validation, optional 
                :param top_layer_epochs: how many epochs for which to finetune classification head (happens first)
                :param unfreeze_proportions: list of proportions representing how much of the base ImageNet model one wants to
                    unfreeze (later layers unfrozen) for another round of finetuning
                :param all_layer_epochs: how many epochs for which to finetune entire model (happens second)
                :param class_weight: class weights (used for both training steps)
                :param optimizer_top: optimizer to use for training of classification head
                :param optimizer_full: optimizer to use for training full classification model
                    * suggest to use lower learning rate / more conservative optimizer for this step to 
                      prevent catastrophic forgetting 
                :param callbacks: optional list of callbacks to use for each round of finetuning
                :param num_workers: number of workers to use for multiprocess data loading
        '''
        
        fitting_histories = []

        if self.finetune_model is None:
            raise ValueError("""Trying to fit a finetuning model that hasn't been created yet. Please call 
            create_finetune_model() first""")

        # freeze all convolutional InceptionV3 layers, retrain top layer
        for layer in self.model.layers:
            layer.trainable = False
        self.finetune_model.compile(
            optimizer=optimizer_top, 
            loss='categorical_crossentropy')

        #train_gen = ImageNetGen(images_array, np.array(labels), batch_size=batch_size)
        fitting_histories.append(self.finetune_model.fit(train_dataset, 
            validation_data = val_dataset,
            epochs = top_layer_epochs,
            class_weight = class_weight, 
            shuffle = True, 
            use_multiprocessing = True, 
            workers = num_workers, 
            callbacks = callbacks))

        # iteratively unfreeze specified proportion of later ImageNet base layers and finetune
        self.finetune_model.compile(
            # SGD(lr=0.0001, momentum=0.9)
            optimizer=optimizer_full, 
            loss='categorical_crossentropy')
        for p in unfreeze_proportions:
            freeze_count = int(len(self.model.layers) * p)
            for layer in self.finetune_model.layers[:freeze_count]:
                layer.trainable = False
            for layer in self.finetune_model.layers[freeze_count:]:
                layer.trainable = True

            fitting_histories.append(self.finetune_model.fit(train_dataset, 
                validation_data = val_dataset,
                epochs = all_layer_epochs,
                class_weight = class_weight, 
                shuffle = True, 
                use_multiprocessing = True, 
                workers = num_workers, 
                callbacks = callbacks))
        
        return fitting_histories

    def classify(self, 
        test_dataset, 
        num_workers = 8):
        ''' Uses the finetuned model to predict on a test dataset. 
                :param test_dataset: X, tf.constant tensor for inference
                :param num_workers: number of workers to use for multiprocess data loading
                :return: array of softmaxed prediction probabilities
        '''    

        return self.finetune_model.predict_generator(test_dataset, 
            use_multiprocessing = True, 
            workers = num_workers)

class ImageNetGen(Sequence):
    """ Tf.Keras Sequence for ImageNet input data """

    def __init__(self, X, y = None, batch_size = 32):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.y is None:
            return tf.constant(batch_x)
        else:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return tf.constant(batch_x), tf.constant(batch_y)