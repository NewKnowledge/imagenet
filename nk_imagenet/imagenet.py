""" Classes for performing image clustering """

import logging
import os
import pickle

import numpy as np
from cachetools import LRUCache
from keras.applications import inception_v3, mobilenetv2, xception, vgg19, vgg16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

from .utils import image_array_from_path, image_array_from_url, partition

# NUM_OBJECTS = int(os.ggitenv('NUM_OBJECTS', '5'))

logging.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[logging.StreamHandler()],
)
# logging.FileHandler(f"{logPath}/{fileName}.log"),


class ImagenetModel:

    """ A class for featurizing images using pre-trained neural nets """

    def __init__(
        self,
        include_top=False,
        pooling=None,
        n_channels=None,
        cache_size=int(1e4),
        model="inception_v3",
        weights="imagenet",
        cache_dir=None,
        n_objects=None,
    ):

        self.include_top = (
            include_top
        )  # determines if used for classification or featurization, TODO separate into two classes?
        self.n_channels = n_channels
        self.n_objects = n_objects
        self.pooling = pooling

        self.failed_urls = set()

        # NOTE: set cache_dir to None to turn off caching
        if cache_dir:
            # create default cache path in the current file dir w/ filename specifying config
            config = [
                f"objects-{NUM_OBJECTS}" if include_top else "features",
                str(cache_size),
                model,
                pooling if pooling else "",
                str(n_channels) if n_channels else "",
            ]
            config_str = "-".join(
                [c for c in config if c]
            )  # filter out empty strings and join w/ -
            cache_fname = f"imagenet-cache-{config_str}.pkl"
            self.cache_path = os.path.join(cache_dir, cache_fname)
            # TODO allow larger cache_size to still load from previous smaller caches
        else:
            self.cache_path = None

        if self.cache_path and os.path.isfile(self.cache_path):
            self.load_cache()
        else:
            self.cache = LRUCache(cache_size)

        if model == "xception":
            self.model = xception.Xception(
                weights=weights, include_top=include_top, pooling=pooling
            )
            self.preprocess = xception.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = xception.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (
                    1 if pooling else 10 ** 2
                )
        elif model == "inception_v3":
            self.model = inception_v3.InceptionV3(
                weights=weights, include_top=include_top, pooling=pooling
            )
            self.preprocess = inception_v3.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = inception_v3.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (
                    1 if pooling else 8 ** 2
                )
        elif model == "mobilenet_v2":
            self.model = mobilenetv2.MobileNetV2(
                weights=weights, include_top=include_top, pooling=pooling
            )
            self.preprocess = mobilenetv2.preprocess_input
            self.target_size = (244, 244)
            if include_top:
                self.decode = mobilenetv2.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 1280) * (
                    1 if pooling else 7 ** 2
                )
        elif model == "vgg19":
            self.vgg19.VGG19(weights=weights, include_top=include_top, pooling=pooling)
            self.preprocess = vgg19.preprocess_input
            self.target_size = (244, 244)
            if include_top:
                self.decode = vgg19.decode_predictions
            else:
                raise Exception("Feature Output not implemented for VGG19")
        elif model == "vgg16":
            self.vgg16.VGG16(weights=weights, include_top=include_top, pooling=pooling)
            self.preprocess = vgg16.preprocess_input
            self.target_size = (244, 244)
            if include_top:
                self.decode = vgg16.decode_predictions
            else:
                raise Exception("Feature Output not implemented for VGG19")
        else:
            raise Exception("model option not implemented")

        # NOTE: we force the imagenet model to load in the same scope as the functions using it to avoid tensorflow weirdness
        self.model.predict(np.zeros((1, *self.target_size, 3)))
        logging.info("imagenet loaded")

    def save_cache(self, cache_path=None):
        """ saves cache of image identifier (url or path) to image features at the given cache path """
        logging.info("saving cache")
        cache_path = cache_path if cache_path else self.cache_path
        with open(cache_path, "wb") as pkl_file:
            pickle.dump(
                {"cache": self.cache, "failed_urls": self.failed_urls}, pkl_file
            )

    def load_cache(self, cache_path=None):
        """ loads cache of image identifier (url or path) to image features """
        cache_path = cache_path if cache_path else self.cache_path
        logging.info(f"loading cache from {cache_path}")
        if not os.path.isfile(cache_path):
            logging.error(f"cache file not present at: {cache_path}")
        else:
            with open(cache_path, "rb") as pkl_file:
                pkl_data = pickle.load(pkl_file)
                self.cache = pkl_data["cache"]
                self.failed_urls = pkl_data["failed_urls"]

            logging.info(
                f"successfully loaded cache with {len(self.cache)} entries \
                         and failed urls with {len(self.failed_urls)} entries"
            )

    def get_objects_from_url(self, image_url, ignore_failed=True):
        """ detects objects from image in a url, returns None if url download failed """
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
        """ detects objects in image provided as an array """
        logging.debug(f"recognizing objects")
        image_array = self.preprocess(image_array)
        objects = self.model.predict(image_array)
        objects = self.decode(objects, top=self.n_objects)[0]
        return {
            obj[1]: obj[2] for obj in objects
        }  # objects = [{'object': obj[1], 'score': obj[2]} for obj in objects]

    def get_features_from_paths(self, image_paths):
        """ takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images """
        if self.include_top:
            raise Exception(
                "getting features from a classification model with include_top=True is currently not supported"
            )
        # TODO add caching for paths like urls
        images_array = np.array(
            (
                image_array_from_path(fpath, target_size=self.target_size)
                for fpath in image_paths
            )
        )
        return self.get_features(images_array)

    def get_features_from_url(self, image_url):
        """ attempt to download the image at the given url, then return the imagenet features if successful, and None if not """
        if self.include_top:
            raise Exception(
                "getting features from a classification model with include_top=True is currently not supported"
            )

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
        """ takes a list of image urls and returns the features resulting from applying the imagenet model to
        successfully downloaded images along with the urls that were successful. Cached values are used when available
        """
        if self.include_top:
            raise Exception(
                "getting features from a classification model with include_top=True is currently not supported"
            )
        # split urls into new ones and ones that have cached results
        new_urls = image_urls
        cached_urls = []
        # new_urls, cached_urls = partition(lambda x: x in self.cache, image_urls, as_list=True)
        logging.info(
            f"getting image arrays from {len(image_urls)} urls; \
                     {len(new_urls)} new urls and {len(cached_urls)} cached urls"
        )
        if cached_urls:
            logging.debug(f"loading features for {len(cached_urls)} images from cache")
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
                cached_image_features = np.array(
                    [self.cache[url] for url in cached_urls]
                )
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
            logging.debug(f"num new urls before dopping fails: {len(new_urls)}")
            new_urls = list(filter(lambda x: x not in self.failed_urls, new_urls))

        if new_urls:
            logging.debug(f"computing features for {len(new_urls)} images from urls")
            # attempt to download images and convert to constant-size arrays  # TODO what to do with failed urls, try again, cache failure?
            new_image_arrays = (
                image_array_from_url(url, target_size=self.target_size)
                for url in new_urls
            )

            # filter out unsuccessful image urls which output None
            failed_images, downloaded_images = partition(
                lambda x: x[1] is not None,
                zip(new_urls, new_image_arrays),
                as_list=True,
            )

            logging.debug(f"found {len(failed_images)} failed url images")
            logging.info(f"successfully downloaded {len(downloaded_images)} url images")
            # add failed urls to list
            logging.debug("saving failed urls to failed set")
            self.failed_urls.update(pair[0] for pair in failed_images)
            # downloaded_images = [(url, img) for url, img in zip(new_urls, new_image_arrays) if img is not None]

            if downloaded_images:
                # unzip any successful url, img pairs and convert data types
                new_urls, new_image_arrays = zip(*downloaded_images)
                new_urls = list(new_urls)
                new_image_arrays = np.array(new_image_arrays)

                logging.debug(
                    f"getting features from image arrays with shape {new_image_arrays.shape}"
                )
                new_image_features = self.get_features(new_image_arrays)
                assert new_image_features.ndim == 2
                logging.debug(
                    f"got features array with shape {new_image_features.shape}"
                )
                # add new image features to cache
                logging.info("saving features to cache")

                self.cache.update(zip(new_urls, new_image_features))

        if cached_urls and new_urls and downloaded_images:
            # print('cached:', cached_image_features.shape)
            # print('new: ', new_image_features.shape)
            logging.debug("cached and new")
            # combine results
            image_features = np.vstack((cached_image_features, new_image_features))
            image_urls = cached_urls + new_urls
        elif cached_urls:
            logging.debug("cached")
            image_features = cached_image_features
            image_urls = cached_urls
        elif new_urls and downloaded_images:
            logging.debug("new")
            image_features = new_image_features
            image_urls = new_urls
        else:
            logging.debug("no new or cached urls")
            return np.array([[]]), []

        return image_features, image_urls

    def get_features(self, images_array, use_batch=False, batch_size=512):
        """ takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array """

        if images_array.ndim != 4:
            raise Exception("invalid input shape for images_array, expects a 4d array")

        # preprocess and compute image features
        logging.debug(f"preprocessing {images_array.shape[0]} images")
        images_array = self.preprocess(images_array)
        logging.debug(f"computing image features")
        if use_batch:
            image_features = self.model.predict_on_batch(images_array)
        else:
            image_features = self.model.predict(images_array, batch_size=batch_size)
        if self.include_top:
            return self.decode(image_features)

        # if n_channels is specified, only keep that number of channels
        if self.n_channels:
            logging.debug(f"truncating to first {self.n_channels} channels")
            image_features = image_features.T[: self.n_channels].T

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))

    def predict(self, images_array, use_batch=False, batch_size=512):
        """ alias for get_features to more closely match scikit-learn interface """
        return self.get_features(
            images_array, use_batch=use_batch, batch_size=batch_size
        )

    def finetune(
        self,
        image_paths,
        labels,
        pooling="avg",
        nclasses=2,
        batch_size=32,
        top_layer_epochs=1,
        frozen_layer_count=249,
        all_layer_epochs=5,
        class_weight=None,
        optimizer="rmsprop",
    ):
        """ Finetunes the Imagenet model iteratively on a smaller set of images with (potentially) a smaller set of classes.
            First finetunes last layer then freezes bottom N layers and retrains the rest
        """

        # preprocess images
        images_array = np.array(
            [
                image_array_from_path(fpath, target_size=self.target_size)
                for fpath in image_paths
            ]
        )
        logging.debug(f"preprocessing {images_array.shape[0]} images")
        if images_array.ndim != 4:
            raise Exception("invalid input shape for images_array, expects a 4d array")
        images_array = self.preprocess(images_array)

        # transform labels to categorical variable
        labels = to_categorical(labels)

        # create new model for finetuned classification
        out = self.model.output
        if self.pooling is None:
            out = (
                GlobalAveragePooling2D()(out)
                if pooling == "avg"
                else GlobalMaxPooling2D()(out)
            )
        dense = Dense(1024, activation="relu")(out)
        preds = Dense(nclasses, activation="softmax")(dense)
        self.finetune_model = Model(inputs=self.model.input, outputs=preds)

        # freeze all convolutional InceptionV3 layers, retrain top layer
        for layer in self.finetune_model.layers:
            layer.trainable = False
        self.finetune_model.compile(
            optimizer=optimizer, loss="categorical_crossentropy"
        )
        self.finetune_model.fit(
            images_array,
            np.array(labels),
            batch_size=batch_size,
            epochs=top_layer_epochs,
            class_weight=class_weight,
        )

        # freeze bottom N convolutional layers, retrain top M-N layers (M = total number of layers)
        for layer in self.finetune_model.layers[:frozen_layer_count]:
            layer.trainable = False
        for layer in self.finetune_model.layers[frozen_layer_count:]:
            layer.trainable = True

        # use SGD and low learning rate to prevent catastrophic forgetting in these blocks
        self.finetune_model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9), loss="categorical_crossentropy"
        )
        self.finetune_model.fit(
            images_array,
            np.array(labels),
            batch_size=batch_size,
            epochs=all_layer_epochs,
            class_weight=class_weight,
        )

    def finetuned_predict(self, images_array):
        """ Uses the finetuned model to predict on an image array. Returns array of softmax prediction probabilities 
        """

        # preprocess images
        images_array = np.array(
            [
                image_array_from_path(fpath, target_size=self.target_size)
                for fpath in image_paths
            ]
        )
        logging.debug(f"preprocessing {images_array.shape[0]} images")
        if images_array.ndim != 4:
            raise Exception("invalid input shape for images_array, expects a 4d array")
        images_array = self.preprocess(images_array)

        return self.finetune_model.predict(images_array)
