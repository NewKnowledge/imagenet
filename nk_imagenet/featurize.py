''' Classes for performing image featurization '''

import logging
import os
from PIL import Image

import numpy as np

from keras.applications import inception_v3, mobilenetv2, xception



# NUM_OBJECTS = int(os.ggitenv('NUM_OBJECTS', '5'))

logging.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])
# logging.FileHandler(f"{logPath}/{fileName}.log"),


class Featurize:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, include_top=False, pooling=None, n_channels=None, model='inception_v3', weights = 'imagenet', n_objects=None):

        self.include_top = include_top  # determines if used for classification or featurization, TODO separate into two classes?
        self.n_channels = n_channels
        self.n_objects = n_objects
        self.pooling = pooling
        self.model = model

        self.failed_urls = set()

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

        # NOTE: we force the imagenet model to load in the same scope as the functions using it to avoid tensorflow weirdness
        self.model.predict(np.zeros((1, *self.target_size, 3)))
        logging.info('imagenet loaded')

    def get_features(self, images_array):
        ''' takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array '''
        if self.include_top:
            raise Exception('getting features from a classification model with include_top=True is currently not supported')

        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')

        # preprocess and compute image features
        logging.debug(f'preprocessing {images_array.shape[0]} images')
        images_array = self.preprocess(images_array)
        logging.debug(f'computing image features')
        image_features = self.model.predict(images_array)

        # if n_channels is specified, only keep that number of channels
        if self.n_channels:
            logging.debug(f'truncating to first {self.n_channels} channels')
            image_features = image_features.T[: self.n_channels].T

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))

    def predict(self, images_array):
        ''' alias for get_features to more closely match scikit-learn interface '''
        return self.get_features(images_array)

if __name__ == "__main__":
    image_path = '/home/alexmably/images/'
    images = []
    for r, d, f in os.walk(image_path):
        for file in f:    
            image = np.array(Image.open(os.path.join(r, file)))
            images.append(image)
   
    #image = Image.open(image_path)
    #a = np.array(image)
    im = np.array(images)
    print(im.shape)
    out = Featurize()

    print(out.get_features(im))