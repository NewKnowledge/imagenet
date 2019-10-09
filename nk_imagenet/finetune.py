import numpy as np
import logging
import os

from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.applications import inception_v3, mobilenetv2, xception
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD
from .utils import image_array_from_path

logging.basicConfig(
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

class Finetuning:

    ''' 
    Functions used to finetune the Imagenet model iteratively on a smaller set of images with (potentially) a smaller set of classes
    The finetuned model is then used to predict on an image array. 
    '''

    def __init__(self, include_top=False, pooling=None, n_channels=None, model='inception_v3', weights = 'imagenet'):

        self.include_top = include_top  # determines if used for classification or featurization, TODO separate into two classes?
        self.n_channels = n_channels
        self.pooling = pooling

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

    def finetune(self, 
                    image_paths, 
                    labels, 
                    pooling = 'avg',
                    nclasses = 2,
                    batch_size = 32, 
                    top_layer_epochs = 1,
                    frozen_layer_count = 249,
                    all_layer_epochs = 5, 
                    class_weight = None,
                    optimizer = 'rmsprop'
            ):
            ''' 
            First finetunes last layer then freezes bottom N layers and retrains the rest
            '''

            # preprocess images
            images_array = np.array([image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths])
            logging.debug(f'preprocessing {images_array.shape[0]} images')
            if images_array.ndim != 4:
                raise Exception('invalid input shape for images_array, expects a 4d array')
            images_array = self.preprocess(images_array)

            # transform labels to categorical variable
            labels = to_categorical(labels)
            
            # create new model for finetuned classification
            out = self.model.output
            if self.pooling is None:
                out = GlobalAveragePooling2D()(out) if pooling == 'avg' else GlobalMaxPooling2D()(out)
            dense = Dense(1024, activation='relu')(out)
            preds = Dense(nclasses, activation='softmax')(dense)
            self.finetune_model = Model(inputs = self.model.input, outputs = preds)

            # freeze all convolutional InceptionV3 layers, retrain top layer
            for layer in self.finetune_model.layers:
                layer.trainable = False
            self.finetune_model.compile(optimizer=optimizer, loss='categorical_crossentropy')
            self.finetune_model.fit(images_array, 
                np.array(labels), 
                batch_size = batch_size,
                epochs = top_layer_epochs,
                class_weight = class_weight)

            # freeze bottom N convolutional layers, retrain top M-N layers (M = total number of layers)
            for layer in self.finetune_model.layers[:frozen_layer_count]:
                layer.trainable = False
            for layer in self.finetune_model.layers[frozen_layer_count:]:
                layer.trainable = True

            # use SGD and low learning rate to prevent catastrophic forgetting in these blocks
            self.finetune_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
            self.finetune_model.fit(images_array, 
                np.array(labels), 
                batch_size = batch_size,
                epochs = all_layer_epochs,
                class_weight = class_weight)

    def finetune_predict(self, images_array):
        ''' 
        Uses the finetuned model to predict on an image array. Returns array of softmax prediction probabilities 
        '''    
        
        # preprocess images
        images_array = np.array([image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths])
        logging.debug(f'preprocessing {images_array.shape[0]} images')
        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')
        images_array = self.preprocess(images_array)

        return self.finetune_model.predict(images_array)

if __name__ == '__main__':
    image_path = '/home/alexmably/images/'
    labels = [0,0,0,1]
    images = []
    for r, d, f in os.walk(image_path):
        image_paths = np.array([os.path.join(r, file) for file in f])
    client = Finetuning()
    client.finetune(image_paths, labels)
    print(client.finetune_predict(image_paths))
