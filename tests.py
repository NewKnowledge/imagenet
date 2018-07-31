from nk_imagenet import ImagenetModel, image_array_from_path
import numpy as np
from glob import glob
import time

model = ImagenetModel()


def test_features_from_urls():
    test_urls = ["https://i.redd.it/677lnhjyt5dz.jpg", "https://i.redd.it/c2kkan9uky7z.png",
                 "https://i.redd.it/z45uldtv71e01.jpg", "https://i.redd.it/4tsdj4ied2601.jpg"]

    # call method with all new, mixed new and cached, all cached urls
    for urls in [test_urls[:2], test_urls, test_urls]:
        features, urls = model.get_features_from_urls(urls)
        assert isinstance(urls, list)
        assert len(urls) > 0 and isinstance(urls[0], str)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(urls)
        assert features.shape[1] > 1
        assert str(features.dtype)[:5] == 'float'


def test_featurize_performance():

    test_images = glob('images/*.jpg')

    start_time = time.time()
    image_arrays = np.array([image_array_from_path(img_path) for img_path in test_images])
    print('time to load images as arrays:', time.time() - start_time)

    start_time = time.time()
    feats = model.get_features(image_arrays)
    print('time to get features from image arrays:', time.time() - start_time)

    small_model = ImagenetModel(pooling='max')
    start_time = time.time()
    feats = small_model.get_features(image_arrays)
    print('time to get pooled featurize from image arrays:', time.time() - start_time)


if __name__ == '__main__':
    test_features_from_urls()
