from nk_imagenet import ImagenetModel, image_array_from_path
import numpy as np
from glob import glob
import time

full_model = ImagenetModel()
pool_model = ImagenetModel(pooling='max')


test_urls = ["https://i.redd.it/677lnhjyt5dz.jpg", "https://i.redd.it/c2kkan9uky7z.png",
             "https://i.redd.it/z45uldtv71e01.jpg", "https://i.redd.it/4tsdj4ied2601.jpg"]


def test_features_from_url():
    for model in [full_model, pool_model]:
        # test non-batch method
        features = model.get_features_from_url(test_urls[0])
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[1] > 1
        assert features.shape[0] == 1
        assert str(features.dtype)[:5] == 'float'

        # test non-batch with failing url
        features = model.get_features_from_url('http://www.fake-url.com')
        assert features is None  # should return None


def test_features_from_url_batch():

    # call method with all new, mixed new and cached, all cached urls and a url that will fail
    for urls in [test_urls[:2], test_urls, test_urls + ['http://www.fake-url.com']]:
        for model in [full_model, pool_model]:
            features, urls = model.get_features_from_url_batch(urls)

            assert isinstance(urls, list) or isinstance(urls, tuple)
            assert len(urls) > 0 and isinstance(urls[0], str)

            assert isinstance(features, np.ndarray)
            assert features.ndim == 2
            assert features.shape[0] == len(urls)
            assert features.shape[1] > 1
            assert str(features.dtype)[:5] == 'float'


# def test_featurize_performance():

#     test_images = glob('images/*.jpg')

#     start_time = time.time()
#     image_arrays = np.array([image_array_from_path(img_path) for img_path in test_images])
#     print('time to load images as arrays:', time.time() - start_time)

#     start_time = time.time()
#     feats = full_model.get_features(image_arrays)
#     print('time to get features from image arrays:', time.time() - start_time)

#     start_time = time.time()
#     feats = pool_model.get_features(image_arrays)
#     print('time to get pooled featurize from image arrays:', time.time() - start_time)

# TODO test handling of failed urls
