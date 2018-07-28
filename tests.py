from nk_imagenet import ImagenetModel
import numpy as np


def test_features_from_urls():
    test_urls = ["https://i.redd.it/677lnhjyt5dz.jpg", "https://i.redd.it/c2kkan9uky7z.png",
                 "https://i.redd.it/z45uldtv71e01.jpg", "https://i.redd.it/4tsdj4ied2601.jpg"]

    model = ImagenetModel()

    # call method with all new, mixed new and cached, all cached urls
    for urls in [test_urls[:2], test_urls, test_urls]:
        features, urls = model.get_features_from_urls(urls)
        assert isinstance(urls, list)
        assert len(urls) > 0 and isinstance(urls[0], str)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(urls)
        assert features.shape[1] > 1
        assert str(features.dtype)[:5] == 'float'


if __name__ == '__main__':
    test_features_from_urls()
