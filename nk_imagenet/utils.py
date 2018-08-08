''' Utility functions for image processing designed for use with keras networks, including (path or url) -> array functions and caching '''
import io
import logging
import os
from itertools import filterfalse, tee
from urllib.parse import urlsplit

import requests
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image


def partition(pred, iterable, as_list=False):
    'Use a predicate to partition entries into false entries and true entries'
    t1, t2 = tee(iterable)
    if as_list:
        return list(filterfalse(pred, t1)), list(filter(pred, t2))
    return filterfalse(pred, t1), filter(pred, t2)


def image_array_from_path(fpath, target_size=(299, 299)):
    img = load_img(fpath, target_size=target_size)
    return img_to_array(img)


def image_array_from_url(url, target_size=(299, 299)):
    try:
        img = load_image_url(url, target_size=target_size)
        return img_to_array(img)
    except Exception as err:
        logging.warning(f'\nerror reading url:\n {err}')


def strip_alpha_channel(image):
    ''' Strip the alpha channel of an image and fill with fill color '''
    background = Image.new(image.mode[:-1], image.size, '#ffffff')  # TODO is filling with black here a good idea?
    background.paste(image, image.split()[-1])
    return background


def load_image_url(url, target_size=None):
    ''' downloads image at url, fills transparency, convert to jpeg format, and resamples to target size before returning PIL image object '''
    response = requests.get(url)
    with Image.open(io.BytesIO(response.content)) as img:
        # fill transparency if needed
        if img.mode in ('RGBA', 'LA'):
            img = strip_alpha_channel(img)
        # convert to jpeg
        if img.format != 'jpeg':
            img = img.convert('RGB')
        # resample to target size
        if target_size:
            img = img.resize(target_size)  # TODO use interpolation to downsample? (e.g. PIL.Image.LANCZOS)

        return img


def save_url_images(image_urls, write_dir='images'):
    ''' takes a list of urls then downloads and saves the image files to write_dir '''
    if not os.path.isdir(write_dir):
        logging.info(f'creating directory for downloaded images: {write_dir}')
        os.makedirs(write_dir)

    for url in image_urls:
        try:
            img = load_image_url(url)
            if img:
                filename = os.path.split(urlsplit(url)[2])[-1]
                filepath = os.path.join(write_dir, filename)
                if not os.path.isfile(filepath):
                    img.save(filepath)
                else:
                    logging.warning(f'file {filepath} already present')
        except OSError as err:
            logging.warning(f'\nerror requesting url {url}: \n{err}')
