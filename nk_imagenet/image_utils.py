''' Utility functions for image processing designed for use with keras networks, including (path or url) -> array functions and caching '''
import io
import logging
import os
import re

import requests
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image


def image_array_from_path(fpath, target_size=(299, 299)):
    img = load_img(fpath, target_size=target_size)
    return img_to_array(img)


def image_array_from_url(url, target_size=(299, 299)):
    try:
        img = load_image_url(url, target_size=target_size)
        return img_to_array(img)
    except Exception as err:
        logging.error(f'\n\nerror reading url:\n {err}')


def strip_alpha_channel(image):
    ''' Strip the alpha channel of an image and fill with fill color '''
    background = Image.new(image.mode[:-1], image.size, '#ffffff')
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


URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def is_url(url):
    ''' takes input string and returns True if string is a url. '''
    return bool(URL_REGEX.match(url))
