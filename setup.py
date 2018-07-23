from setuptools import setup


setup(name='nk_imagenet',
      version='1.0.0',
      description='Interface for using pretrained imagenet models to generate features or object predictions',
      packages=['nk_imagenet'],
      include_package_data=True,
      install_requires=[
          'cachetools >= 2.1.0',
          'Keras >= 2.1.6',
          'numpy >= 1.13.3',
          'Pillow >= 5.1.0',
          'pytest >= 3.6.2',
          'requests >= 2.18.4',
          'tensorflow >= 1.8.0',
      ])
