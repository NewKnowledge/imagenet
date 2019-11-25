from setuptools import setup


setup(name='nk_imagenet',
      version='1.0.1',
      description='Interface for using pretrained imagenet models to generate features or object predictions',
      packages=['nk_imagenet'],
      include_package_data=True,
      install_requires=[
          'cachetools >= 2.1.0',
          'numpy>=1.15.4,<=1.17.3',
          'Pillow >= 5.1.0',
          'pytest >= 3.6.2',
          'requests>=2.19.1,<=2.22.0',
          'tensorflow-gpu == 2.0.0',
      ])
