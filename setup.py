from setuptools import setup


setup(
    name="nk_imagenet",
    version="1.1.0",
    description="Interface for using pretrained imagenet models to generate features or object predictions",
    packages=["nk_imagenet"],
    include_package_data=True,
    install_requires=[
        "cachetools >= 2.1.0",
        "Keras == 2.2.4",
        "numpy == 1.15.4",
        "Pillow >= 5.1.0",
        "pytest >= 3.6.2",
        "requests == 2.19.1",
    ],
)
