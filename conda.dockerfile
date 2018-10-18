FROM continuumio/miniconda3:latest

ENV HOME=/app
WORKDIR $HOME

COPY environment.yml $HOME/
RUN conda update -n base conda && \
    conda env update -f /app/environment.yml

# force dockerfile to download imagenet weights (.h5) into the image to avoid download on spin-up or first use
RUN python -c 'from keras.applications.inception_v3 import InceptionV3; InceptionV3()' 

COPY . $HOME/
RUN pip install -e .

CMD ["pytest", "--color=yes", "-s", "tests.py"]