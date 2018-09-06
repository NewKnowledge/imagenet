FROM continuumio/miniconda3:latest

ENV HOME=/app
WORKDIR $HOME

COPY environment.yml $HOME/
RUN conda update -n base conda && \
    conda env update -f /app/environment.yml

RUN python -c 'from keras.applications.inception_v3 import InceptionV3; InceptionV3()' 

COPY . $HOME/
RUN pip install -e .

CMD ["pytest", "--color=yes", "-s", "tests.py"]