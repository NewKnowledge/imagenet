FROM python:3.6-slim

ENV HOME=/app 

WORKDIR $HOME

COPY requirements.txt $HOME/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# force dockerfile to download InceptionV3 imagenet weights (.h5) into the image to avoid download on spin-up or first use
RUN python -c "from keras.applications.inception_v3 import InceptionV3; InceptionV3(weights='imagenet', include_top=False)"

COPY . $HOME/
RUN pip install -e .

CMD ["pytest", "--color=yes", "-s", "tests.py"]