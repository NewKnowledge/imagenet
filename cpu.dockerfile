FROM ubuntu:16.04

ENV HOME=/root
WORKDIR $HOME

RUN apt-get update && apt-get install -y \
    curl \
    build-essential 

# install miniconda
ENV PATH=$HOME/miniconda3/bin:$PATH 
RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > /root/Miniconda3-latest-Linux-x86_64.sh && \
    chmod u+x /root/Miniconda3-latest-Linux-x86_64.sh && \
    /root/Miniconda3-latest-Linux-x86_64.sh -b && \
    rm /root/Miniconda3-latest-Linux-x86_64.sh && \
    conda update -n base conda

# install packages listed in environment.yml including cpu-optimized tensorflow
COPY environment.yml $HOME/
RUN conda env update -f /root/environment.yml && \
    pip install --ignore-installed --no-deps --upgrade https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl

# force dockerfile to download imagenet weights (.h5) into the image to avoid download on spin-up or first use
RUN python -c "from keras.applications.xception import Xception; Xception(weights='imagenet', include_top=False)"

COPY . $HOME/
RUN pip install --no-deps -e .

CMD ["pytest", "--color=yes", "-s", "tests.py"]