FROM ubuntu:16.04

ENV HOME=/root

ENV PYTHON_LIB_PATH=$HOME/anaconda3/lib/python3.6/site-packages \
    PYTHON_BIN_PATH=$HOME/anaconda3/bin/python \
    PYTHONPATH=$HOME/tensorflow/lib \
    PYTHON_ARG=$HOME/tensorflow/lib \
    TF_NEED_GCP=0 \
    TF_NEED_CUDA=0 \ 
    TF_NEED_HDFS=0 \
    TF_NEED_OPENCL=0 \
    TF_NEED_JEMALLOC=1 \
    TF_ENABLE_XLA=0 \
    TF_NEED_VERBS=0 \
    TF_NEED_MKL=1 \
    TF_DOWNLOAD_MKL=1 \
    TF_NEED_MPI=0 \
    TF_NEED_AWS=0 \
    TF_NEED_KAFKA=0 \
    TF_NEED_GDR=0 \
    TF_NEED_OPENCL=0 \
    TF_DOWNLOAD_CLANG=0 \
    TF_SET_ANDROID_WORKSPACE=0 \
    CC_OPT_FLAGS="-march=native" \
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc 


WORKDIR $HOME

# Add a few needed packages to the base Ubuntu 16.04
# OK, maybe *you* don't need emacs :-)
RUN \
    apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openjdk-8-jdk \
    && rm -rf /var/lib/lists/*

# Add the repo for bazel and install it.
# RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" >> /etc/apt/sources.list.d/bazel.list
COPY bazel.list /etc/apt/sources.list.d/
RUN \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y bazel

COPY Anaconda3-5.2.0-Linux-x86_64.sh /root/
RUN \
    cd /root; chmod 755 Anaconda3*.sh && \
    ./Anaconda3*.sh -b && \
    echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> .bashrc && \
    rm -f Anaconda3*.sh

# RUN export PATH="/root/anaconda3/bin:$PATH" 

RUN git clone https://github.com/tensorflow/tensorflow /root/tensorflow

RUN /root/tensorflow/configure

ENV PATH=/root/anaconda3/bin:$PATH

RUN echo $PATH
USER root

RUN /root/anaconda3/bin/conda create -n imagenet && \
    /bin/bash -c "source activate imagenet"
# # RUN bazel clean
RUN cd /root/tensorflow; bazel build --config=opt --config=mkl //tensorflow/tools/pip_package:build_pip_package
RUN cd /root/tensorflow; /bin/bash -c "bazel-bin/tensorflow/tools/pip_package/build_pip_package /root/tensorflow/pip/tensorflow_pkg"
RUN pip install /root/tensorflow/pip/tensorflow_pkg/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl

COPY requirements.txt $HOME/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# # # force dockerfile to download InceptionV3 imagenet weights (.h5) into the image to avoid download on spin-up or first use
RUN python3 -c "from keras.applications.xception import Xception; Xception(weights='imagenet', include_top=False)"
# # RUN python3 -c "from keras.applications.inception_v3 import InceptionV3; InceptionV3(weights='imagenet', include_top=False)"

COPY . $HOME/
RUN pip install -e .

CMD ["pytest", "--color=yes", "-s", "tests.py"]