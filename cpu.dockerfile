FROM ubuntu:16.04

ENV HOME=/root
WORKDIR $HOME

ENV PATH=$HOME/miniconda3/bin:$PATH \ 
    PYTHON_LIB_PATH=$HOME/miniconda3/lib/python3.6/site-packages \
    PYTHON_BIN_PATH=$HOME/miniconda3/bin/python \
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

# Add the repo for bazel and install it and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openjdk-8-jdk && \
    /bin/bash -c "echo 'deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8' > /etc/apt/sources.list.d/bazel.list" && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \   
    apt-get update && apt-get install -y bazel && \
    rm -rf /var/lib/lists/* && \
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > /root/Miniconda3-latest-Linux-x86_64.sh && \
    chmod u+x /root/Miniconda3-latest-Linux-x86_64.sh && \
    /root/Miniconda3-latest-Linux-x86_64.sh -b && \
    rm /root/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash -c "source activate base" && \
    conda update -n base conda && \
    conda install numpy  
# install numpy first bc we need it to compile tensorflow

RUN git clone https://github.com/tensorflow/tensorflow /root/tensorflow && \
    cd /root/tensorflow; ./configure && \
    bazel build --config=opt --config=mkl //tensorflow/tools/pip_package:build_pip_package && \
    /bin/bash -c "bazel-bin/tensorflow/tools/pip_package/build_pip_package /root/tensorflow/pip/tensorflow_pkg"
# RUN bazel clean, shutdown?

# install remaining packages listed in environment.yml including compiled tensorflow
COPY environment.yml /root/
RUN conda env update -f /root/environment.yml

# force dockerfile to download imagenet weights (.h5) into the image to avoid download on spin-up or first use
RUN python -c "from keras.applications.xception import Xception; Xception(weights='imagenet', include_top=False)"

COPY . $HOME/
RUN pip install -e .

CMD ["pytest", "--color=yes", "-s", "tests.py"]