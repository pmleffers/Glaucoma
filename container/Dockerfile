#-----------------------------------------------------------------------------
# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.
#-----------------------------------------------------------------------------
FROM ubuntu:16.04

MAINTAINER PIETER LEFFERS

# 1. Define the packages required in our environment. 
RUN apt-get -y update && apt-get -y install --no-install-recommends \
    build-essential \
    libopencv-dev \
    libopenblas-dev \
    libjemalloc-dev \
    libgfortran3 \
    python-dev \
    python3-dev \
    unzip \
    python3-pip \
    python-pip \
    wget \
    curl \
    git \
    python \
    python3 \
    nginx \
    ca-certificates && rm -rf /var/lib/apt/lists/

# Download TensorFlow Serving
# https://www.tensorflow.org/serving/setup#installing_the_modelserver
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server

RUN rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/ml/input/data/ /opt/ml/output /opt/ml/model /opt/ml/input/config/ /opt/program
RUN mkdir /root/.keras

COPY glaucoma.py /opt/program
COPY model_folder /opt/program
COPY preprocess.py /opt/program
COPY keras.json /root/.keras/
COPY hyperparameters.json /opt/ml/input/config/
COPY data /tmp/data

RUN chmod +x /opt/program/train

EXPOSE 8888

#-----------------------------------------------------------------------------
# 2. Here we define all python packages we want to include in our environment.
# Pip leaves the install caches populated which uses a significant amount of space. 
# These optimizations save a fair amount of space in the image, which reduces start up time.
#-----------------------------------------------------------------------------

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip3 install cryptography==2.5 oci six Shapely Pillow matplotlib scikit-image \
    jupyter numpy==1.15.0 scipy==1.1.0 scikit-learn==0.19.1 pandas matplotlib xlrd \
    flask gevent gunicorn imageio imgaug tensorflow keras sagemaker-containers 

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

#-----------------------------------------------------------------------------
# 3. Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
#-----------------------------------------------------------------------------
RUN echo 'alias python=python3' >> ~/.bashrc 

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

#-----------------------------------------------------------------------------
# 4. Define the folder where our inference code is located
#-----------------------------------------------------------------------------

WORKDIR /opt/program

# to start notebook inside Container for working
#     docker run -it -p 8888:8888 <image-name>
# inside docker type 
#     jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser .

#Location of hyper parameters in the container: /opt/ml/input/config/hyperparameters.json.
#Location of input data parameters in the container: /opt/ml/input/data. 


#aws ecr get-login --no-include-email
#docker tag keras-glaucoma:latest 160220872109.dkr.ecr.us-east-1.amazonaws.com/keras-glaucoma:latest
#docker push 160220872109.dkr.ecr.us-east-1.amazonaws.com/keras-glaucoma:latest
