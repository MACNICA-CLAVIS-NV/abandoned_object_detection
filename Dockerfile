#
# Dockerfile to build the image of the abandoned_object_detection application
#

ARG BASE_IMAGE=jetpack:r34.1.1
FROM ${BASE_IMAGE}

ARG REPOSITORY_NAME=abandoned_object_detection

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /tmp

RUN apt-get update && apt-get install -y ca-certificates
# COPY  nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# COPY  jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc
RUN apt-get update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        python3-dev \
        build-essential \
        zlib1g-dev \
        zip \
        libjpeg8-dev \
        protobuf-compiler \
        libprotoc-dev \
        cmake && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        setuptools \
        Cython \
        wheel
RUN pip3 install numpy
RUN pip3 install \
        Pillow>=5.2.0 \
        wget>=3.2 \
        pycuda>=2017.1.1 \
        onnx>=1.9.0 \
        paho-mqtt
RUN pip3 uninstall -y protobuf
RUN pip3 install protobuf==3.20.0

RUN mkdir /${REPOSITORY_NAME}
COPY ./ /${REPOSITORY_NAME}

WORKDIR /${REPOSITORY_NAME}/plugins
RUN make

WORKDIR /${REPOSITORY_NAME}/yolo

RUN ./download_yolo.sh
RUN python3 yolo_to_onnx.py -m yolov4-416
RUN python3 onnx_to_tensorrt.py -m yolov4-416

WORKDIR /${REPOSITORY_NAME}
