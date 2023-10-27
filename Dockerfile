FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    git libglib2.0-0 libsm6 libxrender-dev ffmpeg libsm6 libxext6 &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

RUN pip install torchvision

RUN pip install -U openmim &&\
    mim install mmengine &&\
    mim install mmcv &&\
    mim install mmdet &&\
    mim install mmpose &&\
    pip install mmaction2 &&\
    pip install flask flask_cors flask_restful

RUN pip install pandas openpyxl

# COPY . .

# RUN mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]