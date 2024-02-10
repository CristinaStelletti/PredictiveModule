FROM nvidia/cuda:11.8.0-base-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y openssh-server curl nano

WORKDIR /root

RUN curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN source .bashrc

ENV PATH=$PATH:/root/miniconda3/bin
RUN source .bashrc

RUN conda install -c conda-forge -y cudatoolkit=11.8.0 cudnn

RUN conda install -c conda-forge -y nvidia/label/cuda-11.8.0::cuda-nvcc

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/' >> .bashrc
RUN source .bashrc

RUN conda init
RUN source .bashrc

RUN mkdir -p /root/miniconda3/etc/conda/activate.d

RUN chsh -s /bin/bash

ENV CONDA_OVERRIDE_CUDA="11.8"

RUN mkdir /root/PredictiveModule

COPY . /root/PredictiveModule

RUN conda env update --file /root/PredictiveModule/requirements.yaml

WORKDIR /root/PredictiveModule/predictionModels

RUN chmod 744 ../config.properties

ENV CUDA_DIR=$CUDA_DIR:/root/miniconda3
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/root/miniconda3"

CMD ["conda", "run", "-n", "base", "python3", "Forecasting.py"]