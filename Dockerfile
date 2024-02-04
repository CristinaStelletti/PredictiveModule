FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN apt-get update && apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN python3 --version
RUN apt-get update && apt-get -y install python3.9
RUN apt-get -y install python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /PredictiveModule

COPY . .

WORKDIR /PredictiveModule/predictionModels

RUN chmod 744 ../config.properties

CMD ["python3", "Forecasting.py"]