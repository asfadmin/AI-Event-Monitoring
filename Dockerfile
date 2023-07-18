FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3
ENV TZ=America/Anchorage
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git gdal-bin libgdal-dev libx11-dev python3-tk\
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/ml /opt/ml/input /opt/ml/input/data /opt/ml/output /opt/ml/input/config /opt/ml/models

COPY . /opt/ml/code

WORKDIR /opt/ml/code

RUN pip3 install .
ENTRYPOINT ["sagemaker"]