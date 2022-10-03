FROM ubuntu:latest

ENV TZ=America/Anchorage
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip nvidia-cuda-toolkit git gdal-bin libgdal-dev libx11-dev python3-tk\
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash user

USER user
SHELL ["/bin/bash", "-l", "-c"]

COPY --chown=user . /home/user/AI-Event-Monitoring

WORKDIR /home/user/AI-Event-Monitoring

RUN pip3 install -r requirements.txt
RUN pip3 install ./perlin-numpy