
FROM ubuntu:22.04

RUN apt-get update && apt upgrade -y
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

ENV TZ=FR DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3.12 python3-pip ffmpeg libsm6 libxext6 libopenjp2-7-dev libopenjp2-tools openslide-tools

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /repo

ENTRYPOINT ["streamlit"]
CMD ["run","src/app.py"]