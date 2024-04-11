FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

# Any working directory can be chosen as per choice like '/' or '/home' etc
WORKDIR /home

RUN apt update -y && apt upgrade -y

RUN apt-get -y install python3 python3-venv python3-pip python3-sphinx dotenv strace
RUN apt-get -y install libpq-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev libjpeg-dev zlib1g-dev git

# Install dependencies (install this here instead of in the requirements.txt file to avoid having to install them every time the image is built)
RUN python3 -m pip install \
    numpy \
    pandas \
    pause \
    pika \
    scikit_learn \
    torch \
    coloredlogs\
    pickle\
    nltk\
    urllib3\
    pymisp\
    pytorch-lightning

## install sudo and gosu
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    gosu \
    sudo \
    && rm -rf /var/lib/apt/lists/*


ADD files /home/file

RUN chmod 777 /home/file/entrypoint.sh

# Set the entrypoint to the shell script
ENTRYPOINT ["/home/file/entrypoint.sh"]
