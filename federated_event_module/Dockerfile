FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update -y && apt upgrade -y

RUN apt-get -y install python3 python3-venv python3-pip python3-sphinx dotenv strace
RUN apt-get -y install libpq-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev libjpeg-dev zlib1g-dev git

# Install dependencies (install this here instead of in the requirements.txt file to avoid having to install them every time the image is built)
RUN python3 -m pip install \
    cbor2 \
    numpy \
    pandas \
    pause \
    pika \
    scikit_learn \
    stix2 \
    torch \
    pytorch-lightning

## install sudo and gosu
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    gosu \
    sudo \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the shell script and Python script from the "files" folder into the container
COPY files/entrypoint.sh /app
COPY files/FederatedCTI.py /app


# Set the entrypoint to the shell script
ENTRYPOINT ["/app/entrypoint.sh"]
