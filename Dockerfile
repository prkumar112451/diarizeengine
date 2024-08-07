# Use the NVIDIA CUDA base image with Python 3.10, CUDA 11.8, and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install Python 3.10, pip, and other required packages
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv && \
    apt-get install -y git ffmpeg curl && \
    apt-get install -y build-essential && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Upgrade pip, setuptools, wheel, and cython
RUN pip install --upgrade pip setuptools wheel cython

# Install conda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH

# Create and activate the whisperx environment
RUN conda create --name whisperx python=3.10 && \
    echo "source activate whisperx" > ~/.bashrc
ENV PATH /opt/conda/envs/whisperx/bin:$PATH
RUN echo "source activate whisperx" > ~/.bashrc

# Install PyTorch with CUDA 11.8 support
RUN conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install whisperX
RUN pip install git+https://github.com/m-bain/whisperx.git --upgrade

# Install the spaCy English model
RUN python -m spacy download en_core_web_sm

# Set the LD_LIBRARY_PATH to point to the virtual environment's local CUDA 11.8 Python lib
ENV LD_LIBRARY_PATH=${PWD}/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV FLASK_APP=app.py

# Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=80"]
