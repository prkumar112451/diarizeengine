# Use the NVIDIA CUDA base image with Python 3.10, CUDA 11.8, and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set environment variables to configure tzdata non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install necessary tools and dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    git \
    ffmpeg \
    curl \
    build-essential && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel, and cython
RUN pip install --upgrade pip setuptools wheel cython

# Install conda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Clean conda cache non-interactively
RUN /opt/conda/bin/conda clean --all --yes

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH

# Create and activate the whisperx environment
RUN conda create --name whisperx python=3.10 && \
    echo "source activate whisperx" > ~/.bashrc
ENV PATH /opt/conda/envs/whisperx/bin:$PATH
RUN echo "source activate whisperx" > ~/.bashrc

# Install PyTorch with CUDA 11.8 support using pip to avoid conda compatibility issues
RUN pip install torch==2.0.0+cu121 torchaudio==2.0.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install faster-whisper==1.0.0
RUN pip install ctranslate2==4.4.0

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
