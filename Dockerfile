# Use the NVIDIA CUDA base image with Python 3.10, CUDA 11.8, and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

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

# Install PyTorch with CUDA 11.8 support using pip to avoid conda compatibility issues
RUN pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install whisperX
RUN pip install git+https://github.com/m-bain/whisperx.git --upgrade

# Reinstall the specific version of faster-whisper after whisperx installation
RUN pip install faster-whisper==1.0.0

# Install the spaCy English model
RUN python -m spacy download en_core_web_sm

# Set the LD_LIBRARY_PATH to include CUDA and cuDNN
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Enable TensorFloat-32 for improved performance
ENV TORCH_ALLOW_TF32_CUBLAS=1
ENV CUBLAS_WORKSPACE_CONFIG=:16:8

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV FLASK_APP=app.py

# Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=80"]
