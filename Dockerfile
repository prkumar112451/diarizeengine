# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install Python and necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python3.10 as python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel cython

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the spaCy English model
RUN pip install numpy==1.26.4 spacy
RUN python -m spacy download en_core_web_sm

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV FLASK_APP=app.py

# Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=80"]
