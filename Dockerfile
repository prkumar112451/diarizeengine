# Use an official Python runtime as a parent image
FROM python:3.10

# Upgrade pip, setuptools, wheel, and cython
RUN pip install --upgrade pip setuptools wheel cython

# Install Git and ffmpeg
RUN apt-get update && apt-get install -y git ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Reinstall the specific version of faster-whisper after whisperx installation
RUN pip install faster-whisper==1.0.2

# Install the spaCy English model
RUN pip install numpy==1.26.4 spacy
RUN python -m spacy download en_core_web_sm

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV FLASK_APP=app.py

# Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=80"]
