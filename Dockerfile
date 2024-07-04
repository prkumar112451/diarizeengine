# Use an official Python runtime as a parent image
FROM python:3.10

RUN pip install --upgrade pip setuptools wheel cython

# Install Git
RUN apt-get update && apt-get install -y git ffmpeg

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

# Run unicorn when the container launches
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=80"]

