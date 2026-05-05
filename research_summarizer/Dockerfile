# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (espeak is needed for pyttsx3 on Linux)
RUN apt-get update && apt-get install -y \
    build-essential \
    espeak \
    ffmpeg \
    libespeak1 \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir Flask-SQLAlchemy Flask-Login werkzeug gunicorn

# Make port 7860 available to the world outside this container (HF Spaces default)
ENV PORT=7860
ENV FLASK_HOST="0.0.0.0"
EXPOSE 7860

# Run gunicorn when the container launches
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
