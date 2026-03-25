# Use an official, lightweight Python 3.12 image
FROM python:3.12-slim

# Set environment variables to avoid python writing .pyc files and buffer issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (sometimes required for compiled python packages like numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port Streamlit runs on (Render assigns random ports, but Streamlit binds to the port it's told to)
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
