FROM python:3.11-slim

# Upgrade pip
RUN pip install --upgrade pip==24.3.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libexpat1 \
    libexpat1-dev \
    wget \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Matplotlib to use a writable directory
ENV MPLCONFIGDIR=/tmp/matplotlib

# Install Python dependencies
RUN pip install jupyterlab ipykernel

# Set the working directory
WORKDIR /home

# Install python dependencies 
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Ensure all files in /home are owned by the non-root user
RUN chown -R 1000:1000 /home

# Set the timezone (optional)
ENV TZ=UTC

# Create a custom .bashrc file
RUN echo 'export PS1="\u@\h (\$(date +%H:%M)):\w\$ "' >> /root/.bashrc

CMD ["/bin/bash"]

# docker run -it \
#   -v $(pwd)/:/home/ \
#   --user $(id -u):$(id -g) \
#   stations_optim:latest