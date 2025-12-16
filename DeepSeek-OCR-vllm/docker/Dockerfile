# Use Python 3.12 base image
FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./ZscalerRootCertificate.crt ./

# Upgrade pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --cert ./ZscalerRootCertificate.crt

# Copy requirements files
COPY server/requirements.txt /app/server/requirements.txt
COPY gradio/requirements.txt /app/gradio/requirements.txt

# Install Python dependencies for online OCR mode
# Note: All dependencies are installed to ensure code compatibility,
# but only necessary modules are loaded at runtime based on ONLINE_OCR_MODE configuration.
# In online mode, heavy dependencies like transformers are not actually used.
RUN pip install -r /app/server/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --cert ./ZscalerRootCertificate.crt
RUN pip install -r /app/gradio/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --cert ./ZscalerRootCertificate.crt

# Copy the application code
COPY . /app

# Set working directory
WORKDIR /app

# Expose ports for server (8001) and gradio (7861)
EXPOSE 8001 7861

# Create a startup script for online OCR mode
RUN echo '#!/bin/bash\n\
echo "Starting DeepSeek OCR services in online mode..."\n\
echo "Starting server in background..."\n\
./server/start_server.sh &\n\
SERVER_PID=$!\n\
echo "Starting Gradio interface in background..."\n\
cd gradio && python app.py &\n\
GRADIO_PID=$!\n\
echo "Services started. Server running on port 8001, Gradio on port 7861"\n\
echo "Server PID: $SERVER_PID, Gradio PID: $GRADIO_PID"\n\
wait $SERVER_PID $GRADIO_PID' > /app/start_services_docker.sh

RUN chmod +x /app/start_services_docker.sh

# Set the entrypoint
ENTRYPOINT ["/app/start_services_docker.sh"]