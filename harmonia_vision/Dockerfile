# Harmonia Vision — Flower server/clients (GPU optional; CPU works for smoke tests)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace/harmonia_vision
ENV PYTHONPATH=/workspace

# Default: overridden in docker-compose for server vs clients
CMD ["python", "-m", "harmonia_vision.server"]
