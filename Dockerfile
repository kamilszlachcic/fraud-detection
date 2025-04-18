# Dockerfile

FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into the container
COPY . .

# Set PYTHONPATH so Python finds src/
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api.main:app", "--host=0.0.0.0", "--port=8000", "--reload"]
