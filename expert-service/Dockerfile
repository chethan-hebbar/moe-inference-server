FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
# PYTHONPATH is less critical here but is good practice
ENV PYTHONPATH=/app

# Copy the service-specific requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy this service's entire 'app' directory, which now contains
# its main.py, all model definitions, and weights.
COPY ./app .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]