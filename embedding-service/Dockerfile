FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the file that lists the dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app .

# Expose the port the service will run on.clear
# We will choose the port in docker-compose.
EXPOSE 8000

# The command to run when the container launches.
# Note: The EXPERT_ID is NOT defined here. It will be passed in from docker-compose.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]