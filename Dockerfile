# Set the working directory in the container
WORKDIR /medibot

# Install system dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends git curl unzip \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
COPY . /emart
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose any necessary ports (e.g., FastAPI: 8000)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# docker build -t emart_api .
# docker run -p 8000:8000 -d emart_api