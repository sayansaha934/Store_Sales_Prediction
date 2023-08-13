FROM python:3.8-slim

RUN apt-get update

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -r requirements.txt

# Copy your application code
COPY . /app/

EXPOSE 5000

RUN apt install default-jdk -y

# Define the entry point and default command
ENTRYPOINT [ "python" ]
CMD [ "-u", "main.py" ]
