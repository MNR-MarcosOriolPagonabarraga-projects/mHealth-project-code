FROM ml_base:latest

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

