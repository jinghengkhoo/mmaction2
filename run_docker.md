docker build -t mmaction2 .
docker run --gpus all --shm-size=8g -p 80:80 mmaction2