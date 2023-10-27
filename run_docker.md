docker build -t mmaction2 .
docker run --gpus all --shm-size=8g -p 8000:8000 -v /home/ubuntu/jh/mmaction2:/workspace -d mmaction2