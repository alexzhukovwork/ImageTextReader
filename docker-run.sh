docker run -it -P -p 5000:80 \
    -v $(pwd):/app/main \
    image-text-reader:latest \
    python main.py