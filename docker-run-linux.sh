docker run -it -P -p 5000:80 \
	--add-host=host.docker.internal:172.17.0.1 \
    -v $(pwd):/app/main \
    image-text-reader:latest \
    python main.py