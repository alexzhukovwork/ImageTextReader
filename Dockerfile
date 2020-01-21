FROM continuumio/miniconda3

WORKDIR /app/main/


RUN  conda install python=3.7 \
  && pip install --no-cache-dir \
      opencv-python \
      tensorflow \
      opencv-contrib-python \
      keras \
      numpy \
      pillow \
      numba \
      flask \
      matplotlib \
  && conda clean -ay


ENV FLASK_APP main.py

EXPOSE 80
