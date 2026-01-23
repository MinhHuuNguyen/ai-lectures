FROM python:3.10

WORKDIR /yolov5_app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Dependencies for opencv
RUN apt-get update --fix-missing && apt-get install ffmpeg libsm6 libxext6 -y

COPY flask_api_server.py .

EXPOSE 1504

CMD python flask_api_server.py
