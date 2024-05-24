FROM python:3.9-slim
COPY . /app
WORKDIR /app
CMD ["python", "pixy_converter.py"]