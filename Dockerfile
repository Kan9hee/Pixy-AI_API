FROM python:3.8
COPY . /app
RUN pip install --no-cache-dir flask tensorflow nltk
WORKDIR /app
CMD ["python", "pixy_converter.py"]