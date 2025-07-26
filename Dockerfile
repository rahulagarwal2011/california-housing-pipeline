FROM python:3.9-slim

WORKDIR /app

COPY ./src /app/src
COPY ./model /app/model
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000


CMD ["python", "src/predict.py"]