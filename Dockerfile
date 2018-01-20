FROM python:3.6.1-slim
RUN apt-get update && apt-get install -y git build-essential && \
    pip install virtualenv && \
    virtualenv .env

EXPOSE 5000

COPY requirements.txt ./

RUN .env/bin/pip install -r requirements.txt

COPY app.py utils.py ./
COPY ./data ./

CMD ["python", "app.py"]
