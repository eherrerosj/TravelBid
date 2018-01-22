FROM python:3.6.1-slim
RUN apt-get update && apt-get install -y git build-essential

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY app.py utils.py ./
ADD data ./data

ENTRYPOINT ["python", "app.py"]
