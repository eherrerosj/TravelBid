FROM python:3.6.1-slim
RUN apt-get update && apt-get install -y git build-essential

EXPOSE 5000

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY app.py utils.py ./
ADD data ./data

CMD ["python", "app.py"]
