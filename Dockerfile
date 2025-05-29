FROM python:3.12.6 

RUN apt update -y && apt install awscli -y 
WORKDIR /app

COPY . /app

RUN pip install requirements.txt

CMD ["python3","app.py"]

