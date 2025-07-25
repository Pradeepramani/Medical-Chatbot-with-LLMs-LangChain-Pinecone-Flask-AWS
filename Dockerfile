FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirement.txt

# Env pythonpath = /app

CMD ["python3", "app.py"]
