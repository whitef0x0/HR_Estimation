FROM docker-python3-opencv:latest

RUN mkdir /code/

WORKDIR /code/

RUN apt-get update \
  && apt-get install -y vim

RUN pip install -r requirements.txt

CMD ["python", "start.py"]
