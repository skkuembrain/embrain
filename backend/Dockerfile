#Dockerfile

FROM python:3.11-slim-buster

WORKDIR /backend
COPY ./requirements.txt ./

EXPOSE 8000

RUN pip3 install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app"] 