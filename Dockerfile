FROM python:3.10

COPY reqirements.txt /app/reqirements.txt
COPY src /app/src
COPY media /app/media

RUN apt-get update
RUN apt-get install sox libsndfile1 ffmpeg -y

RUN pip install -r /app/reqirements.txt

WORKDIR /app/src
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]