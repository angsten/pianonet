from tensorflow/tensorflow:2.2.0

# docker build -t pianonet_test:v2 .
# docker run -p 5000:5000 -i -d pianonet_test:v2

# (To get inside container): docker run -it pianonet_test:v2 /bin/bash

RUN mkdir app

COPY pianonet /app/pianonet
COPY requirements.txt /app/requirements.txt

RUN mkdir app/models

RUN mkdir app/data
RUN mkdir app/data/seeds
RUN mkdir app/data/performances

COPY models/r9p0_3500kparams_approx_9_blocks_model app/models/r9p0_3500kparams_approx_9_blocks_model
COPY models/micro_1 app/models/micro_1

ENV PYTHONPATH=/app

RUN pip install -r /app/requirements.txt

EXPOSE 5000

CMD ["python", "/app/pianonet/serving/app.py"]