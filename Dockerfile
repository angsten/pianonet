from tensorflow/tensorflow:2.2.0

# docker build -t pianonet_test:v2 .
# docker run -it pianonet_test:v2 /bin/bash
# docker run -p 5000:5000 -i -d pianonet_test:v2

RUN mkdir app

COPY pianonet /app/pianonet
COPY requirements.txt /app/pianonet/requirements.txt

RUN mkdir app/pianonet/models

COPY models/r9p0_3500kparams_approx_9_blocks_model app/pianonet/models/r9p0_3500kparams_approx_9_blocks_model
COPY 1_performance.midi app/pianonet/1_performance.midi

ENV PYTHONPATH=/app

RUN pip install -r /app/pianonet/requirements.txt

EXPOSE 5000

CMD ["python", "/app/pianonet/serving/app.py"]